import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, LassoLarsIC

from src.constants import TARGET_COLS
from src.data_ingestion import load_competition_data
from src.evaluation.metrics import MAE, RMSE, rMAE, save_metrics_to_csv
from src.features import apply_full_feature_engineering
from src.preprocessing import chronological_train_val_test_split, scale_data

# Suppress convergence warnings from coordinate descent with large feature matrices.
warnings.filterwarnings("ignore", category=ConvergenceWarning)

logger = logging.getLogger(__name__)


def predict_naive(
    y_full: pd.Series, test_indices: pd.DatetimeIndex, lag_hours: int = 168
) -> pd.Series:
    """
    Weekly persistence baseline (T-168h) used as a simple benchmark.
    """
    shifted = y_full.shift(lag_hours)

    # Handle duplicated datetimes (e.g., DST transitions) via datetime+occurrence keys.
    shifted_df = shifted.to_frame("pred")
    shifted_df["datetime"] = shifted_df.index
    shifted_df["occ"] = shifted_df.groupby(level=0).cumcount()
    shifted_keyed = shifted_df.set_index(["datetime", "occ"])["pred"]

    test_idx = pd.DatetimeIndex(test_indices)
    test_df = pd.DataFrame({"datetime": test_idx})
    test_df["occ"] = test_df.groupby("datetime").cumcount()
    test_key = pd.MultiIndex.from_frame(test_df[["datetime", "occ"]])

    preds = shifted_keyed.reindex(test_key)
    preds.index = test_idx
    preds.name = "Naive_Predictions"
    return preds


def predict_lear(
    X_full: pd.DataFrame,
    y_full: pd.Series,
    test_indices: pd.DatetimeIndex,
    calibration_window_days: int,
    alpha: float | None = None,
) -> pd.Series:
    """
    Train 24 independent hourly Lasso models with a daily rolling calibration window.
    """
    test_idx = pd.DatetimeIndex(test_indices)
    unique_test_days = np.unique(test_idx.date)
    predictions = pd.Series(index=test_indices, dtype=float, name="LEAR_Predictions")
    x_idx = pd.DatetimeIndex(X_full.index)
    y_idx = pd.DatetimeIndex(y_full.index)
    last_labeled_ts = y_idx.max() if len(y_idx) > 0 else None

    # Build a one-to-one aligned view of test features using datetime+occurrence keys
    # so duplicated timestamps (DST transitions) remain positionally stable.
    X_keyed = X_full.copy()
    X_keyed["datetime"] = x_idx
    X_keyed["occ"] = X_keyed.groupby(level=0).cumcount()
    X_keyed = X_keyed.set_index(["datetime", "occ"])

    test_map = pd.DataFrame({"datetime": test_idx})
    test_map["occ"] = test_map.groupby("datetime").cumcount()
    test_key = pd.MultiIndex.from_frame(test_map[["datetime", "occ"]])

    X_test_aligned = X_keyed.reindex(test_key)
    X_test_aligned.index = test_idx

    for current_date in unique_test_days:
        current_date_ts = pd.Timestamp(current_date)

        # When predicting beyond the labeled horizon (e.g., Kaggle test), keep
        # using the last available labeled calibration window instead of sliding
        # into an unlabeled period that collapses the calibration set.
        if last_labeled_ts is not None and current_date_ts > last_labeled_ts:
            train_end = last_labeled_ts + pd.Timedelta(hours=1)
        else:
            train_end = current_date_ts
        window_start = train_end - pd.Timedelta(days=calibration_window_days)

        # Strictly prior data for this test date to avoid leakage.
        mask_train_x = (x_idx >= window_start) & (x_idx < train_end)
        mask_train_y = (y_idx >= window_start) & (y_idx < train_end)
        X_calib = X_full.loc[mask_train_x]
        y_calib = y_full.loc[mask_train_y]

        day_mask_test = test_idx.date == current_date
        X_test_day = X_test_aligned.loc[day_mask_test]

        for h in range(24):
            hour_mask_calib = pd.DatetimeIndex(X_calib.index).hour == h
            if not hour_mask_calib.any():
                continue

            X_calib_h = X_calib.loc[hour_mask_calib]
            hour_mask_calib_y = pd.DatetimeIndex(y_calib.index).hour == h
            if not hour_mask_calib_y.any():
                continue
            y_calib_h = y_calib.loc[hour_mask_calib_y]
            common_calib_idx = X_calib_h.index.intersection(y_calib_h.index)
            if common_calib_idx.empty:
                continue
            X_calib_h = X_calib_h.loc[common_calib_idx]
            y_calib_h = y_calib_h.loc[common_calib_idx]

            hour_mask_test_day = pd.DatetimeIndex(X_test_day.index).hour == h
            if not hour_mask_test_day.any():
                continue

            X_test_h = X_test_day.loc[hour_mask_test_day]
            pred_mask = day_mask_test & (test_idx.hour == h)

            # Leak-safe local imputation: fit fill values on calibration slice only.
            if X_calib_h.isna().to_numpy().any() or X_test_h.isna().to_numpy().any():
                X_calib_arr = X_calib_h.to_numpy(dtype=float, copy=True)
                X_test_arr = X_test_h.to_numpy(dtype=float, copy=True)

                # Avoid RuntimeWarning from nanmedian on columns that are entirely NaN.
                all_nan_cols = np.isnan(X_calib_arr).all(axis=0)
                fill_values = np.zeros(X_calib_arr.shape[1], dtype=float)
                valid_cols = ~all_nan_cols
                if valid_cols.any():
                    fill_values[valid_cols] = np.nanmedian(
                        X_calib_arr[:, valid_cols], axis=0
                    )

                nan_mask_calib = np.isnan(X_calib_arr)
                if nan_mask_calib.any():
                    X_calib_arr[nan_mask_calib] = fill_values[
                        np.where(nan_mask_calib)[1]
                    ]

                nan_mask_test = np.isnan(X_test_arr)
                if nan_mask_test.any():
                    X_test_arr[nan_mask_test] = fill_values[np.where(nan_mask_test)[1]]

                X_calib_h = pd.DataFrame(
                    X_calib_arr, index=X_calib_h.index, columns=X_calib_h.columns
                )
                X_test_h = pd.DataFrame(
                    X_test_arr, index=X_test_h.index, columns=X_test_h.columns
                )

            # Drop constant columns to avoid singular design matrices in LARS.
            non_constant_mask = X_calib_h.nunique(dropna=False).to_numpy() > 1
            X_calib_h = X_calib_h.loc[:, non_constant_mask]
            X_test_h = X_test_h.loc[:, non_constant_mask]

            noise_var = (
                float(np.var(y_calib_h))
                if len(y_calib_h) <= X_calib_h.shape[1]
                else None
            )
            primary_model = (
                LassoLarsIC(criterion="aic", max_iter=10000, noise_variance=noise_var)
                if alpha is None
                else Lasso(alpha=alpha, max_iter=10000, tol=1e-3, random_state=42)
            )
            secondary_model = (
                Lasso(alpha=1e-3, max_iter=10000, tol=1e-3, random_state=42)
                if alpha is None
                else None
            )

            try:
                if X_calib_h.empty or X_calib_h.shape[1] == 0:
                    raise ValueError(
                        "No usable features remain after local imputation/pruning"
                    )
                primary_model.fit(X_calib_h, y_calib_h)
                pred_values = np.asarray(primary_model.predict(X_test_h)).reshape(-1)
                if pred_values.size != int(pred_mask.sum()):
                    raise ValueError(
                        f"Prediction length mismatch: pred={pred_values.size}, expected={int(pred_mask.sum())}"
                    )
                predictions.loc[pred_mask] = pred_values
            except Exception as exc:
                if secondary_model is not None:
                    try:
                        secondary_model.fit(X_calib_h, y_calib_h)
                        pred_values = np.asarray(
                            secondary_model.predict(X_test_h)
                        ).reshape(-1)
                        if pred_values.size != int(pred_mask.sum()):
                            raise ValueError(
                                f"Prediction length mismatch: pred={pred_values.size}, expected={int(pred_mask.sum())}"
                            )
                        predictions.loc[pred_mask] = pred_values
                        continue
                    except Exception as exc_secondary:
                        logger.warning(
                            "Failed LEAR fit for %s hour %s. LassoLarsIC error: %s | Lasso fallback error: %s",
                            current_date,
                            h,
                            exc,
                            exc_secondary,
                        )
                else:
                    logger.warning(
                        "Failed LEAR fit for %s hour %s. Reason: %s",
                        current_date,
                        h,
                        exc,
                    )
                predictions.loc[pred_mask] = y_calib_h.mean()

    return predictions


def inverse_transform_target_predictions(
    preds_scaled: pd.Series,
    target_scaler,
    target_cols: list[str],
    target_col: str,
) -> pd.Series:
    """
    Inverse-transform one target from multi-target scaled space.

    RobustScaler inverse_transform expects a 2D array with all target columns,
    so we reconstruct the full target matrix and extract the requested column.
    """
    if target_col not in target_cols:
        raise ValueError(f"Target '{target_col}' not found in fitted target columns")

    target_pos = target_cols.index(target_col)
    valid_mask = ~preds_scaled.isna()
    output = pd.Series(index=preds_scaled.index, dtype=float, name=target_col)

    if valid_mask.any():
        valid_vals = preds_scaled.loc[valid_mask].to_numpy()
        full_target_matrix = np.zeros((valid_vals.shape[0], len(target_cols)))
        full_target_matrix[:, target_pos] = valid_vals
        unscaled = target_scaler.inverse_transform(full_target_matrix)[:, target_pos]
        output.loc[valid_mask] = unscaled

    return output


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    split_cfg = config.get("data", {}).get("splits", {})
    val_start = split_cfg.get("val_start")
    test_start = split_cfg.get("test_start")
    if val_start is None or test_start is None:
        raise ValueError("Missing val_start/test_start in config['data']['splits']")

    lear_cfg = config.get("model_settings", {}).get("lear", {})
    calibration_window = int(
        lear_cfg.get("calibration_window", lear_cfg.get("calibration_window_days", 56))
    )
    alpha_val = lear_cfg.get("alpha", None)
    alpha = float(alpha_val) if alpha_val is not None else None

    val_pred_dir = Path("data/outputs/predictions/val")
    test_pred_dir = Path("data/outputs/predictions/test")
    val_pred_dir.mkdir(parents=True, exist_ok=True)
    test_pred_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading Kaggle training data...")
    df_train = load_competition_data(mode="train")
    df_train = apply_full_feature_engineering(df_train)

    target_cols = [c for c in TARGET_COLS if c in df_train.columns]
    if not target_cols:
        raise ValueError("No configured target columns found in training dataframe")

    feature_cols = [c for c in df_train.columns if c not in target_cols]

    logger.info("Applying chronological train/val/test split...")
    train_df, val_df, test_df = chronological_train_val_test_split(
        df_train,
        val_start=val_start,
        test_start=test_start,
    )
    logger.info(
        "Split sizes (from labeled train data): train=%d, val=%d, test=%d",
        len(train_df),
        len(val_df),
        len(test_df),
    )
    logger.info(
        "Evaluation uses validation split; files in data/outputs/predictions/test/*.csv are Kaggle x_test predictions."
    )

    logger.info("Scaling with RobustScaler fit on train only...")
    train_s, val_s, test_s, _, targ_scaler = scale_data(train_df, val_df, test_df)
    if val_s is None or test_s is None:
        raise ValueError("Validation/test scaled frames are missing")
    val_s_df = val_s
    test_s_df = test_s

    X_full_eval = pd.concat(
        [train_s[feature_cols], val_s_df[feature_cols], test_s_df[feature_cols]]
    ).sort_index()

    val_preds_dict_naive: dict[str, pd.Series] = {}
    val_preds_dict_lear: dict[str, pd.Series] = {}

    logger.info("Running LEAR evaluation per target...")
    # Training-time evaluation is always performed on validation.
    eval_df = val_df
    eval_label = "val"
    for idx in range(len(target_cols)):
        col = target_cols[idx]
        y_full_unscaled = pd.concat([train_df[col], val_df[col]]).sort_index()

        # Naive persistence benchmark in original EUR/MWh space.
        val_preds_naive = predict_naive(
            y_full_unscaled,
            pd.DatetimeIndex(val_df.index),
            lag_hours=168,
        )
        val_preds_dict_naive[col] = val_preds_naive

        val_preds = predict_lear(
            X_full_eval,
            y_full_unscaled,
            pd.DatetimeIndex(val_s_df.index),
            calibration_window_days=calibration_window,
            alpha=alpha,
        )

        val_preds_dict_lear[col] = val_preds

        eval_preds_naive = val_preds_naive
        valid_naive = ~eval_preds_naive.isna()
        y_true_naive = eval_df.loc[valid_naive, col]
        y_pred_naive = eval_preds_naive.loc[valid_naive]

        if y_true_naive.empty:
            logger.warning(
                "Skipping Naive metrics for %s on %s split (no rows)", col, eval_label
            )
        else:
            mae_naive = MAE(y_true_naive, y_pred_naive)
            rmse_naive = RMSE(y_true_naive, y_pred_naive)
            rmae_naive = rMAE(y_true_naive, y_pred_naive, m="W")

            logger.info(
                "[Naive-168h - %s | %s] MAE: %.3f EUR/MWh", col, eval_label, mae_naive
            )
            logger.info(
                "[Naive-168h - %s | %s] RMSE: %.3f EUR/MWh", col, eval_label, rmse_naive
            )
            logger.info(
                "[Naive-168h - %s | %s] rMAE: %.3f", col, eval_label, rmae_naive
            )
            save_metrics_to_csv(
                zone=col,
                model_name=f"Naive 168h ({eval_label})",
                metrics_dict={"MAE": mae_naive, "RMSE": rmse_naive, "rMAE": rmae_naive},
            )

        eval_preds_lear = val_preds
        valid_mask = ~eval_preds_lear.isna()
        y_true = eval_df.loc[valid_mask, col]
        y_pred = eval_preds_lear.loc[valid_mask]

        if y_true.empty:
            logger.warning(
                "Skipping LEAR metrics for %s on %s split (no rows)", col, eval_label
            )
        else:
            mae = MAE(y_true, y_pred)
            rmse = RMSE(y_true, y_pred)
            rmae = rMAE(y_true, y_pred, m="W")

            logger.info("[LEAR - %s | %s] MAE: %.3f EUR/MWh", col, eval_label, mae)
            logger.info("[LEAR - %s | %s] RMSE: %.3f EUR/MWh", col, eval_label, rmse)
            logger.info("[LEAR - %s | %s] rMAE: %.3f", col, eval_label, rmae)
            save_metrics_to_csv(
                zone=col,
                model_name=f"LEAR ({eval_label})",
                metrics_dict={"MAE": mae, "RMSE": rmse, "rMAE": rmae},
            )

    pd.DataFrame(val_preds_dict_naive).to_csv(val_pred_dir / "naive.csv")
    pd.DataFrame(val_preds_dict_lear).to_csv(val_pred_dir / "lear.csv")

    # Kaggle test prediction block: fit on all labeled data and predict x_test.
    try:
        logger.info("Loading Kaggle test features...")
        df_kaggle_test = load_competition_data(mode="test")
        df_kaggle_test = apply_full_feature_engineering(df_kaggle_test)

        missing_test_features = [
            c for c in feature_cols if c not in df_kaggle_test.columns
        ]
        if missing_test_features:
            logger.warning(
                "Adding %s missing test features with ones for schema alignment",
                len(missing_test_features),
            )
            for col in missing_test_features:
                df_kaggle_test[col] = 1.0

        full_train_s, _, kaggle_test_s, _, _ = scale_data(
            df_train,
            None,
            df_kaggle_test,
        )
        if kaggle_test_s is None:
            raise ValueError("Scaled Kaggle test features are missing")

        X_train_full = full_train_s[feature_cols]
        X_test_kaggle = kaggle_test_s[feature_cols]
        X_full_submit = pd.concat([X_train_full, X_test_kaggle]).sort_index()

        test_preds_dict_lear: dict[str, pd.Series] = {}
        for target_col in target_cols:
            y_train_full_raw = df_train[target_col].sort_index()
            test_preds_dict_lear[target_col] = predict_lear(
                X_full_submit,
                y_train_full_raw,
                pd.DatetimeIndex(X_test_kaggle.index),
                calibration_window_days=calibration_window,
                alpha=alpha,
            )

        config_abs_path = Path("config.yaml").resolve()
        raw_dir = config_abs_path.parent / config["data"]["raw_dir"]
        x_test_ids = pd.read_csv(
            raw_dir / config["data"]["test_features_file"], usecols=["id"]
        )["id"]
        if len(x_test_ids) != len(X_test_kaggle):
            raise ValueError(
                "Kaggle id length mismatch between raw x_test and engineered test features"
            )

        lear_test_df = pd.DataFrame(
            {col: test_preds_dict_lear[col].to_numpy() for col in target_cols},
            index=x_test_ids.to_numpy(),
        )
        lear_test_df.index.name = "id"
        lear_test_df.to_csv(test_pred_dir / "lear.csv")
        stale_naive_path = test_pred_dir / "naive.csv"
        if stale_naive_path.exists():
            stale_naive_path.unlink()
            logger.info("Removed stale test prediction file: %s", stale_naive_path)
        logger.info("Saved Kaggle test predictions to %s", test_pred_dir)

    except FileNotFoundError:
        logger.warning(
            "Kaggle test file not found. Skipping test prediction generation."
        )
