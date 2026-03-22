import logging
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LassoLarsIC, Lasso
import sys
import warnings
from sklearn.exceptions import ConvergenceWarning

# Ensure src is in standard path for execution
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# Suppress convergence warnings from coordinate descent with large feature matrices
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from src.data_ingestion import load_and_merge_zone
from src.features import build_features
from src.preprocessing import chronological_train_val_test_split, scale_data
from src.evaluation.metrics import MAE, sMAPE, rMAE
from src.constants import TARGET_COL

logger = logging.getLogger(__name__)


def predict_naive(
    y_full: pd.Series, test_indices: pd.DatetimeIndex, lag_hours: int = 168
) -> pd.Series:
    """
    Predicts using a direct 168-hour persistence model sequentially mapped exactly mapping
    historical indices natively preventing alignment skew.
    """
    predictions = y_full.shift(lag_hours).reindex(test_indices)
    return predictions


def predict_lear(
    X_full: pd.DataFrame,
    y_full: pd.Series,
    test_indices: pd.DatetimeIndex,
    calibration_window_days: int,
    alpha: float | None = None,
) -> pd.Series:
    """
    Trains 24 independent hourly LassoLarsIC models evaluating a daily moving calibration window.
    Extracts the core logical architecture from Jesus Lago's epftoolbox tracking structural
    mathematics without the excessive computational overhead.
    """
    unique_test_days = np.unique(test_indices.date)
    predictions = pd.Series(index=test_indices, dtype=float, name="LEAR_Predictions")

    # Loop over every unique day in the testing partition
    for current_date in unique_test_days:
        current_date_ts = pd.Timestamp(
            current_date, tz=test_indices.tz if hasattr(test_indices, "tz") else "UTC"
        )
        window_start = current_date_ts - pd.Timedelta(days=calibration_window_days)

        # 1. Geometrically slice strictly prior data bounds safely ensuring 0 leakage
        mask_train = (X_full.index >= window_start) & (X_full.index < current_date_ts)
        X_calib = X_full.loc[mask_train]
        y_calib = y_full.loc[mask_train]

        # Identify the target slices available on the specific test date
        mask_test = test_indices.date == current_date
        day_test_indices = test_indices[mask_test]

        # Subset features
        X_test_day = X_full.loc[day_test_indices]

        # 2. Iterate structurally through the 24 internal electricity models
        for h in range(24):
            # Select samples specifically matching this hour from the calibration window
            hour_mask_calib = X_calib.index.hour == h
            if not hour_mask_calib.any():
                continue

            X_calib_h = X_calib.loc[hour_mask_calib]
            y_calib_h = y_calib.loc[hour_mask_calib]

            # Verify the current testing day actually contains this target hour
            hour_mask_test = X_test_day.index.hour == h
            if not hour_mask_test.any():
                continue

            X_test_h = X_test_day.loc[hour_mask_test]

            # 3. Instantiate, Fit, and Predict isolated LEAR hourly model
            model = (
                LassoLarsIC(criterion="aic", max_iter=10000)
                if alpha is None
                else Lasso(alpha=alpha, max_iter=10000, tol=1e-3, random_state=42)
            )
            try:
                model.fit(X_calib_h, y_calib_h)
                preds = model.predict(X_test_h)
                predictions.loc[X_test_h.index] = preds
            except Exception as e:
                logger.warning(
                    f"Failed LEAR fit Day: {current_date}, Hour: {h}. Reason: {e}"
                )
                # Naive fallback mean assignment
                predictions.loc[X_test_h.index] = y_calib_h.mean()

    return predictions


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    raw_directory = config.get("data", {}).get("raw_dir", "data/raw/auhack_legacy/")
    base_calibration_window = (
        config.get("model_settings", {})
        .get("lear", {})
        .get("calibration_window_days", 182)
    )
    base_lear_alpha = (
        config.get("model_settings", {}).get("lear", {}).get("alpha", None)
    )

    try:
        with open("best_hyperparameters.yaml", "r") as f:
            best_hyperparams = yaml.safe_load(f) or {}
    except FileNotFoundError:
        best_hyperparams = {}

    val_pred_dir = Path("data/outputs/predictions/val")
    test_pred_dir = Path("data/outputs/predictions/test")
    val_pred_dir.mkdir(parents=True, exist_ok=True)
    test_pred_dir.mkdir(parents=True, exist_ok=True)

    target_zones = config.get("data", {}).get("target_zones", ["DE"])
    flow_only_zones = config["data"].get("flow_only_zones", [])
    all_zones = target_zones + flow_only_zones
    raw_data_dict = {z: load_and_merge_zone(z, raw_directory) for z in all_zones}

    missing_targets = [z for z in target_zones if z not in raw_data_dict]
    if missing_targets:
        raise ValueError(
            f"Missing required target zones in preloaded raw_data_dict: {missing_targets}"
        )

    val_preds_dict_naive = {}
    val_preds_dict_lear = {}
    test_preds_dict_naive = {}
    test_preds_dict_lear = {}

    zone_mae_naive = {}
    zone_mae_lear = {}

    for target_zone in target_zones:
        logger.info(f"Establishing Baselines natively for {target_zone}...")

        lear_params_zone = best_hyperparams.get("LEAR", {}).get(target_zone, {})
        calibration_window = int(
            lear_params_zone.get("calibration_window", base_calibration_window)
        )
        alpha_val = lear_params_zone.get("alpha", base_lear_alpha)
        alpha_zone = float(alpha_val) if alpha_val is not None else None

        df, active_features = build_features(
            raw_data_dict, target_zone, lag_actual_flows=True
        )

        logger.info("Executing Train / Val / Test exact chronological splits...")
        train_df, val_df, test_df = chronological_train_val_test_split(
            df, val_ratio=0.15, test_ratio=0.15
        )

        X_train = train_df[active_features]
        y_train = train_df[TARGET_COL]

        X_val = val_df[active_features]
        y_val = val_df[TARGET_COL]

        X_test = test_df[active_features]
        y_test = test_df[TARGET_COL]

        logger.info("Scaling features strictly referencing Training distributions...")
        X_train_s, X_val_s, X_test_s, scaler = scale_data(X_train, X_val, X_test)

        # We reconstruct the sequentially shifted mathematical arrays uniformly allowing specific slices
        X_full = pd.concat([X_train_s, X_val_s, X_test_s]).sort_index()
        y_full = pd.concat([y_train, y_val, y_test]).sort_index()
        val_idx = X_val.index
        test_idx = X_test.index

        logger.info("========================================")
        logger.info(
            f"Targeting Naive (168h Persistence) across {len(test_idx)} points..."
        )
        val_preds_naive = predict_naive(y_full, val_idx, lag_hours=168)
        test_preds_naive = predict_naive(y_full, test_idx, lag_hours=168)

        logger.info(
            f"Establishing LEAR pseudo-online calibrations mapping {calibration_window} days (alpha={alpha_zone})..."
        )
        val_preds_lear = predict_lear(
            X_full,
            y_full,
            val_idx,
            calibration_window_days=calibration_window,
            alpha=alpha_zone,
        )
        test_preds_lear = predict_lear(
            X_full,
            y_full,
            test_idx,
            calibration_window_days=calibration_window,
            alpha=alpha_zone,
        )

        logger.info("========================================")
        logger.info("--------- Evaluation Metrics -----------")

        # Validate mathematical alignment maps cleanly
        valid_naive = ~test_preds_naive.isna()
        y_t_n = y_test.loc[valid_naive]
        p_n = test_preds_naive.loc[valid_naive]

        logger.info(f"[NAIVE 168h Baseline - {target_zone}]")
        mae_naive = MAE(y_t_n, p_n)
        smape_naive = sMAPE(y_t_n, p_n) * 100
        rmae_naive = rMAE(y_t_n, p_n, m="W")
        logger.info(f"  MAE:   {mae_naive:.3f} EUR/MWh")
        logger.info(f"  sMAPE: {smape_naive:.3f} %")
        logger.info(f"  rMAE:  {rmae_naive:.3f}")
        zone_mae_naive[target_zone] = mae_naive

        # Store predictions for this zone
        val_preds_dict_naive[target_zone] = val_preds_naive
        test_preds_dict_naive[target_zone] = test_preds_naive

        logger.info("----------------------------------------")

        # Validate LEAR natively dropping execution gaps dynamically
        valid_lear = ~test_preds_lear.isna()
        y_t_l = y_test.loc[valid_lear]
        p_l = test_preds_lear.loc[valid_lear]

        logger.info(f"[LEAR LassoLarsIC (24-Hour Windows) - {target_zone}]")
        mae_lear = MAE(y_t_l, p_l)
        smape_lear = sMAPE(y_t_l, p_l) * 100
        rmae_lear = rMAE(y_t_l, p_l, m="W")
        logger.info(f"  MAE:   {mae_lear:.3f} EUR/MWh")
        logger.info(f"  sMAPE: {smape_lear:.3f} %")
        logger.info(f"  rMAE:  {rmae_lear:.3f}")
        zone_mae_lear[target_zone] = mae_lear

        # Store predictions for this zone
        val_preds_dict_lear[target_zone] = val_preds_lear
        test_preds_dict_lear[target_zone] = test_preds_lear
        logger.info("========================================")

    # Save multi-zone predictions as DataFrames
    pd.DataFrame(val_preds_dict_naive).to_csv(val_pred_dir / "naive.csv")
    pd.DataFrame(test_preds_dict_naive).to_csv(test_pred_dir / "naive.csv")
    pd.DataFrame(val_preds_dict_lear).to_csv(val_pred_dir / "lear.csv")
    pd.DataFrame(test_preds_dict_lear).to_csv(test_pred_dir / "lear.csv")

    # Macro-averages
    logger.info("\n========== MACRO AVERAGES ACROSS ZONES ==========")
    if zone_mae_naive:
        avg_mae_naive = np.mean(list(zone_mae_naive.values()))
        logger.info(f"[NAIVE 168h Baseline] Avg MAE: {avg_mae_naive:.3f} EUR/MWh")
    if zone_mae_lear:
        avg_mae_lear = np.mean(list(zone_mae_lear.values()))
        logger.info(f"[LEAR LassoLarsIC] Avg MAE: {avg_mae_lear:.3f} EUR/MWh")
    logger.info("================================================")
