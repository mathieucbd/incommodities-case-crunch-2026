import logging
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Ensure src is in standard path for execution
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from lightgbm import LGBMRegressor

# Ingestion & Preprocessing
from src.data_ingestion import load_and_merge_zone
from src.features import create_lags, add_deterministic_features, apply_mad_filter
from src.preprocessing import chronological_train_val_test_split
from src.constants import TARGET_COL

# Evaluation
from src.evaluation.probabilistic import pinball_loss, winkler_score
from src.evaluation.metrics import MAE, save_metrics_to_csv

logger = logging.getLogger(__name__)

INT_PARAMS = [
    "num_leaves",
    "max_depth",
    "n_estimators",
    "min_child_weight",
    "min_samples_leaf",
    "min_samples_split",
    "batch_size",
    "depth",
]


def sanitize_int_params(params: dict) -> dict:
    for param in INT_PARAMS:
        if param in params and params[param] is not None:
            params[param] = int(params[param])
    return params


def train_qra(
    y_true,
    base_preds: dict,
    quantiles=[0.05, 0.5, 0.95],
    params: dict | None = None,
):
    """
    Trains Quantile Regression Averaging (QRA) using LightGBM as the meta-learner.
    Input features are the point forecasts from base models.
    Hyperparameters are read from config.yaml via the `params` dict.
    """
    if params is None:
        params = {}

    X_qra = pd.DataFrame(base_preds)
    seed = params.get("random_state", 42)

    q_models = {}
    for q in quantiles:
        logger.info(f"Training QRA meta-model for quantile: {q}")
        model = LGBMRegressor(
            objective="quantile",
            alpha=q,
            n_estimators=params.get("n_estimators", 500),
            learning_rate=params.get("learning_rate", 0.05),
            num_leaves=params.get("num_leaves", 31),
            min_child_samples=params.get("min_child_samples", 20),
            random_state=seed,
            verbose=-1,
        )
        model.fit(X_qra, y_true)
        q_models[q] = model

    return q_models


def run_ensemble():
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    try:
        with open("best_hyperparameters.yaml", "r") as f:
            best_hyperparams = yaml.safe_load(f) or {}
    except FileNotFoundError:
        best_hyperparams = {}

    raw_directory = config.get("data", {}).get("raw_dir", "data/raw/auhack_legacy/")
    qra_config = config.get("model_settings", {}).get("qra", {})
    quantiles = qra_config.get("quantiles", [0.05, 0.5, 0.95])
    config_qra_params = qra_config.copy()
    config_qra_params.pop("quantiles", None)
    config_qra_params.pop("alpha_winkler", None)
    global_seed = config.get("pipeline", {}).get("global_seed", 42)
    target_zones = config.get("data", {}).get("target_zones", [])

    val_pred_dir = Path("data/outputs/predictions/val")
    test_pred_dir = Path("data/outputs/predictions/test")

    # Macro-tracking dicts
    zone_mae_qra = {}
    zone_winkler_qra = {}
    test_preds_dict_qra = {}

    for target_zone in target_zones:
        # 1. Load & Preprocess Data per zone
        logger.info("========================================")
        logger.info(
            f"Initiating Step 4: Blind Quantile Regression Averaging (QRA) for {target_zone}..."
        )

        df = load_and_merge_zone(target_zone, raw_directory)
        df["Spot_Price_Filtered"] = apply_mad_filter(
            df[TARGET_COL], window="24h", z=3.0
        )
        df = add_deterministic_features(df)

        lag_targets = ["Spot_Price_Filtered", "Residual_Load"]
        lags_list = [24, 48, 168]
        df = create_lags(df, lag_targets, lags_list)

        active_features = ["Hour", "DayOfWeek", "Month"]
        for col in lag_targets:
            for lag in lags_list:
                active_features.append(f"{col}_lag_{lag}")

        df = df.dropna(subset=active_features + [TARGET_COL])

        _, val_df, test_df = chronological_train_val_test_split(
            df, 
            val_start=__import__("yaml").safe_load(open("config.yaml"))["data"]["val_start"], 
            test_start=__import__("yaml").safe_load(open("config.yaml"))["data"]["test_start"]
        )

        y_val_raw = val_df[TARGET_COL]
        y_test_raw = test_df[TARGET_COL]

        def _load_prediction_matrix_zone(folder: Path, zone: str) -> pd.DataFrame:
            files = sorted(folder.glob("*.csv"))
            if not files:
                raise FileNotFoundError(f"No prediction CSV files found in {folder}.")

            series_list = []
            for file_path in files:
                pred_df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                if pred_df.empty:
                    continue
                # Extract ONLY the column corresponding to the current zone
                if zone in pred_df.columns:
                    pred_s = pred_df[zone]
                    pred_s.name = file_path.stem.lower()
                    series_list.append(pred_s)

            if not series_list:
                raise ValueError(
                    f"No usable prediction series found for {zone} in {folder}."
                )

            return pd.concat(series_list, axis=1, sort=False).sort_index()

        logger.info("========================================")
        logger.info(f"Loading base prediction lake from CSVs for {target_zone}...")
        pred_val_all = _load_prediction_matrix_zone(val_pred_dir, target_zone)
        pred_test_all = _load_prediction_matrix_zone(test_pred_dir, target_zone)

        common_models = sorted(
            set(pred_val_all.columns).intersection(pred_test_all.columns)
        )
        if not common_models:
            raise ValueError(
                f"No common model prediction files found between val and test folders for {target_zone}."
            )

        X_qra_val = pred_val_all[common_models].copy()
        X_qra_test = pred_test_all[common_models].copy()

        common_val_idx = X_qra_val.index.intersection(y_val_raw.index)
        common_test_idx = X_qra_test.index.intersection(y_test_raw.index)

        X_qra_val = X_qra_val.loc[common_val_idx].sort_index()
        y_val_aligned = y_val_raw.loc[X_qra_val.index]
        val_mask = ~X_qra_val.isna().any(axis=1)
        X_qra_val = X_qra_val.loc[val_mask]
        y_val_aligned = y_val_aligned.loc[val_mask]

        if X_qra_val.empty:
            raise ValueError(
                f"No aligned validation prediction rows available for QRA training ({target_zone})."
            )

        X_qra_test = X_qra_test.loc[common_test_idx].sort_index()
        y_test_aligned = y_test_raw.loc[X_qra_test.index]
        test_mask = ~X_qra_test.isna().any(axis=1)
        X_qra_test = X_qra_test.loc[test_mask]
        y_test_aligned = y_test_aligned.loc[test_mask]

        if X_qra_test.empty:
            raise ValueError(
                f"No aligned test prediction rows available for QRA evaluation ({target_zone})."
            )

        logger.info(f"Loaded models into QRA: {', '.join(common_models)}")

        # 2. Train QRA
        qra_params_zone = (
            best_hyperparams.get("QRA", {}).get(target_zone, config_qra_params)
            or config_qra_params
        ).copy()
        qra_params_zone = sanitize_int_params(qra_params_zone)
        qra_params_zone["quantiles"] = quantiles
        qra_params_zone["random_state"] = global_seed
        q_models = train_qra(
            y_val_aligned,
            X_qra_val.to_dict(orient="series"),
            quantiles=qra_params_zone["quantiles"],
            params=qra_params_zone,
        )

        # 3. Generate Quantile Predictions on Test Set
        q_results = {}
        for q, model in q_models.items():
            q_results[q] = model.predict(X_qra_test)

        # Prevent crossing (Simple post-processing: enforce sorting)
        q_arr = np.array([q_results[q] for q in sorted(quantiles)])  # (3, N)
        q_arr.sort(axis=0)

        for i, q in enumerate(sorted(quantiles)):
            q_results[q] = q_arr[i]

        # Persist median QRA prediction for this zone in the prediction lake
        test_preds_dict_qra[target_zone] = pd.Series(
            q_results[0.5], index=X_qra_test.index
        )

        # 4. Evaluation
        logger.info("========================================")
        logger.info(f"Ensemble Evaluation (Test Set) - {target_zone}:")

        # Median MAE (q=0.5)
        mae_05 = MAE(y_test_aligned, q_results[0.5])
        logger.info(f"QRA Median (0.50) MAE: {mae_05:.3f} EUR/MWh")
        zone_mae_qra[target_zone] = mae_05

        metrics_to_save = {"MAE": mae_05}

        # Pinball Loss
        for q in quantiles:
            pl = pinball_loss(y_test_aligned, q_results[q], q)
            logger.info(f"Pinball Loss (q={q}): {pl:.3f}")
            metrics_to_save[f"Pinball Loss q={q}"] = pl

        # Winkler Score (0.05 to 0.95 interval)
        if 0.05 in q_results and 0.95 in q_results:
            ws = winkler_score(
                y_test_aligned, q_results[0.05], q_results[0.95], alpha=0.1
            )
            logger.info(f"Winkler Score (90% interval): {ws:.3f}")
            zone_winkler_qra[target_zone] = ws
            metrics_to_save["Winkler Score (90% interval)"] = ws

        save_metrics_to_csv(
            zone=target_zone,
            model_name="QRA",
            metrics_dict=metrics_to_save,
        )

        logger.info("========================================")

    # Macro-averages across zones
    logger.info("\n========== MACRO AVERAGES ACROSS ZONES ==========")
    if zone_mae_qra:
        avg_mae_qra = np.mean(list(zone_mae_qra.values()))
        logger.info(f"[QRA] Avg MAE: {avg_mae_qra:.3f} EUR/MWh")
    if zone_winkler_qra:
        avg_winkler_qra = np.mean(list(zone_winkler_qra.values()))
        logger.info(f"[QRA] Avg Winkler Score: {avg_winkler_qra:.3f}")

    # Save multi-zone QRA median predictions to the Prediction Lake
    pd.DataFrame(test_preds_dict_qra).to_csv(
        "data/outputs/predictions/test/qra_median.csv"
    )
    logger.info("================================================")


if __name__ == "__main__":
    run_ensemble()
