import logging
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import torch

# Ensure src is in standard path for execution
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from lightgbm import LGBMRegressor

# Ingestion & Preprocessing
from src.data_ingestion import load_and_merge_zone
from src.features import create_lags, add_deterministic_features, apply_mad_filter
from src.preprocessing import chronological_train_val_test_split, scale_data
from src.constants import TARGET_COL

# Models
from src.models.tree_models import train_lightgbm, train_xgboost, train_catboost
from src.models.deep_learning import (
    EPFMultivariateDNN,
    reshape_to_daily,
    train_pytorch_dnn,
)

# Evaluation
from src.evaluation.probabilistic import pinball_loss, winkler_score
from src.evaluation.metrics import MAE, sMAPE, rMAE

logger = logging.getLogger(__name__)


def train_qra(
    y_true, base_preds: dict, quantiles=[0.05, 0.5, 0.95], params: dict = None
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

    raw_directory = config.get("data", {}).get("raw_dir", "data/raw/auhack_legacy/")
    use_data_augmentation = (
        config.get("model_settings", {})
        .get("dnn", {})
        .get("use_data_augmentation", True)
    )

    # 1. Load & Preprocess Data for DE
    target_zone = "DE"
    logger.info("========================================")
    logger.info(
        f"Initiating Step 4: Quantile Regression Averaging (QRA) for {target_zone}..."
    )

    df = load_and_merge_zone(target_zone, raw_directory)
    df["Spot_Price_Filtered"] = apply_mad_filter(df[TARGET_COL], window="24h", z=3.0)
    df = add_deterministic_features(df)

    lag_targets = ["Spot_Price_Filtered", "Residual_Load"]
    lags_list = [24, 48, 168]
    df = create_lags(df, lag_targets, lags_list)

    active_features = ["Hour", "DayOfWeek", "Month"]
    for col in lag_targets:
        for lag in lags_list:
            active_features.append(f"{col}_lag_{lag}")

    df = df.dropna(subset=active_features + [TARGET_COL])

    train_df, val_df, test_df = chronological_train_val_test_split(
        df, val_ratio=0.15, test_ratio=0.15
    )

    X_train_raw = train_df[active_features]
    y_train_raw = train_df[TARGET_COL]
    X_val_raw = val_df[active_features]
    y_val_raw = val_df[TARGET_COL]
    X_test_raw = test_df[active_features]
    y_test_raw = test_df[TARGET_COL]

    # 2. Get Optimal Params from Config
    trees_config = config.get("model_settings", {}).get("trees", {})
    dnn_config = config.get("model_settings", {}).get("dnn", {})
    qra_config = config.get("model_settings", {}).get("qra", {})
    quantiles = qra_config.get("quantiles", [0.05, 0.5, 0.95])

    # 3. Train Base Models and Generate Validation & Test Predictions
    logger.info("========================================")
    logger.info("Training Base Models...")

    # LightGBM
    logger.info("--> Building LightGBM (1000 trees)...")
    lgb_params = trees_config.get("lgb", {}).copy()
    lgb_params["early_stopping_rounds"] = 50
    model_lgb = train_lightgbm(
        X_train_raw, y_train_raw, X_val_raw, y_val_raw, params=lgb_params
    )
    val_preds_lgb = model_lgb.predict(X_val_raw)
    test_preds_lgb = model_lgb.predict(X_test_raw)

    # XGBoost
    logger.info("--> Building XGBoost (1500 trees)...")
    xgb_params = trees_config.get("xgb", {}).copy()
    xgb_params["early_stopping_rounds"] = 50
    model_xgb = train_xgboost(
        X_train_raw, y_train_raw, X_val_raw, y_val_raw, params=xgb_params
    )
    val_preds_xgb = model_xgb.predict(X_val_raw)
    test_preds_xgb = model_xgb.predict(X_test_raw)

    # CatBoost
    logger.info(
        "--> Building CatBoost (1500 deep trees, this will take a few minutes)..."
    )
    cat_params = trees_config.get("cat", {}).copy()
    cat_params["early_stopping_rounds"] = 50
    cat_params["train_dir"] = "data/outputs/catboost_info"
    model_cat = train_catboost(
        X_train_raw, y_train_raw, X_val_raw, y_val_raw, params=cat_params
    )
    val_preds_cat = model_cat.predict(X_val_raw)
    test_preds_cat = model_cat.predict(X_test_raw)

    # DNN
    logger.info("--> Training Multivariate PyTorch DNN (150 Epochs limit)...")
    X_train_s, X_val_s, X_test_s, _ = scale_data(X_train_raw, X_val_raw, X_test_raw)
    y_train_s_df, y_val_s_df, y_test_s_df, y_scaler = scale_data(
        y_train_raw.to_frame(), y_val_raw.to_frame(), y_test_raw.to_frame()
    )
    y_train_s = y_train_s_df[TARGET_COL]
    y_val_s = y_val_s_df[TARGET_COL]
    y_test_s = y_test_s_df[TARGET_COL]

    X_train_d, y_train_d = reshape_to_daily(
        X_train_s, y_train_s, augment=use_data_augmentation
    )
    X_val_d, y_val_d = reshape_to_daily(X_val_s, y_val_s, augment=False)
    X_test_d, y_test_d = reshape_to_daily(X_test_s, y_test_s, augment=False)

    dnn_params = {
        "lr": dnn_config.get("learning_rate", 0.001),
        "dropout_rate": dnn_config.get("dropout_rate", 0.2),
        "epochs": dnn_config.get("epochs", 150),
        "batch_size": dnn_config.get("batch_size", 64),
        "patience": dnn_config.get("patience", 15),
    }
    model_dnn, device = train_pytorch_dnn(
        X_train_d, y_train_d, X_val_d, y_val_d, params=dnn_params
    )

    def get_dnn_preds(model, X_daily, y_scaler):
        model.eval()
        with torch.no_grad():
            preds = model(torch.tensor(X_daily).to(device)).cpu().numpy().flatten()
        return y_scaler.inverse_transform(preds.reshape(-1, 1)).flatten()

    test_preds_dnn_full = get_dnn_preds(model_dnn, X_test_d, y_scaler)
    val_preds_dnn_full = get_dnn_preds(model_dnn, X_val_d, y_scaler)

    # We must align the hourly predictions to the days that were NOT dropped in DNN reshaping
    # This alignment is tricky. For simplicity, let's create aligned DataFrames.

    def align_preds(df_raw, preds_flattened):
        df_copy = pd.DataFrame(
            {"Target": df_raw.values, "Date": df_raw.index.date}, index=df_raw.index
        )
        valid_indices = []
        for date, group in df_copy.groupby("Date"):
            if len(group) == 24:
                valid_indices.extend(group.index)
        return pd.Series(preds_flattened, index=valid_indices)

    val_preds_dnn = align_preds(y_val_raw, val_preds_dnn_full)
    test_preds_dnn = align_preds(y_test_raw, test_preds_dnn_full)

    # Now align all base models to the DNN valid indices to have a square matrix for QRA
    common_val_idx = val_preds_dnn.index
    common_test_idx = test_preds_dnn.index

    val_base_preds = {
        "LGBM": pd.Series(val_preds_lgb, index=y_val_raw.index).loc[common_val_idx],
        "XGB": pd.Series(val_preds_xgb, index=y_val_raw.index).loc[common_val_idx],
        "Cat": pd.Series(val_preds_cat, index=y_val_raw.index).loc[common_val_idx],
        "DNN": val_preds_dnn,
    }

    test_base_preds = {
        "LGBM": pd.Series(test_preds_lgb, index=y_test_raw.index).loc[common_test_idx],
        "XGB": pd.Series(test_preds_xgb, index=y_test_raw.index).loc[common_test_idx],
        "Cat": pd.Series(test_preds_cat, index=y_test_raw.index).loc[common_test_idx],
        "DNN": test_preds_dnn,
    }

    y_val_aligned = y_val_raw.loc[common_val_idx]
    y_test_aligned = y_test_raw.loc[common_test_idx]

    # 4. Train QRA (Params from config)
    qra_params = qra_config.copy()
    qra_params.pop("quantiles", None)
    qra_params.pop("alpha_winkler", None)
    qra_params["random_state"] = config.get("pipeline", {}).get("global_seed", 42)
    q_models = train_qra(
        y_val_aligned, val_base_preds, quantiles=quantiles, params=qra_params
    )

    # 5. Generate Quantile Predictions on Test Set
    X_test_qra = pd.DataFrame(test_base_preds)
    q_results = {}
    for q, model in q_models.items():
        q_results[q] = model.predict(X_test_qra)

    # Prevent crossing (Simple post-processing: enforce sorting)
    q_arr = np.array([q_results[q] for q in sorted(quantiles)])  # (3, N)
    q_arr.sort(axis=0)

    for i, q in enumerate(sorted(quantiles)):
        q_results[q] = q_arr[i]

    # 6. Evaluation
    logger.info("========================================")
    logger.info("Ensemble Evaluation (Test Set):")

    # Median MAE (q=0.5)
    mae_05 = MAE(y_test_aligned, q_results[0.5])
    logger.info(f"QRA Median (0.50) MAE: {mae_05:.3f} EUR/MWh")

    # Pinball Loss
    for q in quantiles:
        pl = pinball_loss(y_test_aligned, q_results[q], q)
        logger.info(f"Pinball Loss (q={q}): {pl:.3f}")

    # Winkler Score (0.05 to 0.95 interval)
    if 0.05 in q_results and 0.95 in q_results:
        ws = winkler_score(y_test_aligned, q_results[0.05], q_results[0.95], alpha=0.1)
        logger.info(f"Winkler Score (90% interval): {ws:.3f}")

    logger.info("========================================")


if __name__ == "__main__":
    run_ensemble()
