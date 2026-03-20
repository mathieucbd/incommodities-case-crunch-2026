"""
Step 5: Statistical Significance Evaluation.
Runs the Diebold-Mariano (DM) test per hour to prove the QRA Ensemble
statistically outperforms the LEAR baseline.
"""

import logging
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.data_ingestion import load_and_merge_zone
from src.features import create_lags, add_deterministic_features, apply_mad_filter
from src.preprocessing import chronological_train_val_test_split, scale_data
from src.constants import TARGET_COL
from src.evaluation.metrics import MAE, sMAPE, rMAE

from src.models.baselines import predict_lear
from src.models.tree_models import train_lightgbm, train_xgboost, train_catboost
from src.models.deep_learning import reshape_to_daily, train_pytorch_dnn
from src.models.ensembles import train_qra

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DM Test (1D, hourly-series compatible)
# ---------------------------------------------------------------------------


def dm_test(actual: np.ndarray, pred1: np.ndarray, pred2: np.ndarray, h: int = 1):
    """
    Author: Jesus Lago (Adapted for standalone pipeline)
    One-sided Diebold-Mariano test: H1 that pred2 is MORE accurate than pred1.

    .. math::
        d_t = |e_{1,t}| - |e_{2,t}|, \\text{ then } DM = \\bar{d} / \\sqrt{\\hat{V}(\\bar{d})/T}

    Parameters
    ----------
    actual : array-like
        The actual prices.
    pred1 : array-like
        Model 1 predictions (the baseline — LEAR).
    pred2 : array-like
        Model 2 predictions (the challenger — QRA).
    h : int
        Forecast horizon for variance correction (1 for one-step-ahead).

    Returns
    -------
    float, float
        DM statistic and one-sided p-value (lower = more significant).
    """
    actual = np.array(actual).flatten()
    pred1 = np.array(pred1).flatten()
    pred2 = np.array(pred2).flatten()

    e1 = actual - pred1
    e2 = actual - pred2

    # Loss differential: positive d means pred1 is worse
    d = np.abs(e1) - np.abs(e2)

    T = len(d)
    d_bar = np.mean(d)

    # HAC variance of the loss differential (Newey-West with lag h-1)
    gamma0 = np.var(d, ddof=1)
    gamma_sum = 0.0
    for k in range(1, h):
        gamma_k = np.mean((d[k:] - d_bar) * (d[:-k] - d_bar))
        gamma_sum += (1 - k / h) * gamma_k

    var_d = (gamma0 + 2 * gamma_sum) / T
    if var_d <= 0:
        return np.nan, np.nan

    dm_stat = d_bar / np.sqrt(var_d)
    # One-sided: H1 pred2 is better, so we want large positive DM stat
    p_value = 1 - stats.norm.cdf(dm_stat)

    return float(dm_stat), float(p_value)


# ---------------------------------------------------------------------------
# Data loading helper (shared setup)
# ---------------------------------------------------------------------------


def _load_pipeline_data(config):
    raw_dir = config["data"]["raw_dir"]
    df = load_and_merge_zone("DE", raw_dir)
    df["Spot_Price_Filtered"] = apply_mad_filter(df[TARGET_COL], window="24h", z=3.0)
    df = add_deterministic_features(df)

    lag_targets = ["Spot_Price_Filtered", "Residual_Load"]
    lags_list = [24, 48, 168]
    df = create_lags(df, lag_targets, lags_list)

    features = ["Hour", "DayOfWeek", "Month"] + [
        f"{c}_lag_{l}" for c in lag_targets for l in lags_list
    ]
    df = df.dropna(subset=features + [TARGET_COL])
    return df, features


# ---------------------------------------------------------------------------
# Execution block
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    seed = config.get("pipeline", {}).get("global_seed", 42)
    aug = (
        config.get("model_settings", {})
        .get("dnn", {})
        .get("use_data_augmentation", True)
    )
    trees_c = config.get("model_settings", {}).get("trees", {})
    dnn_c = config.get("model_settings", {}).get("dnn", {})
    qra_c = config.get("model_settings", {}).get("qra", {})
    quantiles = qra_c.get("quantiles", [0.05, 0.5, 0.95])

    # -----------------------------------------------------------------------
    # 1. Load data
    # -----------------------------------------------------------------------
    logger.info("Loading and engineering features...")
    df, features = _load_pipeline_data(config)

    train_df, val_df, test_df = chronological_train_val_test_split(
        df, val_ratio=0.15, test_ratio=0.15
    )

    X_train = train_df[features]
    y_train = train_df[TARGET_COL]
    X_val = val_df[features]
    y_val = val_df[TARGET_COL]
    X_test = test_df[features]
    y_test = test_df[TARGET_COL]

    # For LEAR we need a scaled X_full spanning train+val+test (no leakage — scaler is fit on train only)
    X_tr_s, X_va_s, X_te_s, x_scaler = scale_data(X_train, X_val, X_test)
    X_full_s = pd.concat([X_tr_s, X_va_s, X_te_s]).sort_index()
    y_full = pd.concat([y_train, y_val, y_test]).sort_index()

    # -----------------------------------------------------------------------
    # 2. LEAR baseline predictions on the Test Set
    # -----------------------------------------------------------------------
    logger.info("Generating LEAR predictions (this may take a few minutes)...")
    calibration_window = (
        config.get("model_settings", {})
        .get("lear", {})
        .get("calibration_window_days", 182)
    )
    lear_preds = predict_lear(
        X_full_s, y_full, X_test.index, calibration_window_days=calibration_window
    )

    # -----------------------------------------------------------------------
    # 3. QRA Ensemble median (q=0.5) predictions on the Test Set
    # -----------------------------------------------------------------------
    logger.info("Training base models for QRA...")

    lgb_p = trees_c.get("lgb", {}).copy()
    lgb_p["early_stopping_rounds"] = 50
    lgb_p["random_state"] = seed
    xgb_p = trees_c.get("xgb", {}).copy()
    xgb_p["early_stopping_rounds"] = 50
    xgb_p["random_state"] = seed
    cat_p = trees_c.get("cat", {}).copy()
    cat_p["early_stopping_rounds"] = 50
    cat_p["train_dir"] = "data/outputs/catboost_info"
    cat_p["random_state"] = seed

    m_lgb = train_lightgbm(X_train, y_train, X_val, y_val, params=lgb_p)
    m_xgb = train_xgboost(X_train, y_train, X_val, y_val, params=xgb_p)
    m_cat = train_catboost(X_train, y_train, X_val, y_val, params=cat_p)

    # DNN scaling — reuse x_scaler already built above
    y_tr_s_df, y_va_s_df, y_te_s_df, y_scaler = scale_data(
        y_train.to_frame(), y_val.to_frame(), y_test.to_frame()
    )
    y_tr_s = y_tr_s_df[TARGET_COL]
    y_va_s = y_va_s_df[TARGET_COL]
    y_te_s = y_te_s_df[TARGET_COL]

    X_tr_d, y_tr_d = reshape_to_daily(X_tr_s, y_tr_s, augment=aug)
    X_va_d, y_va_d = reshape_to_daily(X_va_s, y_va_s, augment=False)
    X_te_d, _ = reshape_to_daily(X_te_s, y_te_s, augment=False)

    dnn_params = {
        "lr": dnn_c.get("learning_rate", 0.001),
        "dropout_rate": dnn_c.get("dropout_rate", 0.2),
        "weight_decay": dnn_c.get("weight_decay", 0.0),
        "epochs": dnn_c.get("epochs", 150),
        "batch_size": dnn_c.get("batch_size", 64),
        "patience": dnn_c.get("patience", 15),
        "seed": seed,
    }
    m_dnn, device = train_pytorch_dnn(X_tr_d, y_tr_d, X_va_d, y_va_d, params=dnn_params)

    def _get_dnn_preds(model, X_daily):
        model.eval()
        with torch.no_grad():
            raw = model(torch.tensor(X_daily).to(device)).cpu().numpy().flatten()
        return y_scaler.inverse_transform(raw.reshape(-1, 1)).flatten()

    def _align(y_raw, preds_flat):
        df_tmp = pd.DataFrame({"D": y_raw.index.date}, index=y_raw.index)
        idx = [i for d, g in df_tmp.groupby("D") if len(g) == 24 for i in g.index]
        return pd.Series(preds_flat, index=idx)

    val_dnn = _align(y_val, _get_dnn_preds(m_dnn, X_va_d))
    test_dnn = _align(y_test, _get_dnn_preds(m_dnn, X_te_d))

    common_val = val_dnn.index
    common_test = test_dnn.index

    val_base = {
        "LGBM": pd.Series(m_lgb.predict(X_val), index=y_val.index).loc[common_val],
        "XGB": pd.Series(m_xgb.predict(X_val), index=y_val.index).loc[common_val],
        "Cat": pd.Series(m_cat.predict(X_val), index=y_val.index).loc[common_val],
        "DNN": val_dnn,
    }
    test_base = {
        "LGBM": pd.Series(m_lgb.predict(X_test), index=y_test.index).loc[common_test],
        "XGB": pd.Series(m_xgb.predict(X_test), index=y_test.index).loc[common_test],
        "Cat": pd.Series(m_cat.predict(X_test), index=y_test.index).loc[common_test],
        "DNN": test_dnn,
    }

    qra_params = {
        k: v for k, v in qra_c.items() if k not in ("quantiles", "alpha_winkler")
    }
    qra_params["random_state"] = seed
    q_models = train_qra(
        y_val.loc[common_val], val_base, quantiles=quantiles, params=qra_params
    )

    # Anti-crossing post-process
    X_te_qra = pd.DataFrame(test_base)
    q_arr = np.array([q_models[q].predict(X_te_qra) for q in sorted(quantiles)])
    q_arr.sort(axis=0)
    q_results = {q: q_arr[i] for i, q in enumerate(sorted(quantiles))}

    # QRA median as a Series aligned to common_test
    qra_median = pd.Series(q_results[0.5], index=common_test)
    y_test_aligned = y_test.loc[common_test]
    lear_aligned = lear_preds.loc[common_test]

    # -----------------------------------------------------------------------
    # 4. Hourly DM Test Loop
    # -----------------------------------------------------------------------
    logger.info("========================================")
    logger.info("  Hourly Diebold-Mariano Test: LEAR vs QRA (H1: QRA is more accurate)")
    logger.info("  * = statistically significant at 5% level")
    logger.info("========================================")
    header = f"{'Hour':>5}  {'LEAR MAE':>10}  {'QRA MAE':>10}  {'DM Stat':>9}  {'p-value':>9}  {'Sig':>4}"
    logger.info(header)
    logger.info("-" * len(header))

    rows = []
    for h in range(24):
        mask = y_test_aligned.index.hour == h

        y_h = y_test_aligned[mask].values
        lear_h = lear_aligned[mask].values
        qra_h = qra_median[mask].values

        # Drop NaNs from LEAR (it may have NaN for days before calibration window fills)
        valid = ~(np.isnan(lear_h) | np.isnan(qra_h))
        y_h = y_h[valid]
        lear_h = lear_h[valid]
        qra_h = qra_h[valid]

        mae_lear = float(np.mean(np.abs(y_h - lear_h))) if len(y_h) > 0 else np.nan
        mae_qra = float(np.mean(np.abs(y_h - qra_h))) if len(y_h) > 0 else np.nan

        dm_stat, p_val = dm_test(y_h, lear_h, qra_h, h=1)
        sig = (
            " *" if (p_val is not None and not np.isnan(p_val) and p_val < 0.05) else ""
        )

        row = {
            "Hour": h,
            "LEAR_MAE": mae_lear,
            "QRA_MAE": mae_qra,
            "DM_stat": dm_stat,
            "p_value": p_val,
            "significant": sig.strip(),
        }
        rows.append(row)

        p_str = (
            f"{p_val:.4f}" if p_val is not None and not np.isnan(p_val) else "   N/A"
        )
        d_str = (
            f"{dm_stat:.4f}"
            if dm_stat is not None and not np.isnan(dm_stat)
            else "   N/A"
        )
        logger.info(
            f"  {h:>3}    {mae_lear:>9.3f}   {mae_qra:>9.3f}   {d_str:>9}  {p_str:>9}  {sig}"
        )

    logger.info("========================================")
    results_df = pd.DataFrame(rows)
    n_sig = results_df["significant"].eq("*").sum()
    logger.info(f"Hours with p < 0.05 (QRA significantly better than LEAR): {n_sig}/24")
    logger.info(f"Overall LEAR MAE:  {results_df['LEAR_MAE'].mean():.3f} EUR/MWh")
    logger.info(f"Overall QRA  MAE:  {results_df['QRA_MAE'].mean():.3f} EUR/MWh")
    logger.info("========================================")
