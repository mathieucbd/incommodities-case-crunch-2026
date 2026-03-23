"""
Step 5: Statistical Significance Evaluation (Prediction Lake mode).
Runs the Diebold-Mariano (DM) test per hour to compare LEAR baseline
against QRA median predictions loaded from CSVs.
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
from src.preprocessing import chronological_train_val_test_split
from src.constants import TARGET_COL

logger = logging.getLogger(__name__)


def dm_test(actual: np.ndarray, pred1: np.ndarray, pred2: np.ndarray, h: int = 1):
    """
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

    # Loss differential: positive d means pred1 is worse
    T = len(d)
    if T < 2:
        return np.nan, np.nan

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


def _load_y_test_for_zone(target_zone: str, raw_dir: str) -> pd.Series:
    """Rebuild test target using the standard ingestion + feature + chrono split flow."""
    df = load_and_merge_zone(target_zone, raw_dir)
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
    _, _, test_df = chronological_train_val_test_split(
        df, 
        val_start=__import__("yaml").safe_load(open("config.yaml"))["data"]["val_start"], 
        test_start=__import__("yaml").safe_load(open("config.yaml"))["data"]["test_start"]
    )
    return test_df[TARGET_COL]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    raw_dir = config.get("data", {}).get("raw_dir", "data/raw/auhack_legacy/")
    target_zones = config.get("data", {}).get("target_zones", [])

    lear_df = pd.read_csv(
        "data/outputs/predictions/test/lear.csv", index_col=0, parse_dates=True
    )
    qra_df = pd.read_csv(
        "data/outputs/predictions/test/qra_median.csv", index_col=0, parse_dates=True
    )

    macro_lear_mae = {}
    macro_qra_mae = {}

    for target_zone in target_zones:
        logger.info("========================================")
        logger.info(f"Hourly Diebold-Mariano Test: LEAR vs QRA Median ({target_zone})")
        logger.info("H1: QRA Median is more accurate than LEAR")
        logger.info("* = statistically significant at 5% level")
        logger.info("========================================")

        if target_zone not in lear_df.columns:
            logger.warning(
                f"Zone {target_zone} missing in LEAR predictions CSV. Skipping."
            )
            continue
        if target_zone not in qra_df.columns:
            logger.warning(
                f"Zone {target_zone} missing in QRA median predictions CSV. Skipping."
            )
            continue

        y_test = _load_y_test_for_zone(target_zone, raw_dir)
        lear_s = lear_df[target_zone]
        qra_s = qra_df[target_zone]

        common_idx = y_test.index.intersection(lear_s.index).intersection(qra_s.index)
        y_aligned = y_test.loc[common_idx].sort_index()
        lear_aligned = lear_s.loc[common_idx].sort_index()
        qra_aligned = qra_s.loc[common_idx].sort_index()

        eval_df = pd.concat(
            [
                y_aligned.rename("y"),
                lear_aligned.rename("lear"),
                qra_aligned.rename("qra"),
            ],
            axis=1,
        ).dropna()

        if eval_df.empty:
            logger.warning(
                f"No aligned rows available after dropping NaNs for zone {target_zone}."
            )
            continue

        overall_lear_mae = float(np.mean(np.abs(eval_df["y"] - eval_df["lear"])))
        overall_qra_mae = float(np.mean(np.abs(eval_df["y"] - eval_df["qra"])))

        macro_lear_mae[target_zone] = overall_lear_mae
        macro_qra_mae[target_zone] = overall_qra_mae

        header = f"{'Hour':>5}  {'LEAR MAE':>10}  {'QRA MAE':>10}  {'DM Stat':>9}  {'p-value':>9}  {'Sig':>4}"
        logger.info(header)
        logger.info("-" * len(header))

        rows = []
        for h in range(24):
            hour_mask = pd.DatetimeIndex(eval_df.index).hour == h
            hour_df = eval_df[hour_mask]

            y_h = np.asarray(hour_df["y"].values, dtype=float)
            lear_h = np.asarray(hour_df["lear"].values, dtype=float)
            qra_h = np.asarray(hour_df["qra"].values, dtype=float)

            if len(y_h) == 0:
                mae_lear = np.nan
                mae_qra = np.nan
                dm_stat, p_val = np.nan, np.nan
            else:
                mae_lear = float(np.mean(np.abs(y_h - lear_h)))
                mae_qra = float(np.mean(np.abs(y_h - qra_h)))
                dm_stat, p_val = dm_test(y_h, lear_h, qra_h, h=1)

            sig = " *" if (not np.isnan(p_val) and p_val < 0.05) else ""
            p_str = f"{p_val:.4f}" if not np.isnan(p_val) else "   N/A"
            d_str = f"{dm_stat:.4f}" if not np.isnan(dm_stat) else "   N/A"
            ml_str = f"{mae_lear:.3f}" if not np.isnan(mae_lear) else "   N/A"
            mq_str = f"{mae_qra:.3f}" if not np.isnan(mae_qra) else "   N/A"

            logger.info(
                f"  {h:>3}    {ml_str:>9}   {mq_str:>9}   {d_str:>9}  {p_str:>9}  {sig}"
            )

            rows.append(
                {
                    "Hour": h,
                    "LEAR_MAE": mae_lear,
                    "QRA_MAE": mae_qra,
                    "DM_stat": dm_stat,
                    "p_value": p_val,
                    "significant": sig.strip(),
                }
            )

        logger.info("----------------------------------------")
        results_df = pd.DataFrame(rows)
        n_sig = results_df["significant"].eq("*").sum()
        logger.info(f"Hours with p < 0.05: {n_sig}/24")
        logger.info(f"Overall LEAR MAE: {overall_lear_mae:.3f} EUR/MWh")
        logger.info(f"Overall QRA  MAE: {overall_qra_mae:.3f} EUR/MWh")
        logger.info("========================================")

    # After the target_zones loop finishes:
    logger.info("\n========== MACRO AVERAGES ACROSS ZONES ==========")
    if macro_lear_mae and macro_qra_mae:
        avg_lear = np.mean(list(macro_lear_mae.values()))
        avg_qra = np.mean(list(macro_qra_mae.values()))
        logger.info(f"[LEAR] Avg MAE: {avg_lear:.3f} EUR/MWh")
        logger.info(f"[QRA Median] Avg MAE: {avg_qra:.3f} EUR/MWh")
    logger.info("================================================")
