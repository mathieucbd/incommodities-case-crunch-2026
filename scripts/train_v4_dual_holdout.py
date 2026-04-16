"""Attack: Winter Holdout Recalibration (ACTION 3)

Compare regime weights + HBC calibrated on:
  - SPRING holdout: Train Jul'22->Jan'24, Val Feb'24->Jun'24 (v17 default)
  - WINTER holdout: Train Jul'22->Jun'23, Val Jul'23->Jan'24

Then retrain on full data and generate submissions with:
  1. Spring weights (= v17 baseline)
  2. Winter weights
  3. Averaged weights

Usage: cd incommodities-case-crunch-2026 && PYTHONIOENCODING=utf-8 python -u scripts/train_v4_dual_holdout.py
"""

import sys, json, time, warnings
import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, ".")
from src.data_loading import load_data
from src.feature_engineering import build_features
from src.models import (
    compute_rmse, compute_hbc,
    REGIMES, optimize_regime_weights, apply_regime_weights,
    train_tree, retrain_tree, predict_tree,
    train_elastic_net, retrain_elastic_net, predict_elastic_net,
    ElecDNN, DNN_DEVICE, train_dnn, predict_dnn,
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.model_selection import KFold
from statsmodels.tsa.seasonal import STL
import lightgbm as lgb

warnings.filterwarnings("ignore")

with open("config.yaml") as f:
    config = yaml.safe_load(f)

# ══════════════════════════════════════════════════════════════════════════
# MODEL HYPERPARAMETERS (same as v17)
# ══════════════════════════════════════════════════════════════════════════
CB_FR_P = {
    "loss_function": "Quantile:alpha=0.6", "eval_metric": "Quantile:alpha=0.6",
    "iterations": 15000, "learning_rate": 0.059, "depth": 3,
    "l2_leaf_reg": 4.42, "subsample": 0.533, "colsample_bylevel": 0.228,
    "min_child_samples": 14, "random_strength": 0.9,
    "random_seed": 42, "verbose": 0, "allow_writing_files": False, "use_best_model": True,
}
CB_UK_P = {
    "loss_function": "Quantile:alpha=0.6", "eval_metric": "Quantile:alpha=0.6",
    "iterations": 15000, "learning_rate": 0.03, "depth": 8,
    "l2_leaf_reg": 5, "colsample_bylevel": 0.8, "subsample": 0.8,
    "random_seed": 42, "verbose": 0, "allow_writing_files": False, "use_best_model": True,
}
LGB_FR_P = {
    "objective": "mae", "metric": "mae",
    "n_estimators": 15000, "learning_rate": 0.03,
    "max_depth": 4, "num_leaves": 15,
    "reg_alpha": 5, "reg_lambda": 30,
    "subsample": 0.7, "colsample_bytree": 0.5, "min_child_samples": 50,
    "random_state": 42, "verbose": -1,
}
LGB_UK_P = {
    "objective": "huber", "huber_delta": 5.0, "metric": "huber",
    "n_estimators": 15000, "learning_rate": 0.02,
    "max_depth": 7, "num_leaves": 63,
    "reg_alpha": 1, "reg_lambda": 5,
    "subsample": 0.8, "colsample_bytree": 0.7, "min_child_samples": 30,
    "random_state": 42, "verbose": -1,
}
XGB_FR_P = {
    "objective": "reg:pseudohubererror", "huber_slope": 20, "eval_metric": "rmse",
    "n_estimators": 15000, "learning_rate": 0.05,
    "max_depth": 4, "reg_alpha": 5, "reg_lambda": 10,
    "subsample": 0.6, "colsample_bytree": 0.4, "min_child_weight": 15,
    "random_state": 42, "verbosity": 0, "tree_method": "hist",
}
XGB_UK_P = {
    "objective": "reg:pseudohubererror", "huber_slope": 20, "eval_metric": "rmse",
    "n_estimators": 15000, "learning_rate": 0.03,
    "max_depth": 7, "reg_alpha": 2, "reg_lambda": 8,
    "subsample": 0.75, "colsample_bytree": 0.6, "min_child_weight": 20,
    "random_state": 42, "verbosity": 0, "tree_method": "hist",
}
XGB_UK_CLUSTER_P = {**XGB_UK_P, "max_depth": 5, "min_child_weight": 50}

CLUSTER_SPLIT_SHIFTED_6H = {
    "early": [3, 4, 5, 6, 7, 8],
    "mid":   [9, 10, 11, 12, 13, 14],
    "late":  [15, 16, 17, 18, 19, 20],
    "night": [21, 22, 23, 0, 1, 2],
}

CB_SMALL = {
    "loss_function": "MAE", "eval_metric": "MAE",
    "iterations": 500, "learning_rate": 0.05, "depth": 3,
    "l2_leaf_reg": 10, "subsample": 0.8, "colsample_bylevel": 0.8,
    "min_child_samples": 50,
    "random_seed": 42, "verbose": 0, "allow_writing_files": False, "use_best_model": True,
}
XGB_SMALL = {
    "objective": "reg:pseudohubererror", "huber_slope": 10, "eval_metric": "rmse",
    "n_estimators": 500, "learning_rate": 0.05,
    "max_depth": 3, "reg_alpha": 5, "reg_lambda": 10,
    "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 50,
    "random_state": 42, "verbosity": 0, "tree_method": "hist",
}

FR_T2_ALGOS = ["ridge", "elasticnet", "lgb_small", "xgb_small"]
UK_T2_ALGOS = ["ridge", "elasticnet", "cb_small", "xgb_small"]

FR_GROUPS = {
    "fr_renewable": ["fr_wind_f","fr_solar_f","fr_wind_change_24h","fr_solar_change_24h","fr_wind_ramp_1h","fr_wind_ramp_3h","fr_wind_ramp_6h","fr_solar_ramp_1h","fr_solar_ramp_3h","fr_renewable_pen","fr_wind_pen","fr_solar_pen","fr_wind_x_hour","continent_re_pen"],
    "fr_gas": ["de_gas","fr_gas_roll_168h_mean","fr_gas_momentum_24h","uk_gas","gas_spread_uk_eu","eu_emission","eu_emission_roll_168h_mean","uk_emission","carbon_to_gas_ratio","fr_spark_spread","fr_spark_spread_ccgt","fr_spark_spread_log","fr_spot_minus_spark"],
    "fr_interco": ["atc_fr-uk-1_f","atc_uk-fr-1_f","fr_uk_atc_total","ntc_dk1-uk_f","atc_nl-uk_f","all_to_uk_atc","all_from_uk_atc","fr_uk_utilization","uk_fr_utilization","fr_congestion_x_spark_diff","spread_fr_uk_la","spread_fr_de_la","spread_fr_es"],
    "fr_load_raw": ["fr_load_f","fr_load_roll_168h_mean","fr_load_change_24h","fr_load_ramp_1h","fr_load_ramp_3h","fr_load_zscore_14d"],
    "fr_load_residual": ["fr_residual_load","fr_residual_change_24h","fr_residual_ramp_1h","fr_residual_ramp_3h","fr_residual_zscore_14d","fr_baseload_surplus","fr_baseload_gap"],
    "fr_price": ["fr_spot_la","fr_spot_lag_48h","fr_spot_lag_168h","fr_spot_la_roll_168h_mean","fr_spot_la_roll_168h_std","fr_spot_la_deviation_24h","fr_spot_la_deviation_168h","fr_spot_la_ewm_24h","fr_spot_la_roll_24h_mean","fr_spot_la_roll_24h_std","fr_spot_la_roll_24h_range","fr_mean_reversion_strength","fr_dynamic_marginal","fr_price_per_mw_7d"],
    "fr_uk_sig": ["uk_spot_la","uk_spot_la_roll_168h_mean","uk_spot_la_roll_168h_std","uk_spot_la_deviation_24h","uk_spot_la_deviation_168h","uk_price_per_mw_7d","uk_load_roll_168h_mean","uk_load_change_24h","uk_nuclear_avail_ratio","uk_residual_zscore_14d","uk_wind_f","uk_gas_cost_per_mw"],
    "fr_calendar": ["hour_sin","hour_cos","day_of_week","doy_sin","doy_cos","dow_sin","dow_cos","hour_x_dow","quarter","is_holiday_or_weekend_fr","days_since_last_holiday_fr","is_bridge_day_fr"],
    "fr_scarcity": ["euro_scarcity_ratio","fr_scarcity_barrier","uk_scarcity_ratio","uk_scarcity_extreme","fr_supply_demand_ratio","uk_supply_demand_ratio","fr_security_margin","fr_stress_index","fr_thermal_gap","uk_thermal_gap","fr_oversupply_risk","fr_gas_on_margin"],
}
FR_COMBO_PAIRS = [("fr_price","fr_calendar"),("fr_renewable","fr_load_raw"),("fr_renewable","fr_load_residual")]

UK_GROUPS = {
    "uk_wind_core": ["uk_wind_f","uk_wind_roll_24h_mean","uk_wind_change_24h","uk_wind_high","uk_wind_pen_squared","uk_wind_x_gas","uk_wind_share_flexible","uk_wind_ramp_1h","uk_wind_ramp_3h","uk_wind_ramp_6h","uk_solar_pen"],
    "uk_wind_continent": ["continental_wind_total","continental_wind_pen","nl_wind_f","be_wind_f"],
    "uk_price": ["uk_spot_la","uk_spot_la_roll_168h_mean","uk_spot_la_deviation_24h","uk_spot_la_deviation_168h","uk_spot_lag_168h","uk_spot_lag_48h","uk_basis_v2","uk_basis_v2_roll_24h_mean","uk_basis_v2_lag_48h","uk_price_per_mw_7d","uk_spot_change_168h_la","uk_lag_reliability_ratio","uk_mean_reversion_strength","uk_asinh_spot_la"],
    "uk_interco": ["atc_fr-uk-1_f","atc_uk-fr-1_f","fr_uk_atc_total","atc_uk-fr-1_f_ratio","atc_uk-fr-2_f","atc_uk-fr-2_f_ratio","atc_uk-fr-3_f_ratio","all_to_uk_atc","all_from_uk_atc","atc_nl-uk_f","uk_fr_atc_total","fr_uk_utilization","uk_fr_utilization"],
    "uk_nuclear_fr": ["fr_nuclear_avcap_f","fr_nuclear_rolling_7d_mean","fr_nuclear_change_48h","fr_nuclear_change_168h","fr_nuclear_deviation_from_7d","fr_nuclear_pct_of_load","fr_nuclear_trend_3d","uk_nuclear_avcap_ratio","uk_nuclear_pct_of_load","fr_nuke_shortfall","fr_total_dispatchable"],
    "uk_continent": ["continental_residual_load","de_residual_load","continental_load_total","continental_wind_total","continental_solar_total","continent_weighted_price","de_load_f","de_load_change_24h","iberian_load","es_residual_load","spread_fr_de_la","spread_fr_es"],
    "uk_fr_price": ["fr_spot_la","fr_spot_la_roll_168h_mean","fr_spot_la_deviation_24h","fr_spot_la_deviation_168h","spread_fr_uk_la","fr_dynamic_marginal","fr_merit_order_cost","fr_price_per_mw_7d","fr_spot_minus_spark","cost_nl-uk_la","fr_uk_cost_spread_la"],
    "uk_calendar": ["hour_sin","hour_cos","day_of_week","doy_sin","doy_cos","dow_sin","dow_cos","hour_x_dow","quarter","is_holiday_or_weekend_uk","days_since_last_holiday_fr"],
    "uk_emissions": ["uk_emission","eu_emission","eu_emission_roll_168h_mean","carbon_to_gas_ratio","alpine_hydro_total","ch_hydro_total","at_hydro_total","fr_hydro_change_24h","de_max_river_temp"],
}
UK_COMBO_PAIRS = [("uk_price","uk_fr_price"),("uk_wind_core","uk_nuclear_fr"),("uk_nuclear_fr","uk_calendar")]


# ══════════════════════════════════════════════════════════════════════════
# T2 HELPER FUNCTIONS (same as v17)
# ══════════════════════════════════════════════════════════════════════════

def build_fundamental_features(df, market="fr"):
    feats = {}
    for col_suffix in ["load_f", "wind_f", "solar_f", "nuclear_avcap_f", "residual_load", "spot_la"]:
        col = f"{market}_{col_suffix}"
        if col in df.columns:
            feats[col_suffix] = df[col].fillna(0).values
    col_gas = f"{market}_gas" if f"{market}_gas" in df.columns else "de_gas"
    if col_gas in df.columns:
        feats["gas"] = df[col_gas].ffill().fillna(0).values
    return pd.DataFrame(feats, index=df.index)


def train_t2(algo, X_tr, y_tr, X_va, y_va=None):
    if algo == "ridge":
        scaler = StandardScaler()
        Xts = scaler.fit_transform(np.nan_to_num(X_tr.copy(), 0))
        Xvs = scaler.transform(np.nan_to_num(X_va.copy(), 0))
        m = Ridge(alpha=10.0); m.fit(Xts, y_tr)
        return m.predict(Xvs), {"model": m, "scaler": scaler, "algo": algo}
    elif algo == "elasticnet":
        scaler = StandardScaler()
        Xts = scaler.fit_transform(np.nan_to_num(X_tr.copy(), 0))
        Xvs = scaler.transform(np.nan_to_num(X_va.copy(), 0))
        m = ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=5000); m.fit(Xts, y_tr)
        return m.predict(Xvs), {"model": m, "scaler": scaler, "algo": algo}
    elif algo == "lgb_small":
        ds = lgb.Dataset(np.nan_to_num(X_tr.copy(), 0), label=y_tr)
        params = {"objective": "mae", "max_depth": 3, "learning_rate": 0.05,
                  "num_leaves": 8, "min_child_samples": 50,
                  "subsample": 0.8, "colsample_bytree": 0.8, "verbose": -1}
        m = lgb.train(params, ds, num_boost_round=500)
        return m.predict(np.nan_to_num(X_va.copy(), 0)), {"model": m, "algo": algo, "params": params}
    elif algo == "cb_small":
        from catboost import CatBoostRegressor, Pool
        m = CatBoostRegressor(**CB_SMALL)
        Xt = np.nan_to_num(X_tr.copy(), 0); Xv = np.nan_to_num(X_va.copy(), 0)
        yt = np.array(y_tr, dtype=np.float64)
        fit_kw = {"verbose": 0}
        if y_va is not None:
            fit_kw["eval_set"] = Pool(Xv, np.array(y_va, dtype=np.float64))
            fit_kw["early_stopping_rounds"] = 100
        m.fit(Pool(Xt, yt), **fit_kw)
        return m.predict(Xv), {"model": m, "algo": algo, "best_iter": m.get_best_iteration()}
    elif algo == "xgb_small":
        import xgboost as xgb_lib
        m = xgb_lib.XGBRegressor(**XGB_SMALL, early_stopping_rounds=100)
        Xt = np.nan_to_num(X_tr.copy(), 0); Xv = np.nan_to_num(X_va.copy(), 0)
        yt = np.array(y_tr, dtype=np.float64)
        if y_va is not None:
            m.fit(Xt, yt, eval_set=[(Xv, np.array(y_va, dtype=np.float64))], verbose=False)
        else:
            m.fit(Xt, yt, verbose=False)
        return m.predict(Xv), {"model": m, "algo": algo, "best_iter": m.best_iteration if hasattr(m, 'best_iteration') else 500}


def retrain_t2(algo, X_full, y_full, info):
    if algo in ("ridge", "elasticnet"):
        scaler = StandardScaler()
        Xfs = scaler.fit_transform(np.nan_to_num(X_full.copy(), 0))
        m = Ridge(alpha=10.0) if algo == "ridge" else ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=5000)
        m.fit(Xfs, y_full)
        return {"model": m, "scaler": scaler}
    elif algo == "lgb_small":
        ds = lgb.Dataset(np.nan_to_num(X_full.copy(), 0), label=y_full)
        m = lgb.train(info["params"], ds, num_boost_round=500)
        return {"model": m}
    elif algo == "cb_small":
        from catboost import CatBoostRegressor, Pool
        params = {**CB_SMALL, "use_best_model": False, "iterations": info.get("best_iter", 500)}
        m = CatBoostRegressor(**params)
        m.fit(Pool(np.nan_to_num(X_full.copy(), 0), np.array(y_full, dtype=np.float64)), verbose=0)
        return {"model": m}
    elif algo == "xgb_small":
        import xgboost as xgb_lib
        params = {**XGB_SMALL, "n_estimators": info.get("best_iter", 500)}
        m = xgb_lib.XGBRegressor(**params)
        m.fit(np.nan_to_num(X_full.copy(), 0), np.array(y_full, dtype=np.float64), verbose=False)
        return {"model": m}


def predict_t2(retrained, X_test):
    if "scaler" in retrained:
        return retrained["model"].predict(retrained["scaler"].transform(np.nan_to_num(X_test.copy(), 0)))
    else:
        return retrained["model"].predict(np.nan_to_num(X_test.copy(), 0))


# ══════════════════════════════════════════════════════════════════════════
# HOLDOUT PIPELINE FUNCTION
# ══════════════════════════════════════════════════════════════════════════

def run_holdout(name, df_train_h, df_val_h, feat_fr, feat_uk, feat_dnn_final,
                fr_groups, uk_groups):
    """Run full v17-equivalent pipeline on a holdout split.
    Returns dict with regime weights, HBC, model info, and validation metrics."""

    print(f"\n{'='*90}")
    print(f"  HOLDOUT: {name}")
    print(f"  Train: {len(df_train_h)} | Val: {len(df_val_h)}")
    print(f"{'='*90}")
    t_start = time.time()

    # ── Targets ──
    fr_rm_tr = df_train_h["fr_stl_trend"].values
    fr_rm_va = df_val_h["fr_stl_trend"].values
    fr_y_dev_tr = (df_train_h["fr_spot"] - df_train_h["fr_stl_trend"]).values
    fr_y_dev_va = (df_val_h["fr_spot"] - df_val_h["fr_stl_trend"]).values
    fr_valid_tr = np.isfinite(fr_y_dev_tr)
    fr_valid_va = np.isfinite(fr_y_dev_va)
    spot_va_fr = df_val_h["fr_spot"].values
    hours_va = df_val_h["hour"].values
    dow_va = pd.to_datetime(df_val_h["datetime_CET"]).dt.dayofweek.values

    df_train_uk_h = df_train_h.copy()
    uk_spot_va = df_val_h["uk_spot"].values
    uk_moc_tr = df_train_uk_h["uk_merit_order_cost"].values
    uk_moc_va = df_val_h["uk_merit_order_cost"].values
    y_basis_tr = df_train_uk_h["uk_spot"].values - uk_moc_tr
    y_basis_va = uk_spot_va - uk_moc_va
    valid_basis_tr = np.isfinite(y_basis_tr)
    valid_basis_va = np.isfinite(y_basis_va)
    vb = valid_basis_va

    # ── V9 Models ──
    print(f"  [{name}] V9 models...")

    cb_fr = train_tree("catboost", CB_FR_P,
        df_train_h.loc[df_train_h.index[fr_valid_tr], feat_fr].values,
        fr_y_dev_tr[fr_valid_tr],
        df_val_h.loc[df_val_h.index[fr_valid_va], feat_fr].values,
        fr_y_dev_va[fr_valid_va])
    preds_fr_cb = fr_rm_va + predict_tree(cb_fr.model, df_val_h[feat_fr].values)

    lgb_fr = train_tree("lightgbm", LGB_FR_P,
        df_train_h.loc[df_train_h.index[fr_valid_tr], feat_fr].values,
        fr_y_dev_tr[fr_valid_tr],
        df_val_h.loc[df_val_h.index[fr_valid_va], feat_fr].values,
        fr_y_dev_va[fr_valid_va])
    preds_fr_lgb = fr_rm_va + predict_tree(lgb_fr.model, df_val_h[feat_fr].values)

    xgb_fr = train_tree("xgboost", XGB_FR_P,
        df_train_h.loc[df_train_h.index[fr_valid_tr], feat_fr].values,
        fr_y_dev_tr[fr_valid_tr],
        df_val_h.loc[df_val_h.index[fr_valid_va], feat_fr].values,
        fr_y_dev_va[fr_valid_va])
    preds_fr_xgb = fr_rm_va + predict_tree(xgb_fr.model, df_val_h[feat_fr].values)

    en_fr = train_elastic_net(
        df_train_h.loc[df_train_h.index[fr_valid_tr], feat_fr].values.copy(),
        fr_y_dev_tr[fr_valid_tr].copy(),
        df_val_h[feat_fr].values.copy(), alpha=10.0, l1_ratio=0.9)
    preds_fr_en = fr_rm_va + en_fr.preds_val

    dnn_scaler_fr = StandardScaler()
    X_dnn_tr_fr = dnn_scaler_fr.fit_transform(np.nan_to_num(df_train_h[feat_dnn_final].values.copy(), 0))
    X_dnn_va_fr = dnn_scaler_fr.transform(np.nan_to_num(df_val_h[feat_dnn_final].values.copy(), 0))
    torch.manual_seed(42); np.random.seed(42)
    dnn_fr = ElecDNN(len(feat_dnn_final), [192, 96], dropout=0.2)
    dnn_fr, dnn_fr_epochs = train_dnn(
        dnn_fr, X_dnn_tr_fr[fr_valid_tr], fr_y_dev_tr[fr_valid_tr].astype(np.float32),
        X_dnn_va_fr[fr_valid_va], fr_y_dev_va[fr_valid_va].astype(np.float32))
    preds_fr_dnn = fr_rm_va + predict_dnn(dnn_fr, X_dnn_va_fr)

    # UK models
    cb_uk = train_tree("catboost", CB_UK_P,
        df_train_uk_h.loc[df_train_uk_h.index[valid_basis_tr], feat_uk].values,
        y_basis_tr[valid_basis_tr],
        df_val_h.loc[df_val_h.index[valid_basis_va], feat_uk].values,
        y_basis_va[valid_basis_va])
    preds_uk_cb = uk_moc_va + predict_tree(cb_uk.model, df_val_h[feat_uk].values)

    lgb_uk = train_tree("lightgbm", LGB_UK_P,
        df_train_uk_h.loc[df_train_uk_h.index[valid_basis_tr], feat_uk].values,
        y_basis_tr[valid_basis_tr],
        df_val_h.loc[df_val_h.index[valid_basis_va], feat_uk].values,
        y_basis_va[valid_basis_va])
    preds_uk_lgb = uk_moc_va + predict_tree(lgb_uk.model, df_val_h[feat_uk].values)

    xgb_uk = train_tree("xgboost", XGB_UK_P,
        df_train_uk_h.loc[df_train_uk_h.index[valid_basis_tr], feat_uk].values,
        y_basis_tr[valid_basis_tr],
        df_val_h.loc[df_val_h.index[valid_basis_va], feat_uk].values,
        y_basis_va[valid_basis_va])
    preds_uk_xgb = uk_moc_va + predict_tree(xgb_uk.model, df_val_h[feat_uk].values)

    en_uk = train_elastic_net(
        df_train_uk_h.loc[df_train_uk_h.index[valid_basis_tr], feat_uk].values.copy(),
        y_basis_tr[valid_basis_tr].copy(),
        df_val_h[feat_uk].values.copy(), alpha=1.0, l1_ratio=0.9)
    preds_uk_en = uk_moc_va + en_uk.preds_val

    dnn_scaler_uk = StandardScaler()
    X_dnn_tr_uk = dnn_scaler_uk.fit_transform(np.nan_to_num(df_train_uk_h[feat_dnn_final].values.copy(), 0))
    X_dnn_va_uk = dnn_scaler_uk.transform(np.nan_to_num(df_val_h[feat_dnn_final].values.copy(), 0))
    torch.manual_seed(42); np.random.seed(42)
    dnn_uk = ElecDNN(len(feat_dnn_final), [768, 384, 192], dropout=0.3)
    dnn_uk, dnn_uk_epochs = train_dnn(
        dnn_uk, X_dnn_tr_uk[valid_basis_tr], y_basis_tr[valid_basis_tr].astype(np.float32),
        X_dnn_va_uk[valid_basis_va], y_basis_va[valid_basis_va].astype(np.float32),
        criterion=torch.nn.MSELoss())
    preds_uk_dnn = uk_moc_va + predict_dnn(dnn_uk, X_dnn_va_uk)

    # XGB cluster UK
    hours_tr = df_train_uk_h["hour"].values
    hours_v = df_val_h["hour"].values
    xgb_cluster_models = {}
    preds_uk_xgb_cluster_basis = np.zeros(len(df_val_h))
    for cname, c_hours in CLUSTER_SPLIT_SHIFTED_6H.items():
        c_mask_tr = np.isin(hours_tr, c_hours) & valid_basis_tr
        c_mask_va = np.isin(hours_v, c_hours) & valid_basis_va
        va_hour_mask = np.isin(hours_v, c_hours)
        result = train_tree("xgboost", XGB_UK_CLUSTER_P,
            df_train_uk_h.loc[df_train_uk_h.index[c_mask_tr], feat_uk].values,
            y_basis_tr[c_mask_tr],
            df_val_h.loc[df_val_h.index[c_mask_va], feat_uk].values,
            y_basis_va[c_mask_va])
        xgb_cluster_models[cname] = result
        preds_uk_xgb_cluster_basis[va_hour_mask] = predict_tree(result.model,
            df_val_h.loc[df_val_h.index[va_hour_mask], feat_uk].values)
    preds_uk_xgb_cluster = uk_moc_va + preds_uk_xgb_cluster_basis

    print(f"  [{name}] V9 trained ({time.time()-t_start:.0f}s), DNN: FR={dnn_fr_epochs}ep, UK={dnn_uk_epochs}ep")

    # ── Ridge Fondamentales ──
    fund_fr_tr = build_fundamental_features(df_train_h, "fr")
    fund_fr_va = build_fundamental_features(df_val_h, "fr")
    scaler_rf = StandardScaler()
    X_rf_tr = scaler_rf.fit_transform(np.nan_to_num(fund_fr_tr.values.copy(), 0))
    X_rf_va = scaler_rf.transform(np.nan_to_num(fund_fr_va.values.copy(), 0))
    ridge_fr = Ridge(alpha=10.0)
    ridge_fr.fit(X_rf_tr, df_train_h["fr_spot"].values)
    preds_fr_ridge = ridge_fr.predict(X_rf_va)

    fund_uk_tr = build_fundamental_features(df_train_uk_h, "uk")
    fund_uk_va = build_fundamental_features(df_val_h, "uk")
    scaler_ruk = StandardScaler()
    X_ruk_tr = scaler_ruk.fit_transform(np.nan_to_num(fund_uk_tr.values.copy(), 0))
    X_ruk_va = scaler_ruk.transform(np.nan_to_num(fund_uk_va.values.copy(), 0))
    ridge_uk = Ridge(alpha=10.0)
    ridge_uk.fit(X_ruk_tr, df_train_uk_h["uk_spot"].values)
    preds_uk_ridge = ridge_uk.predict(X_ruk_va)

    # ── V9 ensemble (for SR residual) ──
    v9_fr = {"CB": preds_fr_cb, "LGB": preds_fr_lgb, "XGB": preds_fr_xgb,
             "EN": preds_fr_en, "DNN": preds_fr_dnn}
    v9_uk = {"CB": preds_uk_cb[vb], "LGB": preds_uk_lgb[vb], "XGB": preds_uk_xgb[vb],
             "EN": preds_uk_en[vb], "DNN": preds_uk_dnn[vb]}
    v9_regime_fr, v9_ens_fr = optimize_regime_weights(v9_fr, spot_va_fr, hours_va, f"FR_v9_{name}", step=0.1)
    v9_regime_uk, v9_ens_uk = optimize_regime_weights(v9_uk, uk_spot_va[vb], hours_va[vb], f"UK_v9_{name}", step=0.1)

    # ── T2 Models ──
    print(f"  [{name}] T2 models...")
    t2_start = time.time()
    t2_fr = {}
    for gname, feats in fr_groups.items():
        X_tr = df_train_h[feats].values[fr_valid_tr]
        X_va = df_val_h[feats].values
        y_tr = fr_y_dev_tr[fr_valid_tr]
        y_va = fr_y_dev_va[fr_valid_va]
        for algo in FR_T2_ALGOS:
            try:
                preds_dev, info = train_t2(algo, X_tr, y_tr, X_va,
                                           y_va=y_va if algo in ("cb_small", "xgb_small") else None)
                t2_fr[f"{gname}__{algo}"] = {"preds": fr_rm_va + preds_dev, "group": gname,
                                              "algo": algo, "info": info, "feats": feats}
            except Exception as e:
                print(f"    ERR FR {gname} {algo}: {e}")

    t2_uk = {}
    for gname, feats in uk_groups.items():
        X_tr = df_train_uk_h[feats].values[valid_basis_tr]
        X_va = df_val_h[feats].values
        y_tr = y_basis_tr[valid_basis_tr]
        y_va = y_basis_va[valid_basis_va]
        for algo in UK_T2_ALGOS:
            try:
                preds_basis, info = train_t2(algo, X_tr, y_tr, X_va,
                                              y_va=y_va if algo in ("cb_small", "xgb_small") else None)
                t2_uk[f"{gname}__{algo}"] = {"preds": uk_moc_va + preds_basis, "group": gname,
                                              "algo": algo, "info": info, "feats": feats}
            except Exception as e:
                print(f"    ERR UK {gname} {algo}: {e}")

    # Combos
    for g1, g2 in FR_COMBO_PAIRS:
        if g1 not in fr_groups or g2 not in fr_groups: continue
        feats = list(set(fr_groups[g1] + fr_groups[g2]))
        X_tr = df_train_h[feats].values[fr_valid_tr]
        X_va = df_val_h[feats].values
        preds_dev, info = train_t2("ridge", X_tr, fr_y_dev_tr[fr_valid_tr], X_va)
        t2_fr[f"combo_{g1}+{g2}__ridge"] = {"preds": fr_rm_va + preds_dev, "group": f"{g1}+{g2}",
                                              "algo": "ridge", "info": info, "feats": feats}

    for g1, g2 in UK_COMBO_PAIRS:
        if g1 not in uk_groups or g2 not in uk_groups: continue
        feats = list(set(uk_groups[g1] + uk_groups[g2]))
        X_tr = df_train_uk_h[feats].values[valid_basis_tr]
        X_va = df_val_h[feats].values
        preds_basis, info = train_t2("ridge", X_tr, y_basis_tr[valid_basis_tr], X_va)
        t2_uk[f"combo_{g1}+{g2}__ridge"] = {"preds": uk_moc_va + preds_basis, "group": f"{g1}+{g2}",
                                              "algo": "ridge", "info": info, "feats": feats}

    print(f"  [{name}] T2: {len(t2_fr)} FR + {len(t2_uk)} UK ({time.time()-t2_start:.0f}s)")

    # ── Stacking Résiduel ──
    def make_vb(d):
        return {k: {"preds": v["preds"][vb], **{kk: vv for kk, vv in v.items() if kk != "preds"}}
                for k, v in d.items()}

    # FR: alpha=1
    n_fr = len(spot_va_fr)
    residuals_fr = spot_va_fr - v9_ens_fr[:n_fr]
    fr_t2_names = sorted(t2_fr.keys())
    X_sr_fr = np.column_stack([t2_fr[k]["preds"][:n_fr] for k in fr_t2_names])
    oof_fr = np.zeros(n_fr)
    kf = KFold(n_splits=5, shuffle=False)
    for tr_idx, va_idx in kf.split(X_sr_fr):
        meta = Ridge(alpha=1.0)
        meta.fit(X_sr_fr[tr_idx], residuals_fr[tr_idx])
        oof_fr[va_idx] = meta.predict(X_sr_fr[va_idx])
    sr_fr_val = v9_ens_fr[:n_fr] + oof_fr
    meta_fr = Ridge(alpha=1.0)
    meta_fr.fit(X_sr_fr, residuals_fr)

    # UK: enriched, alpha=500
    t2_uk_vb = make_vb(t2_uk)
    n_uk = vb.sum()
    residuals_uk = uk_spot_va[vb] - v9_ens_uk[:n_uk]
    uk_t2_names = sorted(t2_uk_vb.keys())
    X_sr_uk_t2 = np.column_stack([t2_uk_vb[k]["preds"][:n_uk] for k in uk_t2_names])
    h_uk = hours_va[vb][:n_uk]; d_uk = dow_va[vb][:n_uk]
    X_extra_uk = np.column_stack([
        np.sin(2*np.pi*h_uk/24), np.cos(2*np.pi*h_uk/24),
        np.sin(2*np.pi*d_uk/7), np.cos(2*np.pi*d_uk/7),
    ])
    X_sr_uk = np.hstack([X_sr_uk_t2, X_extra_uk])
    oof_uk = np.zeros(n_uk)
    kf = KFold(n_splits=5, shuffle=False)
    for tr_idx, va_idx in kf.split(X_sr_uk):
        meta = Ridge(alpha=500.0)
        meta.fit(X_sr_uk[tr_idx], residuals_uk[tr_idx])
        oof_uk[va_idx] = meta.predict(X_sr_uk[va_idx])
    sr_uk_val = v9_ens_uk[:n_uk] + oof_uk
    meta_uk = Ridge(alpha=500.0)
    meta_uk.fit(X_sr_uk, residuals_uk)

    # ── Regime Ensemble (7 FR, 8 UK) ──
    v9r_fr = {"CB": preds_fr_cb, "LGB": preds_fr_lgb, "XGB": preds_fr_xgb,
              "EN": preds_fr_en, "DNN": preds_fr_dnn, "RidgeF": preds_fr_ridge}
    v9r_uk = {"CB": preds_uk_cb[vb], "LGB": preds_uk_lgb[vb], "XGB": preds_uk_xgb[vb],
              "EN": preds_uk_en[vb], "DNN": preds_uk_dnn[vb], "RidgeF": preds_uk_ridge[vb]}
    d_fr = {**v9r_fr, "SR": sr_fr_val}
    d_uk = {**v9r_uk, "SR": sr_uk_val, "XGB_C": preds_uk_xgb_cluster[vb]}

    fr_regime_weights, preds_fr_ens = optimize_regime_weights(d_fr, spot_va_fr, hours_va, f"FR_{name}", step=0.1)
    uk_regime_weights, preds_uk_ens = optimize_regime_weights(d_uk, uk_spot_va[vb], hours_va[vb], f"UK_{name}", step=0.1)

    # ── HBC ──
    hbc_fr, rmse_fr_hbc = compute_hbc(preds_fr_ens, spot_va_fr, hours_va)
    hbc_uk, rmse_uk_hbc = compute_hbc(preds_uk_ens, uk_spot_va[vb], hours_va[vb])

    rmse_fr_raw = compute_rmse(spot_va_fr, preds_fr_ens)
    rmse_uk_raw = compute_rmse(uk_spot_va[vb], preds_uk_ens)

    print(f"  [{name}] RESULT: FR={rmse_fr_hbc:.4f} UK={rmse_uk_hbc:.4f} SUM={rmse_fr_hbc+rmse_uk_hbc:.4f}")
    print(f"  [{name}] Total: {time.time()-t_start:.0f}s")

    return {
        "fr_regime_weights": fr_regime_weights,
        "uk_regime_weights": uk_regime_weights,
        "hbc_fr": hbc_fr,
        "hbc_uk": hbc_uk,
        "v9_regime_fr": v9_regime_fr,
        "v9_regime_uk": v9_regime_uk,
        "meta_fr": meta_fr,
        "meta_uk": meta_uk,
        "fr_t2_names": fr_t2_names,
        "uk_t2_names": uk_t2_names,
        "t2_fr": t2_fr,
        "t2_uk": t2_uk,
        "model_iters": {
            "cb_fr": cb_fr.best_iteration, "lgb_fr": lgb_fr.best_iteration,
            "xgb_fr": xgb_fr.best_iteration, "cb_uk": cb_uk.best_iteration,
            "lgb_uk": lgb_uk.best_iteration, "xgb_uk": xgb_uk.best_iteration,
            "dnn_fr_epochs": dnn_fr_epochs, "dnn_uk_epochs": dnn_uk_epochs,
            "xgb_cluster": {c: m.best_iteration for c, m in xgb_cluster_models.items()},
        },
        "val_metrics": {
            "fr_raw": rmse_fr_raw, "uk_raw": rmse_uk_raw,
            "fr_hbc": rmse_fr_hbc, "uk_hbc": rmse_uk_hbc,
            "sum_hbc": rmse_fr_hbc + rmse_uk_hbc,
        },
    }


# ══════════════════════════════════════════════════════════════════════════
# 0. LOAD DATA + FEATURES
# ══════════════════════════════════════════════════════════════════════════
print("=" * 90)
print("  ATTACK: WINTER HOLDOUT RECALIBRATION (ACTION 3)")
print("=" * 90)

t0 = time.time()
x_train, y_train, x_test = load_data("data/raw")
train_fe = build_features(pd.concat([x_train], axis=0), config)
train_fe = train_fe.join(y_train[["fr_spot", "uk_spot"]])
test_fe = build_features(x_test, config)
print(f"  Data loaded in {time.time()-t0:.0f}s")

# Interaction feature
for df in [train_fe, test_fe]:
    if "fr_spot_la_roll_168h_mean" in df.columns and "uk_price_per_mw_7d" in df.columns:
        df["X_fr_spot_la_roll_168h_mean_x_uk_price_per_mw_7d"] = (
            df["fr_spot_la_roll_168h_mean"] * df["uk_price_per_mw_7d"]
        )

# STL decomposition on full train_fe (same as v17)
spot_la_series = train_fe["fr_spot_la"].ffill().bfill()
stl = STL(spot_la_series, period=168, seasonal=13)
result = stl.fit()
train_fe["fr_stl_trend"] = result.trend
train_fe["fr_stl_seasonal"] = result.seasonal

# STL for test (concat for continuity)
test_spot_la = test_fe["fr_spot_la"].ffill().bfill()
full_spot_la = pd.concat([spot_la_series, test_spot_la])
stl_full = STL(full_spot_la, period=168, seasonal=13)
result_full = stl_full.fit()
test_fe["fr_stl_trend"] = result_full.trend.iloc[len(train_fe):].values
test_fe["fr_stl_seasonal"] = result_full.seasonal.iloc[len(train_fe):].values

# Feature lists
_EXCLUDE = {"fr_spot", "uk_spot", "datetime_CET", "datetime_UTC", "date", "id"}
_all_num = [c for c in train_fe.columns
            if c not in _EXCLUDE
            and train_fe[c].dtype in ["float64", "float32", "int64", "int32"]
            and train_fe[c].notna().sum() > len(train_fe) * 0.5]
_corr = train_fe[_all_num].corr().abs()
_to_drop = set()
for _i in range(len(_all_num)):
    if _all_num[_i] in _to_drop: continue
    for _j in range(_i + 1, len(_all_num)):
        if _all_num[_j] in _to_drop: continue
        if _corr.iloc[_i, _j] > 0.99:
            _to_drop.add(_all_num[_j])
feat_dnn = [f for f in _all_num if f not in _to_drop]

with open("outputs/feature_selection_v5_fr.json") as f:
    fs_v5 = json.load(f)
feat_fr = [f for f in fs_v5["features"] + ["X_fr_spot_la_roll_168h_mean_x_uk_price_per_mw_7d"]
           if f in train_fe.columns]
feat_fr.append("fr_stl_seasonal")

with open("outputs/uk_feature_research.json") as f:
    uk_research = json.load(f)
feat_uk = [f for f in uk_research["confirmed_features"] if f in train_fe.columns]

feat_dnn_final = [f for f in feat_dnn if f in train_fe.columns]
feat_dnn_final.append("fr_stl_seasonal")

# Filter groups
fr_groups = {k: [f for f in v if f in train_fe.columns] for k, v in FR_GROUPS.items()}
uk_groups = {k: [f for f in v if f in train_fe.columns] for k, v in UK_GROUPS.items()}
fr_groups = {k: v for k, v in fr_groups.items() if len(v) >= 3}
uk_groups = {k: v for k, v in uk_groups.items() if len(v) >= 3}

print(f"  FR: {len(feat_fr)} features, UK: {len(feat_uk)}, DNN: {len(feat_dnn_final)}")
print(f"  T2 groups: FR={len(fr_groups)}, UK={len(uk_groups)}")


# ══════════════════════════════════════════════════════════════════════════
# 1. RUN BOTH HOLDOUTS
# ══════════════════════════════════════════════════════════════════════════

# Spring holdout (v17 default)
mask_spring = train_fe["datetime_CET"] >= "2024-02-01"
df_train_spring = train_fe[~mask_spring].copy()
df_val_spring = train_fe[mask_spring].copy()
spring = run_holdout("SPRING", df_train_spring, df_val_spring, feat_fr, feat_uk, feat_dnn_final, fr_groups, uk_groups)

# Winter holdout W2
mask_winter_train = train_fe["datetime_CET"] < "2023-07-01"
mask_winter_val = (train_fe["datetime_CET"] >= "2023-07-01") & (train_fe["datetime_CET"] < "2024-02-01")
df_train_winter = train_fe[mask_winter_train].copy()
df_val_winter = train_fe[mask_winter_val].copy()
winter = run_holdout("WINTER", df_train_winter, df_val_winter, feat_fr, feat_uk, feat_dnn_final, fr_groups, uk_groups)


# ══════════════════════════════════════════════════════════════════════════
# 2. COMPARE WEIGHTS
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  WEIGHT COMPARISON: SPRING vs WINTER")
print("=" * 90)

model_names_fr = ["CB", "LGB", "XGB", "EN", "DNN", "RidgeF", "SR"]
model_names_uk = ["CB", "LGB", "XGB", "EN", "DNN", "RidgeF", "SR", "XGB_C"]

# Compute averaged weights
avg_fr_weights = {}
avg_uk_weights = {}
for regime in REGIMES:
    avg_fr_weights[regime] = {}
    for m in model_names_fr:
        ws = spring["fr_regime_weights"].get(regime, {}).get(m, 0)
        ww = winter["fr_regime_weights"].get(regime, {}).get(m, 0)
        avg_fr_weights[regime][m] = (ws + ww) / 2
    avg_uk_weights[regime] = {}
    for m in model_names_uk:
        ws = spring["uk_regime_weights"].get(regime, {}).get(m, 0)
        ww = winter["uk_regime_weights"].get(regime, {}).get(m, 0)
        avg_uk_weights[regime][m] = (ws + ww) / 2

# Compute averaged HBC
avg_hbc_fr = {h: (spring["hbc_fr"].get(h, 0) + winter["hbc_fr"].get(h, 0)) / 2 for h in range(24)}
avg_hbc_uk = {h: (spring["hbc_uk"].get(h, 0) + winter["hbc_uk"].get(h, 0)) / 2 for h in range(24)}

print(f"\n  FR regime weights:")
print(f"  {'Regime':8s} | {'Model':6s} | {'SPRING':>7s} | {'WINTER':>7s} | {'AVG':>5s} | {'Delta':>6s}")
print(f"  {'-'*50}")
for regime in REGIMES:
    for m in model_names_fr:
        ws = spring["fr_regime_weights"].get(regime, {}).get(m, 0)
        ww = winter["fr_regime_weights"].get(regime, {}).get(m, 0)
        if ws > 0 or ww > 0:
            print(f"  {regime:8s} | {m:6s} | {ws:7.2f} | {ww:7.2f} | {(ws+ww)/2:5.2f} | {ww-ws:+6.2f}")

print(f"\n  UK regime weights:")
print(f"  {'Regime':8s} | {'Model':6s} | {'SPRING':>7s} | {'WINTER':>7s} | {'AVG':>5s} | {'Delta':>6s}")
print(f"  {'-'*50}")
for regime in REGIMES:
    for m in model_names_uk:
        ws = spring["uk_regime_weights"].get(regime, {}).get(m, 0)
        ww = winter["uk_regime_weights"].get(regime, {}).get(m, 0)
        if ws > 0 or ww > 0:
            print(f"  {regime:8s} | {m:6s} | {ws:7.2f} | {ww:7.2f} | {(ws+ww)/2:5.2f} | {ww-ws:+6.2f}")

print(f"\n  HBC comparison (top changes):")
print(f"  {'Hour':>4s} | {'S_FR':>6s} {'W_FR':>6s} {'D_FR':>6s} | {'S_UK':>6s} {'W_UK':>6s} {'D_UK':>6s}")
for h in range(24):
    sfr = spring["hbc_fr"].get(h, 0); wfr = winter["hbc_fr"].get(h, 0)
    suk = spring["hbc_uk"].get(h, 0); wuk = winter["hbc_uk"].get(h, 0)
    print(f"  {h:4d} | {sfr:+6.2f} {wfr:+6.2f} {wfr-sfr:+6.2f} | {suk:+6.2f} {wuk:+6.2f} {wuk-suk:+6.2f}")

print(f"\n  Validation metrics:")
print(f"  {'':12s} | {'SPRING':>8s} | {'WINTER':>8s}")
print(f"  {'FR raw':12s} | {spring['val_metrics']['fr_raw']:8.4f} | {winter['val_metrics']['fr_raw']:8.4f}")
print(f"  {'UK raw':12s} | {spring['val_metrics']['uk_raw']:8.4f} | {winter['val_metrics']['uk_raw']:8.4f}")
print(f"  {'FR +HBC':12s} | {spring['val_metrics']['fr_hbc']:8.4f} | {winter['val_metrics']['fr_hbc']:8.4f}")
print(f"  {'UK +HBC':12s} | {spring['val_metrics']['uk_hbc']:8.4f} | {winter['val_metrics']['uk_hbc']:8.4f}")
print(f"  {'SUM':12s} | {spring['val_metrics']['sum_hbc']:8.4f} | {winter['val_metrics']['sum_hbc']:8.4f}")


# ══════════════════════════════════════════════════════════════════════════
# 3. RETRAIN ON FULL DATA (using spring holdout model info)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  3. Retrain on FULL data + test predictions")
print("=" * 90)

t_retrain = time.time()

# Coherent STL for retrain (v17 Fix 2)
full_spot_la_retrain = pd.concat([
    train_fe["fr_spot_la"].ffill().bfill(),
    test_fe["fr_spot_la"].ffill().bfill()
])
stl_retrain = STL(full_spot_la_retrain, period=168, seasonal=13)
result_retrain = stl_retrain.fit()
rm_fr_all_coherent = result_retrain.trend.values
n_full = len(train_fe)
rm_fr_test = rm_fr_all_coherent[n_full:]
spot_fr_full = train_fe["fr_spot"].values
y_dev_fr_full = spot_fr_full - rm_fr_all_coherent[:n_full]
valid_fr_full = np.isfinite(y_dev_fr_full)
hours_test = test_fe["hour"].values
dow_test = pd.to_datetime(test_fe["datetime_CET"]).dt.dayofweek.values

# Use spring holdout best iterations for retrain
si = spring["model_iters"]

# FR CatBoost retrain
cb_fr_final = retrain_tree("catboost", CB_FR_P,
    train_fe.loc[train_fe.index[valid_fr_full], feat_fr], y_dev_fr_full[valid_fr_full], si["cb_fr"])
preds_fr_test_cb = rm_fr_test + predict_tree(cb_fr_final, test_fe[feat_fr])
print(f"  FR CB retrained ({si['cb_fr']} iter)")

# FR LightGBM retrain
lgb_fr_final = retrain_tree("lightgbm", LGB_FR_P,
    train_fe.loc[train_fe.index[valid_fr_full], feat_fr], y_dev_fr_full[valid_fr_full], si["lgb_fr"])
preds_fr_test_lgb = rm_fr_test + predict_tree(lgb_fr_final, test_fe[feat_fr])
print(f"  FR LGB retrained ({si['lgb_fr']} iter)")

# FR XGBoost retrain
xgb_fr_final = retrain_tree("xgboost", XGB_FR_P,
    train_fe.loc[train_fe.index[valid_fr_full], feat_fr], y_dev_fr_full[valid_fr_full], si["xgb_fr"])
preds_fr_test_xgb = rm_fr_test + predict_tree(xgb_fr_final, test_fe[feat_fr])
print(f"  FR XGB retrained ({si['xgb_fr']} iter)")

# FR Elastic Net retrain
en_fr_final, en_fr_scaler = retrain_elastic_net(
    train_fe.loc[train_fe.index[valid_fr_full], feat_fr].values,
    y_dev_fr_full[valid_fr_full], alpha=10.0, l1_ratio=0.9)
preds_fr_test_en = rm_fr_test + predict_elastic_net(en_fr_final, en_fr_scaler, test_fe[feat_fr].values)

# FR DNN retrain
dnn_scaler_full_fr = StandardScaler()
X_dnn_full_fr = dnn_scaler_full_fr.fit_transform(
    np.nan_to_num(train_fe.loc[train_fe.index[valid_fr_full], feat_dnn_final].values.copy(), 0))
X_dnn_test_fr = dnn_scaler_full_fr.transform(np.nan_to_num(test_fe[feat_dnn_final].values.copy(), 0))
torch.manual_seed(42); np.random.seed(42)
dnn_fr_final = ElecDNN(len(feat_dnn_final), [192, 96], dropout=0.2)
retrain_epochs_fr = si["dnn_fr_epochs"] + 5
dnn_fr_final, _ = train_dnn(dnn_fr_final, X_dnn_full_fr,
    y_dev_fr_full[valid_fr_full].astype(np.float32),
    X_dnn_full_fr[:256], y_dev_fr_full[valid_fr_full][:256].astype(np.float32),
    max_epochs=retrain_epochs_fr, patience=retrain_epochs_fr)
preds_fr_test_dnn = rm_fr_test + predict_dnn(dnn_fr_final, X_dnn_test_fr)
print(f"  FR DNN retrained ({retrain_epochs_fr} epochs)")

# FR Ridge fondamentales retrain
fund_fr_full = build_fundamental_features(train_fe, "fr")
fund_fr_test = build_fundamental_features(test_fe, "fr")
scaler_rf_full = StandardScaler()
X_rf_full = scaler_rf_full.fit_transform(np.nan_to_num(fund_fr_full.values.copy(), 0))
X_rf_test = scaler_rf_full.transform(np.nan_to_num(fund_fr_test.values.copy(), 0))
ridge_fr_full = Ridge(alpha=10.0)
ridge_fr_full.fit(X_rf_full, spot_fr_full)
preds_fr_test_ridge = ridge_fr_full.predict(X_rf_test)

# FR T2 retrain + SR (using spring meta model)
print(f"  FR T2 retraining {len(spring['t2_fr'])} models...")
t2_fr_test_preds = {}
for name, t2_info in spring["t2_fr"].items():
    feats = t2_info["feats"]
    algo = t2_info["algo"]
    retrained = retrain_t2(algo, train_fe[feats].values[valid_fr_full], y_dev_fr_full[valid_fr_full], t2_info["info"])
    t2_fr_test_preds[name] = rm_fr_test + predict_t2(retrained, test_fe[feats].values)

# FR v9 test ensemble (using spring v9 weights)
fr_v9_test = {"CB": preds_fr_test_cb, "LGB": preds_fr_test_lgb, "XGB": preds_fr_test_xgb,
              "EN": preds_fr_test_en, "DNN": preds_fr_test_dnn}
v9_ens_fr_test = apply_regime_weights(fr_v9_test, hours_test, spring["v9_regime_fr"])

# FR SR test prediction (using spring meta model)
X_sr_fr_test = np.column_stack([t2_fr_test_preds[k] for k in spring["fr_t2_names"]])
sr_correction_fr = spring["meta_fr"].predict(X_sr_fr_test)
preds_fr_test_sr = v9_ens_fr_test + sr_correction_fr

# ── UK retrain ──
uk_moc_full = train_fe["uk_merit_order_cost"].values
uk_moc_test = test_fe["uk_merit_order_cost"].values
uk_spot_full = train_fe["uk_spot"].values
y_basis_full = uk_spot_full - uk_moc_full
valid_uk_full = np.isfinite(y_basis_full)

cb_uk_final = retrain_tree("catboost", CB_UK_P,
    train_fe.loc[train_fe.index[valid_uk_full], feat_uk], y_basis_full[valid_uk_full], si["cb_uk"])
preds_uk_test_cb = uk_moc_test + predict_tree(cb_uk_final, test_fe[feat_uk])

lgb_uk_final = retrain_tree("lightgbm", LGB_UK_P,
    train_fe.loc[train_fe.index[valid_uk_full], feat_uk], y_basis_full[valid_uk_full], si["lgb_uk"])
preds_uk_test_lgb = uk_moc_test + predict_tree(lgb_uk_final, test_fe[feat_uk])

xgb_uk_final = retrain_tree("xgboost", XGB_UK_P,
    train_fe.loc[train_fe.index[valid_uk_full], feat_uk], y_basis_full[valid_uk_full], si["xgb_uk"])
preds_uk_test_xgb = uk_moc_test + predict_tree(xgb_uk_final, test_fe[feat_uk])

# UK XGB cluster retrain
hours_full = train_fe["hour"].values
xgb_cluster_test_basis = np.zeros(len(test_fe))
for cname, c_hours in CLUSTER_SPLIT_SHIFTED_6H.items():
    c_mask_full = np.isin(hours_full, c_hours) & valid_uk_full
    test_hour_mask = np.isin(hours_test, c_hours)
    best_iter = si["xgb_cluster"][cname]
    c_model = retrain_tree("xgboost", XGB_UK_CLUSTER_P,
        train_fe.loc[train_fe.index[c_mask_full], feat_uk], y_basis_full[c_mask_full], best_iter)
    xgb_cluster_test_basis[test_hour_mask] = predict_tree(c_model,
        test_fe.loc[test_fe.index[test_hour_mask], feat_uk])
preds_uk_test_xgb_cluster = uk_moc_test + xgb_cluster_test_basis

# UK Elastic Net retrain
en_uk_final, en_uk_scaler = retrain_elastic_net(
    train_fe.loc[train_fe.index[valid_uk_full], feat_uk].values,
    y_basis_full[valid_uk_full], alpha=1.0, l1_ratio=0.9)
preds_uk_test_en = uk_moc_test + predict_elastic_net(en_uk_final, en_uk_scaler, test_fe[feat_uk].values)

# UK DNN retrain
dnn_scaler_full_uk = StandardScaler()
X_dnn_full_uk = dnn_scaler_full_uk.fit_transform(
    np.nan_to_num(train_fe.loc[train_fe.index[valid_uk_full], feat_dnn_final].values.copy(), 0))
X_dnn_test_uk = dnn_scaler_full_uk.transform(np.nan_to_num(test_fe[feat_dnn_final].values.copy(), 0))
torch.manual_seed(42); np.random.seed(42)
dnn_uk_final = ElecDNN(len(feat_dnn_final), [768, 384, 192], dropout=0.3)
retrain_epochs_uk = si["dnn_uk_epochs"] + 5
dnn_uk_final, _ = train_dnn(dnn_uk_final, X_dnn_full_uk,
    y_basis_full[valid_uk_full].astype(np.float32),
    X_dnn_full_uk[:256], y_basis_full[valid_uk_full][:256].astype(np.float32),
    max_epochs=retrain_epochs_uk, patience=retrain_epochs_uk,
    criterion=torch.nn.MSELoss())
preds_uk_test_dnn = uk_moc_test + predict_dnn(dnn_uk_final, X_dnn_test_uk)

# UK Ridge fondamentales retrain
fund_uk_full = build_fundamental_features(train_fe, "uk")
fund_uk_test = build_fundamental_features(test_fe, "uk")
scaler_ruk_full = StandardScaler()
X_ruk_full = scaler_ruk_full.fit_transform(np.nan_to_num(fund_uk_full.values.copy(), 0))
X_ruk_test = scaler_ruk_full.transform(np.nan_to_num(fund_uk_test.values.copy(), 0))
ridge_uk_full = Ridge(alpha=10.0)
ridge_uk_full.fit(X_ruk_full, uk_spot_full)
preds_uk_test_ridge = ridge_uk_full.predict(X_ruk_test)

# UK T2 retrain + SR (using spring meta model)
print(f"  UK T2 retraining {len(spring['t2_uk'])} models...")
t2_uk_test_preds = {}
for name, t2_info in spring["t2_uk"].items():
    feats = t2_info["feats"]
    algo = t2_info["algo"]
    retrained = retrain_t2(algo, train_fe[feats].values[valid_uk_full], y_basis_full[valid_uk_full], t2_info["info"])
    t2_uk_test_preds[name] = uk_moc_test + predict_t2(retrained, test_fe[feats].values)

# UK v9 test ensemble (using spring v9 weights)
uk_v9_test = {"CB": preds_uk_test_cb, "LGB": preds_uk_test_lgb, "XGB": preds_uk_test_xgb,
              "EN": preds_uk_test_en, "DNN": preds_uk_test_dnn}
v9_ens_uk_test = apply_regime_weights(uk_v9_test, hours_test, spring["v9_regime_uk"])

# UK SR test prediction (enriched, using spring meta model)
X_sr_uk_test_t2 = np.column_stack([t2_uk_test_preds[k] for k in spring["uk_t2_names"]])
X_extra_test = np.column_stack([
    np.sin(2*np.pi*hours_test/24), np.cos(2*np.pi*hours_test/24),
    np.sin(2*np.pi*dow_test/7), np.cos(2*np.pi*dow_test/7),
])
X_sr_uk_test = np.hstack([X_sr_uk_test_t2, X_extra_test])
sr_correction_uk = spring["meta_uk"].predict(X_sr_uk_test)
preds_uk_test_sr = v9_ens_uk_test + sr_correction_uk

print(f"  Retrain completed in {time.time()-t_retrain:.0f}s")


# ══════════════════════════════════════════════════════════════════════════
# 4. GENERATE SUBMISSIONS (spring, winter, averaged weights)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  4. Generate 3 submissions")
print("=" * 90)

fr_test_models = {"CB": preds_fr_test_cb, "LGB": preds_fr_test_lgb, "XGB": preds_fr_test_xgb,
                  "EN": preds_fr_test_en, "DNN": preds_fr_test_dnn,
                  "RidgeF": preds_fr_test_ridge, "SR": preds_fr_test_sr}
uk_test_models = {"CB": preds_uk_test_cb, "LGB": preds_uk_test_lgb, "XGB": preds_uk_test_xgb,
                  "EN": preds_uk_test_en, "DNN": preds_uk_test_dnn,
                  "RidgeF": preds_uk_test_ridge, "SR": preds_uk_test_sr,
                  "XGB_C": preds_uk_test_xgb_cluster}

# Clipping bounds
fr_q_low = np.percentile(train_fe["fr_spot"].dropna(), 0.1)
fr_q_high = np.percentile(train_fe["fr_spot"].dropna(), 99.9)
uk_q_low = np.percentile(train_fe["uk_spot"].dropna(), 0.1)
uk_q_high = np.percentile(train_fe["uk_spot"].dropna(), 99.9)

weight_sets = [
    ("spring", spring["fr_regime_weights"], spring["uk_regime_weights"], spring["hbc_fr"], spring["hbc_uk"]),
    ("winter", winter["fr_regime_weights"], winter["uk_regime_weights"], winter["hbc_fr"], winter["hbc_uk"]),
    ("averaged", avg_fr_weights, avg_uk_weights, avg_hbc_fr, avg_hbc_uk),
]

submissions = {}
for wname, fr_w, uk_w, hbc_fr, hbc_uk in weight_sets:
    preds_fr = apply_regime_weights(fr_test_models, hours_test, fr_w)
    preds_fr_hbc = preds_fr + np.array([hbc_fr.get(h, 0) for h in hours_test])
    preds_uk = apply_regime_weights(uk_test_models, hours_test, uk_w)
    preds_uk_hbc = preds_uk + np.array([hbc_uk.get(h, 0) for h in hours_test])

    sub = pd.DataFrame({
        "id": test_fe.index,
        "fr_spot": np.clip(preds_fr_hbc, fr_q_low, fr_q_high),
        "uk_spot": np.clip(preds_uk_hbc, uk_q_low, uk_q_high),
    })
    fname = f"outputs/submission_attack_{wname}.csv"
    sub.to_csv(fname, index=False)
    submissions[wname] = sub
    print(f"  {wname:10s}: FR mean={preds_fr_hbc.mean():.1f} std={preds_fr_hbc.std():.1f} | UK mean={preds_uk_hbc.mean():.1f} std={preds_uk_hbc.std():.1f}")
    print(f"              -> {fname}")

# Also compute pairwise differences between submissions
print(f"\n  Submission differences (RMSE between submissions):")
for w1, w2 in [("spring", "winter"), ("spring", "averaged"), ("winter", "averaged")]:
    diff_fr = np.sqrt(np.mean((submissions[w1]["fr_spot"].values - submissions[w2]["fr_spot"].values)**2))
    diff_uk = np.sqrt(np.mean((submissions[w1]["uk_spot"].values - submissions[w2]["uk_spot"].values)**2))
    print(f"  {w1} vs {w2}: FR={diff_fr:.3f}, UK={diff_uk:.3f}")


# ══════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  FINAL SUMMARY — WINTER HOLDOUT RECALIBRATION")
print("=" * 90)

print(f"\n  Holdout validation metrics:")
print(f"    SPRING: FR={spring['val_metrics']['fr_hbc']:.4f} UK={spring['val_metrics']['uk_hbc']:.4f} SUM={spring['val_metrics']['sum_hbc']:.4f}")
print(f"    WINTER: FR={winter['val_metrics']['fr_hbc']:.4f} UK={winter['val_metrics']['uk_hbc']:.4f} SUM={winter['val_metrics']['sum_hbc']:.4f}")

print(f"\n  3 submissions generated:")
print(f"    outputs/submission_attack_spring.csv  (= v17 baseline)")
print(f"    outputs/submission_attack_winter.csv  (winter-calibrated weights)")
print(f"    outputs/submission_attack_averaged.csv (avg spring+winter)")

print(f"\n  Next: submit all 3 to Kaggle to measure impact")
print(f"  Total time: {time.time()-t0:.0f}s")

# Save results as JSON
results = {
    "spring_val": spring["val_metrics"],
    "winter_val": winter["val_metrics"],
    "spring_iters": spring["model_iters"],
    "winter_iters": winter["model_iters"],
}
# Convert non-serializable types
for key in results:
    if isinstance(results[key], dict):
        for k, v in results[key].items():
            if isinstance(v, (np.integer, np.int64)):
                results[key][k] = int(v)
            elif isinstance(v, dict):
                results[key][k] = {kk: int(vv) if isinstance(vv, (np.integer, np.int64)) else vv for kk, vv in v.items()}

with open("outputs/attack_winter_holdout_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"  Results saved to outputs/attack_winter_holdout_results.json")
