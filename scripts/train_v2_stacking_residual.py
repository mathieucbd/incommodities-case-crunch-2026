"""Final pipeline v11 — v9 + Ridge + Residual Stacking T2 (per-country optimized).

v11 architecture:
  Layer 1: v9 models (CB, LGB, XGB, EN, DNN) per market — unchanged from v9
  Layer 1b: Ridge fundamentals (RidgeF) per market
  Layer 2: T2 models (thematic groups × lightweight algos)
    FR: 4 algos (no cb_small) × 9 groups (split fr_load) + 3 Ridge combos
    UK: 4 algos (no lgb_small) × 9 groups (split uk_wind) + 3 Ridge combos
  Stacking: Residual stacking — Ridge meta-learner predicts v9 ensemble errors
    FR: standard Ridge, alpha=1
    UK: enriched Ridge (+ hour_sin/cos, dow_sin/cos), alpha=500 (conservative)
  Ensemble: regime-weighted (7 models: CB+LGB+XGB+EN+DNN+RidgeF+SR) + HBC

Validation scores (conservative anti-overfitting):
  FR: ~14.86  UK: ~8.99  SUM: ~23.85
  Gap vs leader (23.14): ~0.71

Usage: cd incommodities-case-crunch-2026 && PYTHONIOENCODING=utf-8 python -u scripts/train_v2_stacking_residual.py
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
    prepare_stationary,
    REGIMES, optimize_regime_weights, apply_regime_weights,
    train_tree, retrain_tree, predict_tree,
    train_elastic_net, retrain_elastic_net, predict_elastic_net,
    ElecDNN, DNN_DEVICE, train_dnn, predict_dnn,
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.model_selection import KFold
from itertools import combinations
import lightgbm as lgb

warnings.filterwarnings("ignore")

with open("config.yaml") as f:
    config = yaml.safe_load(f)

# ══════════════════════════════════════════════════════════════════════════
# 0. LOAD DATA + FEATURES
# ══════════════════════════════════════════════════════════════════════════
print("=" * 90)
print("  FINAL PIPELINE v11 — v9 + Ridge + SR T2 (per-country)")
print("=" * 90)

t0 = time.time()
x_train, y_train, x_test = load_data("data/raw")
train_fe = build_features(pd.concat([x_train], axis=0), config)
train_fe = train_fe.join(y_train[["fr_spot", "uk_spot"]])
test_fe = build_features(x_test, config)

print(f"  Data loaded in {time.time() - t0:.0f}s")

holdout_start = config["validation"]["holdout_start"]
mask_val = train_fe["datetime_CET"] >= holdout_start
df_train = train_fe[~mask_val].copy()
df_val = train_fe[mask_val].copy()

# Interaction feature
for df in [train_fe, df_train, df_val, test_fe]:
    if "fr_spot_la_roll_168h_mean" in df.columns and "uk_price_per_mw_7d" in df.columns:
        df["X_fr_spot_la_roll_168h_mean_x_uk_price_per_mw_7d"] = (
            df["fr_spot_la_roll_168h_mean"] * df["uk_price_per_mw_7d"]
        )

df_train_uk = df_train.copy()

print(f"  Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(test_fe)}")

# ── Feature lists ────────────────────────────────────────────────────────
with open("outputs/feature_selection_v5_fr.json") as f:
    fs_v5 = json.load(f)
feat_fr = [f for f in fs_v5["features"] + ["X_fr_spot_la_roll_168h_mean_x_uk_price_per_mw_7d"]
           if f in df_train.columns]

with open("outputs/uk_feature_research.json") as f:
    uk_research = json.load(f)
feat_uk = [f for f in uk_research["confirmed_features"] if f in df_train_uk.columns]

# DNN features (all numeric, dedup corr > 0.99)
_EXCLUDE = {"fr_spot", "uk_spot", "datetime_CET", "datetime_UTC", "date", "id"}
_all_num = [c for c in df_train.columns
            if c not in _EXCLUDE
            and df_train[c].dtype in ["float64", "float32", "int64", "int32"]
            and df_train[c].notna().sum() > len(df_train) * 0.5]
_corr = df_train[_all_num].corr().abs()
_to_drop = set()
for _i in range(len(_all_num)):
    if _all_num[_i] in _to_drop: continue
    for _j in range(_i + 1, len(_all_num)):
        if _all_num[_j] in _to_drop: continue
        if _corr.iloc[_i, _j] > 0.99:
            _to_drop.add(_all_num[_j])
feat_dnn = [f for f in _all_num if f not in _to_drop]
feat_dnn_final = [f for f in feat_dnn if f in df_train.columns]

print(f"  FR features: {len(feat_fr)}, UK features: {len(feat_uk)}, DNN features: {len(feat_dnn_final)}")

# ── Targets ──────────────────────────────────────────────────────────────
fr_stat = prepare_stationary("fr_spot_la", "fr_spot", train_fe, df_train, df_val)
spot_va_fr = df_val["fr_spot"].values
hours_va = df_val["hour"].values
dow_va = pd.to_datetime(df_val["datetime_CET"]).dt.dayofweek.values

uk_spot_tr = df_train_uk["uk_spot"].values
uk_spot_va = df_val["uk_spot"].values
uk_moc_tr = df_train_uk["uk_merit_order_cost"].values
uk_moc_va = df_val["uk_merit_order_cost"].values
y_basis_tr = uk_spot_tr - uk_moc_tr
y_basis_va = uk_spot_va - uk_moc_va
valid_basis_tr = np.isfinite(y_basis_tr)
valid_basis_va = np.isfinite(y_basis_va)
vb = valid_basis_va


# ══════════════════════════════════════════════════════════════════════════
# 1. V9 MODELS — Train on train, predict val
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  1. V9 models (CB + LGB + XGB + EN + DNN)")
print("=" * 90)

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

t1 = time.time()

# FR models
cb_fr = train_tree("catboost", CB_FR_P,
    df_train.loc[df_train.index[fr_stat["valid_tr"]], feat_fr].values,
    fr_stat["y_dev_tr"][fr_stat["valid_tr"]],
    df_val.loc[df_val.index[fr_stat["valid_va"]], feat_fr].values,
    fr_stat["y_dev_va"][fr_stat["valid_va"]])
preds_fr_cb = fr_stat["rm_va"] + predict_tree(cb_fr.model, df_val[feat_fr].values)

lgb_fr = train_tree("lightgbm", LGB_FR_P,
    df_train.loc[df_train.index[fr_stat["valid_tr"]], feat_fr].values,
    fr_stat["y_dev_tr"][fr_stat["valid_tr"]],
    df_val.loc[df_val.index[fr_stat["valid_va"]], feat_fr].values,
    fr_stat["y_dev_va"][fr_stat["valid_va"]])
preds_fr_lgb = fr_stat["rm_va"] + predict_tree(lgb_fr.model, df_val[feat_fr].values)

xgb_fr = train_tree("xgboost", XGB_FR_P,
    df_train.loc[df_train.index[fr_stat["valid_tr"]], feat_fr].values,
    fr_stat["y_dev_tr"][fr_stat["valid_tr"]],
    df_val.loc[df_val.index[fr_stat["valid_va"]], feat_fr].values,
    fr_stat["y_dev_va"][fr_stat["valid_va"]])
preds_fr_xgb = fr_stat["rm_va"] + predict_tree(xgb_fr.model, df_val[feat_fr].values)

en_fr = train_elastic_net(
    df_train.loc[df_train.index[fr_stat["valid_tr"]], feat_fr].values,
    fr_stat["y_dev_tr"][fr_stat["valid_tr"]],
    df_val[feat_fr].values, alpha=10.0, l1_ratio=0.9)
preds_fr_en = fr_stat["rm_va"] + en_fr.preds_val

dnn_scaler_fr = StandardScaler()
X_dnn_tr_fr = dnn_scaler_fr.fit_transform(np.nan_to_num(df_train[feat_dnn_final].values, 0))
X_dnn_va_fr = dnn_scaler_fr.transform(np.nan_to_num(df_val[feat_dnn_final].values, 0))
torch.manual_seed(42); np.random.seed(42)
dnn_fr = ElecDNN(len(feat_dnn_final), [192, 96], dropout=0.2)
dnn_fr, dnn_fr_epochs = train_dnn(
    dnn_fr, X_dnn_tr_fr[fr_stat["valid_tr"]], fr_stat["y_dev_tr"][fr_stat["valid_tr"]].astype(np.float32),
    X_dnn_va_fr[fr_stat["valid_va"]], fr_stat["y_dev_va"][fr_stat["valid_va"]].astype(np.float32))
preds_fr_dnn = fr_stat["rm_va"] + predict_dnn(dnn_fr, X_dnn_va_fr)

# UK models
cb_uk = train_tree("catboost", CB_UK_P,
    df_train_uk.loc[df_train_uk.index[valid_basis_tr], feat_uk].values,
    y_basis_tr[valid_basis_tr],
    df_val.loc[df_val.index[valid_basis_va], feat_uk].values,
    y_basis_va[valid_basis_va])
preds_uk_cb = uk_moc_va + predict_tree(cb_uk.model, df_val[feat_uk].values)

lgb_uk = train_tree("lightgbm", LGB_UK_P,
    df_train_uk.loc[df_train_uk.index[valid_basis_tr], feat_uk].values,
    y_basis_tr[valid_basis_tr],
    df_val.loc[df_val.index[valid_basis_va], feat_uk].values,
    y_basis_va[valid_basis_va])
preds_uk_lgb = uk_moc_va + predict_tree(lgb_uk.model, df_val[feat_uk].values)

xgb_uk = train_tree("xgboost", XGB_UK_P,
    df_train_uk.loc[df_train_uk.index[valid_basis_tr], feat_uk].values,
    y_basis_tr[valid_basis_tr],
    df_val.loc[df_val.index[valid_basis_va], feat_uk].values,
    y_basis_va[valid_basis_va])
preds_uk_xgb = uk_moc_va + predict_tree(xgb_uk.model, df_val[feat_uk].values)

en_uk = train_elastic_net(
    df_train_uk.loc[df_train_uk.index[valid_basis_tr], feat_uk].values,
    y_basis_tr[valid_basis_tr],
    df_val[feat_uk].values, alpha=1.0, l1_ratio=0.9)
preds_uk_en = uk_moc_va + en_uk.preds_val

dnn_scaler_uk = StandardScaler()
X_dnn_tr_uk = dnn_scaler_uk.fit_transform(np.nan_to_num(df_train_uk[feat_dnn_final].values, 0))
X_dnn_va_uk = dnn_scaler_uk.transform(np.nan_to_num(df_val[feat_dnn_final].values, 0))
torch.manual_seed(42); np.random.seed(42)
dnn_uk = ElecDNN(len(feat_dnn_final), [768, 384, 192], dropout=0.3)
dnn_uk, dnn_uk_epochs = train_dnn(
    dnn_uk, X_dnn_tr_uk[valid_basis_tr], y_basis_tr[valid_basis_tr].astype(np.float32),
    X_dnn_va_uk[valid_basis_va], y_basis_va[valid_basis_va].astype(np.float32),
    criterion=torch.nn.MSELoss())
preds_uk_dnn = uk_moc_va + predict_dnn(dnn_uk, X_dnn_va_uk)

print(f"  V9 models trained in {time.time()-t1:.0f}s")
print(f"  DNN epochs: FR={dnn_fr_epochs}, UK={dnn_uk_epochs}")


# ══════════════════════════════════════════════════════════════════════════
# 2. RIDGE FONDAMENTALES + V9 ENSEMBLE
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  2. Ridge fondamentales + v9 ensemble")
print("=" * 90)

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

fund_fr_tr = build_fundamental_features(df_train, "fr")
fund_fr_va = build_fundamental_features(df_val, "fr")
scaler_ridge_fr = StandardScaler()
X_rf_tr = scaler_ridge_fr.fit_transform(np.nan_to_num(fund_fr_tr.values, 0))
X_rf_va = scaler_ridge_fr.transform(np.nan_to_num(fund_fr_va.values, 0))
ridge_fr = Ridge(alpha=10.0)
ridge_fr.fit(X_rf_tr, df_train["fr_spot"].values)
preds_fr_ridge = ridge_fr.predict(X_rf_va)

fund_uk_tr = build_fundamental_features(df_train_uk, "uk")
fund_uk_va = build_fundamental_features(df_val, "uk")
scaler_ridge_uk = StandardScaler()
X_ruk_tr = scaler_ridge_uk.fit_transform(np.nan_to_num(fund_uk_tr.values, 0))
X_ruk_va = scaler_ridge_uk.transform(np.nan_to_num(fund_uk_va.values, 0))
ridge_uk = Ridge(alpha=10.0)
ridge_uk.fit(X_ruk_tr, df_train_uk["uk_spot"].values)
preds_uk_ridge = ridge_uk.predict(X_ruk_va)

# v9 ensemble (for residual computation)
v9_fr = {"CB": preds_fr_cb, "LGB": preds_fr_lgb, "XGB": preds_fr_xgb,
         "EN": preds_fr_en, "DNN": preds_fr_dnn}
v9_uk = {"CB": preds_uk_cb[vb], "LGB": preds_uk_lgb[vb], "XGB": preds_uk_xgb[vb],
         "EN": preds_uk_en[vb], "DNN": preds_uk_dnn[vb]}

v9_regime_fr, v9_ens_fr = optimize_regime_weights(v9_fr, spot_va_fr, hours_va, "FR_v9", step=0.1)
v9_regime_uk, v9_ens_uk = optimize_regime_weights(v9_uk, uk_spot_va[vb], hours_va[vb], "UK_v9", step=0.1)

print(f"  V9 ensemble: FR={compute_rmse(spot_va_fr, v9_ens_fr):.2f}, UK={compute_rmse(uk_spot_va[vb], v9_ens_uk):.2f}")


# ══════════════════════════════════════════════════════════════════════════
# 3. T2 MODELS — Per-country optimized groups
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  3. T2 models (per-country optimized groups)")
print("=" * 90)

def filter_existing(features, df):
    return [f for f in features if f in df.columns]

# FR: 9 groups (sans continent/hydro/nuclear, fr_load split into raw + residual)
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

# FR combo pairs (top-3 greedy — conservative to reduce overfitting)
FR_COMBO_PAIRS = [
    ("fr_price", "fr_calendar"),
    ("fr_renewable", "fr_load_raw"),
    ("fr_renewable", "fr_load_residual"),
]

# UK: 9 groups (sans gas/load/scarcity, uk_wind split into core + continent)
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

# UK combo pairs (top-3 greedy — conservative to reduce overfitting)
UK_COMBO_PAIRS = [
    ("uk_price", "uk_fr_price"),
    ("uk_wind_core", "uk_nuclear_fr"),
    ("uk_nuclear_fr", "uk_calendar"),
]

fr_groups = {k: filter_existing(v, df_train) for k, v in FR_GROUPS.items()}
uk_groups = {k: filter_existing(v, df_train) for k, v in UK_GROUPS.items()}
fr_groups = {k: v for k, v in fr_groups.items() if len(v) >= 3}
uk_groups = {k: v for k, v in uk_groups.items() if len(v) >= 3}

# T2 algo params
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

# FR: 4 algos (sans cb_small)
FR_T2_ALGOS = ["ridge", "elasticnet", "lgb_small", "xgb_small"]
# UK: 4 algos (sans lgb_small)
UK_T2_ALGOS = ["ridge", "elasticnet", "cb_small", "xgb_small"]


def train_t2(algo, X_tr, y_tr, X_va, y_va=None):
    """Train a T2 model, return (val_predictions, model_info_for_retrain)."""
    if algo == "ridge":
        scaler = StandardScaler()
        Xts = scaler.fit_transform(np.nan_to_num(X_tr, 0))
        Xvs = scaler.transform(np.nan_to_num(X_va, 0))
        m = Ridge(alpha=10.0); m.fit(Xts, y_tr)
        return m.predict(Xvs), {"model": m, "scaler": scaler, "algo": algo}
    elif algo == "elasticnet":
        scaler = StandardScaler()
        Xts = scaler.fit_transform(np.nan_to_num(X_tr, 0))
        Xvs = scaler.transform(np.nan_to_num(X_va, 0))
        m = ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=5000); m.fit(Xts, y_tr)
        return m.predict(Xvs), {"model": m, "scaler": scaler, "algo": algo}
    elif algo == "lgb_small":
        ds = lgb.Dataset(np.nan_to_num(X_tr, 0), label=y_tr)
        params = {"objective": "mae", "max_depth": 3, "learning_rate": 0.05,
                  "num_leaves": 8, "min_child_samples": 50,
                  "subsample": 0.8, "colsample_bytree": 0.8, "verbose": -1}
        m = lgb.train(params, ds, num_boost_round=500)
        return m.predict(np.nan_to_num(X_va, 0)), {"model": m, "algo": algo, "params": params}
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
    """Retrain T2 model on full data, return model that can predict."""
    if algo in ("ridge", "elasticnet"):
        scaler = StandardScaler()
        Xfs = scaler.fit_transform(np.nan_to_num(X_full, 0))
        if algo == "ridge":
            m = Ridge(alpha=10.0)
        else:
            m = ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=5000)
        m.fit(Xfs, y_full)
        return {"model": m, "scaler": scaler}
    elif algo == "lgb_small":
        ds = lgb.Dataset(np.nan_to_num(X_full, 0), label=y_full)
        m = lgb.train(info["params"], ds, num_boost_round=500)
        return {"model": m}
    elif algo == "cb_small":
        from catboost import CatBoostRegressor, Pool
        params = {**CB_SMALL}
        params["use_best_model"] = False
        params["iterations"] = info.get("best_iter", 500)
        m = CatBoostRegressor(**params)
        m.fit(Pool(np.nan_to_num(X_full.copy(), 0), np.array(y_full, dtype=np.float64)), verbose=0)
        return {"model": m}
    elif algo == "xgb_small":
        import xgboost as xgb_lib
        params = {**XGB_SMALL}
        params["n_estimators"] = info.get("best_iter", 500)
        m = xgb_lib.XGBRegressor(**params)
        m.fit(np.nan_to_num(X_full.copy(), 0), np.array(y_full, dtype=np.float64), verbose=False)
        return {"model": m}


def predict_t2(retrained, X_test):
    """Predict with retrained T2 model."""
    if "scaler" in retrained:
        return retrained["model"].predict(retrained["scaler"].transform(np.nan_to_num(X_test, 0)))
    else:
        return retrained["model"].predict(np.nan_to_num(X_test, 0))


# Train FR T2 models
t2_start = time.time()
t2_fr = {}  # name -> {"preds": ..., "group": ..., "algo": ..., "info": ...}

for gname, feats in fr_groups.items():
    X_tr = df_train[feats].values[fr_stat["valid_tr"]]
    X_va = df_val[feats].values
    y_tr = fr_stat["y_dev_tr"][fr_stat["valid_tr"]]
    y_va = fr_stat["y_dev_va"][fr_stat["valid_va"]]
    for algo in FR_T2_ALGOS:
        try:
            preds_dev, info = train_t2(algo, X_tr, y_tr, X_va,
                                        y_va=y_va if algo in ("cb_small", "xgb_small") else None)
            preds_spot = fr_stat["rm_va"] + preds_dev
            t2_fr[f"{gname}__{algo}"] = {"preds": preds_spot, "group": gname, "algo": algo,
                                          "info": info, "feats": feats}
        except Exception as e:
            print(f"    ERR FR {gname} {algo}: {e}")

# Train UK T2 models
t2_uk = {}

for gname, feats in uk_groups.items():
    X_tr = df_train_uk[feats].values[valid_basis_tr]
    X_va = df_val[feats].values
    y_tr = y_basis_tr[valid_basis_tr]
    y_va = y_basis_va[valid_basis_va]
    for algo in UK_T2_ALGOS:
        try:
            preds_basis, info = train_t2(algo, X_tr, y_tr, X_va,
                                          y_va=y_va if algo in ("cb_small", "xgb_small") else None)
            preds_spot = uk_moc_va + preds_basis
            t2_uk[f"{gname}__{algo}"] = {"preds": preds_spot, "group": gname, "algo": algo,
                                          "info": info, "feats": feats}
        except Exception as e:
            print(f"    ERR UK {gname} {algo}: {e}")

# FR combos (ridge only, greedy-selected pairs)
for g1, g2 in FR_COMBO_PAIRS:
    if g1 not in fr_groups or g2 not in fr_groups:
        continue
    feats = list(set(fr_groups[g1] + fr_groups[g2]))
    X_tr = df_train[feats].values[fr_stat["valid_tr"]]
    X_va = df_val[feats].values
    preds_dev, info = train_t2("ridge", X_tr, fr_stat["y_dev_tr"][fr_stat["valid_tr"]], X_va)
    preds_spot = fr_stat["rm_va"] + preds_dev
    t2_fr[f"combo_{g1}+{g2}__ridge"] = {"preds": preds_spot, "group": f"{g1}+{g2}",
                                          "algo": "ridge", "info": info, "feats": feats}

# UK combos (ridge only, greedy-selected pairs)
for g1, g2 in UK_COMBO_PAIRS:
    if g1 not in uk_groups or g2 not in uk_groups:
        continue
    feats = list(set(uk_groups[g1] + uk_groups[g2]))
    X_tr = df_train_uk[feats].values[valid_basis_tr]
    X_va = df_val[feats].values
    preds_basis, info = train_t2("ridge", X_tr, y_basis_tr[valid_basis_tr], X_va)
    preds_spot = uk_moc_va + preds_basis
    t2_uk[f"combo_{g1}+{g2}__ridge"] = {"preds": preds_spot, "group": f"{g1}+{g2}",
                                          "algo": "ridge", "info": info, "feats": feats}

print(f"  T2: {len(t2_fr)} FR + {len(t2_uk)} UK ({time.time()-t2_start:.0f}s)")


# ══════════════════════════════════════════════════════════════════════════
# 4. STACKING RÉSIDUEL + ENSEMBLE OPTIMIZATION
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  4. Stacking résiduel + regime ensemble optimization")
print("=" * 90)

def make_vb(d):
    return {k: {"preds": v["preds"][vb], **{kk: vv for kk, vv in v.items() if kk != "preds"}}
            for k, v in d.items()}

# FR: standard stacking residuel, alpha=1
n_fr = len(spot_va_fr)
residuals_fr = spot_va_fr - v9_ens_fr[:n_fr]
fr_t2_names = sorted(t2_fr.keys())
X_sr_fr = np.column_stack([t2_fr[k]["preds"][:n_fr] for k in fr_t2_names])

# 5-fold CV for OOF predictions (used for regime weight optimization)
oof_fr = np.zeros(n_fr)
kf = KFold(n_splits=5, shuffle=False)
for tr_idx, va_idx in kf.split(X_sr_fr):
    meta = Ridge(alpha=1.0)
    meta.fit(X_sr_fr[tr_idx], residuals_fr[tr_idx])
    oof_fr[va_idx] = meta.predict(X_sr_fr[va_idx])
sr_fr_val = v9_ens_fr[:n_fr] + oof_fr

# Full-fit meta-learner (for test predictions)
meta_fr = Ridge(alpha=1.0)
meta_fr.fit(X_sr_fr, residuals_fr)

# UK: enriched stacking residuel, alpha=100
t2_uk_vb = make_vb(t2_uk)
n_uk = vb.sum()
residuals_uk = uk_spot_va[vb] - v9_ens_uk[:n_uk]
uk_t2_names = sorted(t2_uk_vb.keys())
X_sr_uk_t2 = np.column_stack([t2_uk_vb[k]["preds"][:n_uk] for k in uk_t2_names])

# Enriched: add temporal features
h_uk = hours_va[vb][:n_uk]; d_uk = dow_va[vb][:n_uk]
X_extra_uk = np.column_stack([
    np.sin(2 * np.pi * h_uk / 24), np.cos(2 * np.pi * h_uk / 24),
    np.sin(2 * np.pi * d_uk / 7), np.cos(2 * np.pi * d_uk / 7),
])
X_sr_uk = np.hstack([X_sr_uk_t2, X_extra_uk])

# 5-fold CV for OOF
oof_uk = np.zeros(n_uk)
kf = KFold(n_splits=5, shuffle=False)
for tr_idx, va_idx in kf.split(X_sr_uk):
    meta = Ridge(alpha=500.0)
    meta.fit(X_sr_uk[tr_idx], residuals_uk[tr_idx])
    oof_uk[va_idx] = meta.predict(X_sr_uk[va_idx])
sr_uk_val = v9_ens_uk[:n_uk] + oof_uk

# Full-fit meta-learner (for test predictions)
meta_uk = Ridge(alpha=500.0)
meta_uk.fit(X_sr_uk, residuals_uk)

# Regime ensemble: 7 models (CB+LGB+XGB+EN+DNN+RidgeF+SR)
v9r_fr = {"CB": preds_fr_cb, "LGB": preds_fr_lgb, "XGB": preds_fr_xgb,
           "EN": preds_fr_en, "DNN": preds_fr_dnn, "RidgeF": preds_fr_ridge}
v9r_uk = {"CB": preds_uk_cb[vb], "LGB": preds_uk_lgb[vb], "XGB": preds_uk_xgb[vb],
           "EN": preds_uk_en[vb], "DNN": preds_uk_dnn[vb], "RidgeF": preds_uk_ridge[vb]}

d_fr = {**v9r_fr, "SR": sr_fr_val}
d_uk = {**v9r_uk, "SR": sr_uk_val}

print(f"\n  FR per-regime weights (7 models):")
fr_regime_weights, preds_fr_ens = optimize_regime_weights(d_fr, spot_va_fr, hours_va, "FR_v11", step=0.1)
_, rmse_fr_hbc = compute_hbc(preds_fr_ens, spot_va_fr, hours_va)
print(f"    +HBC={rmse_fr_hbc:.4f}")

print(f"\n  UK per-regime weights (7 models):")
uk_regime_weights, preds_uk_ens = optimize_regime_weights(d_uk, uk_spot_va[vb], hours_va[vb], "UK_v11", step=0.1)
_, rmse_uk_hbc = compute_hbc(preds_uk_ens, uk_spot_va[vb], hours_va[vb])
print(f"    +HBC={rmse_uk_hbc:.4f}")

val_sum = rmse_fr_hbc + rmse_uk_hbc
print(f"\n  VALIDATION SUM: {val_sum:.4f} (FR={rmse_fr_hbc:.4f} + UK={rmse_uk_hbc:.4f})")


# ══════════════════════════════════════════════════════════════════════════
# 5. HBC CALIBRATION
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  5. HBC calibration")
print("=" * 90)

hbc_fr_final, _ = compute_hbc(preds_fr_ens, spot_va_fr, hours_va)
hbc_uk_final, _ = compute_hbc(preds_uk_ens, uk_spot_va[vb], hours_va[vb])

print(f"  FR HBC corrections: min={min(hbc_fr_final.values()):+.2f}, max={max(hbc_fr_final.values()):+.2f}")
print(f"  UK HBC corrections: min={min(hbc_uk_final.values()):+.2f}, max={max(hbc_uk_final.values()):+.2f}")


# ══════════════════════════════════════════════════════════════════════════
# 6. RETRAIN ON FULL DATA + PREDICT TEST
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  6. Retrain on FULL data + generate test predictions")
print("=" * 90)

# ── FR retrain ──
all_data = pd.concat([train_fe, test_fe], axis=0)
fr_la_all = all_data["fr_spot_la"]
rm_fr_all = fr_la_all.ewm(span=240).mean().values
n_full = len(train_fe)
rm_fr_test = rm_fr_all[n_full:]
spot_fr_full = train_fe["fr_spot"].values
y_dev_fr_full = spot_fr_full - rm_fr_all[:n_full]
valid_fr_full = np.isfinite(y_dev_fr_full)

hours_test = test_fe["hour"].values
dow_test = pd.to_datetime(test_fe["datetime_CET"]).dt.dayofweek.values

# Interaction feature for test
if "fr_spot_la_roll_168h_mean" in test_fe.columns and "uk_price_per_mw_7d" in test_fe.columns:
    test_fe["X_fr_spot_la_roll_168h_mean_x_uk_price_per_mw_7d"] = (
        test_fe["fr_spot_la_roll_168h_mean"] * test_fe["uk_price_per_mw_7d"]
    )

# FR CatBoost retrain
cb_fr_final = retrain_tree("catboost", CB_FR_P,
    train_fe.loc[train_fe.index[valid_fr_full], feat_fr],
    y_dev_fr_full[valid_fr_full], cb_fr.best_iteration)
preds_fr_test_cb = rm_fr_test + predict_tree(cb_fr_final, test_fe[feat_fr])
print(f"  FR CatBoost: retrained ({cb_fr.best_iteration} iter)")

# FR LightGBM retrain
lgb_fr_final = retrain_tree("lightgbm", LGB_FR_P,
    train_fe.loc[train_fe.index[valid_fr_full], feat_fr],
    y_dev_fr_full[valid_fr_full], lgb_fr.best_iteration)
preds_fr_test_lgb = rm_fr_test + predict_tree(lgb_fr_final, test_fe[feat_fr])
print(f"  FR LightGBM: retrained ({lgb_fr.best_iteration} iter)")

# FR XGBoost retrain
xgb_fr_final = retrain_tree("xgboost", XGB_FR_P,
    train_fe.loc[train_fe.index[valid_fr_full], feat_fr],
    y_dev_fr_full[valid_fr_full], xgb_fr.best_iteration)
preds_fr_test_xgb = rm_fr_test + predict_tree(xgb_fr_final, test_fe[feat_fr])
print(f"  FR XGBoost: retrained ({xgb_fr.best_iteration} iter)")

# FR Elastic Net retrain
en_fr_final, en_fr_scaler = retrain_elastic_net(
    train_fe.loc[train_fe.index[valid_fr_full], feat_fr].values,
    y_dev_fr_full[valid_fr_full], alpha=10.0, l1_ratio=0.9)
preds_fr_test_en = rm_fr_test + predict_elastic_net(en_fr_final, en_fr_scaler, test_fe[feat_fr].values)
print(f"  FR Elastic Net: retrained")

# FR DNN retrain
dnn_scaler_full_fr = StandardScaler()
X_dnn_full_fr = dnn_scaler_full_fr.fit_transform(
    np.nan_to_num(train_fe.loc[train_fe.index[valid_fr_full], feat_dnn_final].values, 0))
X_dnn_test_fr = dnn_scaler_full_fr.transform(np.nan_to_num(test_fe[feat_dnn_final].values, 0))
torch.manual_seed(42); np.random.seed(42)
dnn_fr_final = ElecDNN(len(feat_dnn_final), [192, 96], dropout=0.2)
retrain_epochs_fr = dnn_fr_epochs + 5
dnn_fr_final, _ = train_dnn(dnn_fr_final, X_dnn_full_fr,
    y_dev_fr_full[valid_fr_full].astype(np.float32),
    X_dnn_full_fr[:256], y_dev_fr_full[valid_fr_full][:256].astype(np.float32),
    max_epochs=retrain_epochs_fr, patience=retrain_epochs_fr)
preds_fr_test_dnn = rm_fr_test + predict_dnn(dnn_fr_final, X_dnn_test_fr)
print(f"  FR DNN: retrained ({retrain_epochs_fr} epochs)")

# FR Ridge fondamentales retrain
fund_fr_full = build_fundamental_features(train_fe, "fr")
fund_fr_test = build_fundamental_features(test_fe, "fr")
scaler_rf_full = StandardScaler()
X_rf_full = scaler_rf_full.fit_transform(np.nan_to_num(fund_fr_full.values, 0))
X_rf_test = scaler_rf_full.transform(np.nan_to_num(fund_fr_test.values, 0))
ridge_fr_full = Ridge(alpha=10.0)
ridge_fr_full.fit(X_rf_full, spot_fr_full)
preds_fr_test_ridge = ridge_fr_full.predict(X_rf_test)
print(f"  FR Ridge: retrained")

# FR T2 retrain
print(f"  FR T2: retraining {len(t2_fr)} models...")
t2_fr_test_preds = {}
for name, t2_info in t2_fr.items():
    feats = t2_info["feats"]
    algo = t2_info["algo"]
    X_full = train_fe[feats].values[valid_fr_full]
    y_full = y_dev_fr_full[valid_fr_full]
    retrained = retrain_t2(algo, X_full, y_full, t2_info["info"])
    preds_dev = predict_t2(retrained, test_fe[feats].values)
    t2_fr_test_preds[name] = rm_fr_test + preds_dev

# FR v9 test ensemble (for SR computation)
fr_v9_test = {"CB": preds_fr_test_cb, "LGB": preds_fr_test_lgb, "XGB": preds_fr_test_xgb,
              "EN": preds_fr_test_en, "DNN": preds_fr_test_dnn}
v9_ens_fr_test = apply_regime_weights(fr_v9_test, hours_test, v9_regime_fr)

# FR SR test prediction
X_sr_fr_test = np.column_stack([t2_fr_test_preds[k] for k in fr_t2_names])
sr_correction_fr = meta_fr.predict(X_sr_fr_test)
preds_fr_test_sr = v9_ens_fr_test + sr_correction_fr

# FR final ensemble
fr_test_models = {"CB": preds_fr_test_cb, "LGB": preds_fr_test_lgb, "XGB": preds_fr_test_xgb,
                  "EN": preds_fr_test_en, "DNN": preds_fr_test_dnn,
                  "RidgeF": preds_fr_test_ridge, "SR": preds_fr_test_sr}
preds_fr_test = apply_regime_weights(fr_test_models, hours_test, fr_regime_weights)
preds_fr_test_hbc = preds_fr_test + np.array([hbc_fr_final.get(h, 0) for h in hours_test])
print(f"  FR test: mean={preds_fr_test_hbc.mean():.1f}, std={preds_fr_test_hbc.std():.1f}")

# ── UK retrain ──
uk_moc_full = train_fe["uk_merit_order_cost"].values
uk_moc_test = test_fe["uk_merit_order_cost"].values
uk_spot_full = train_fe["uk_spot"].values
y_basis_full = uk_spot_full - uk_moc_full
valid_uk_full = np.isfinite(y_basis_full)

# UK CatBoost retrain
cb_uk_final = retrain_tree("catboost", CB_UK_P,
    train_fe.loc[train_fe.index[valid_uk_full], feat_uk],
    y_basis_full[valid_uk_full], cb_uk.best_iteration)
preds_uk_test_cb = uk_moc_test + predict_tree(cb_uk_final, test_fe[feat_uk])
print(f"  UK CatBoost: retrained ({cb_uk.best_iteration} iter)")

# UK LightGBM retrain
lgb_uk_final = retrain_tree("lightgbm", LGB_UK_P,
    train_fe.loc[train_fe.index[valid_uk_full], feat_uk],
    y_basis_full[valid_uk_full], lgb_uk.best_iteration)
preds_uk_test_lgb = uk_moc_test + predict_tree(lgb_uk_final, test_fe[feat_uk])
print(f"  UK LightGBM: retrained ({lgb_uk.best_iteration} iter)")

# UK XGBoost retrain
xgb_uk_final = retrain_tree("xgboost", XGB_UK_P,
    train_fe.loc[train_fe.index[valid_uk_full], feat_uk],
    y_basis_full[valid_uk_full], xgb_uk.best_iteration)
preds_uk_test_xgb = uk_moc_test + predict_tree(xgb_uk_final, test_fe[feat_uk])
print(f"  UK XGBoost: retrained ({xgb_uk.best_iteration} iter)")

# UK Elastic Net retrain
en_uk_final, en_uk_scaler = retrain_elastic_net(
    train_fe.loc[train_fe.index[valid_uk_full], feat_uk].values,
    y_basis_full[valid_uk_full], alpha=1.0, l1_ratio=0.9)
preds_uk_test_en = uk_moc_test + predict_elastic_net(en_uk_final, en_uk_scaler, test_fe[feat_uk].values)
print(f"  UK Elastic Net: retrained")

# UK DNN retrain
dnn_scaler_full_uk = StandardScaler()
X_dnn_full_uk = dnn_scaler_full_uk.fit_transform(
    np.nan_to_num(train_fe.loc[train_fe.index[valid_uk_full], feat_dnn_final].values, 0))
X_dnn_test_uk = dnn_scaler_full_uk.transform(np.nan_to_num(test_fe[feat_dnn_final].values, 0))
torch.manual_seed(42); np.random.seed(42)
dnn_uk_final = ElecDNN(len(feat_dnn_final), [768, 384, 192], dropout=0.3)
retrain_epochs_uk = dnn_uk_epochs + 5
dnn_uk_final, _ = train_dnn(dnn_uk_final, X_dnn_full_uk,
    y_basis_full[valid_uk_full].astype(np.float32),
    X_dnn_full_uk[:256], y_basis_full[valid_uk_full][:256].astype(np.float32),
    max_epochs=retrain_epochs_uk, patience=retrain_epochs_uk,
    criterion=torch.nn.MSELoss())
preds_uk_test_dnn = uk_moc_test + predict_dnn(dnn_uk_final, X_dnn_test_uk)
print(f"  UK DNN: retrained ({retrain_epochs_uk} epochs)")

# UK Ridge fondamentales retrain
fund_uk_full = build_fundamental_features(train_fe, "uk")
fund_uk_test = build_fundamental_features(test_fe, "uk")
scaler_ruk_full = StandardScaler()
X_ruk_full = scaler_ruk_full.fit_transform(np.nan_to_num(fund_uk_full.values, 0))
X_ruk_test = scaler_ruk_full.transform(np.nan_to_num(fund_uk_test.values, 0))
ridge_uk_full = Ridge(alpha=10.0)
ridge_uk_full.fit(X_ruk_full, uk_spot_full)
preds_uk_test_ridge = ridge_uk_full.predict(X_ruk_test)
print(f"  UK Ridge: retrained")

# UK T2 retrain
print(f"  UK T2: retraining {len(t2_uk)} models...")
t2_uk_test_preds = {}
for name, t2_info in t2_uk.items():
    feats = t2_info["feats"]
    algo = t2_info["algo"]
    X_full = train_fe[feats].values[valid_uk_full]
    y_full = y_basis_full[valid_uk_full]
    retrained = retrain_t2(algo, X_full, y_full, t2_info["info"])
    preds_basis = predict_t2(retrained, test_fe[feats].values)
    t2_uk_test_preds[name] = uk_moc_test + preds_basis

# UK v9 test ensemble (for SR computation)
uk_v9_test = {"CB": preds_uk_test_cb, "LGB": preds_uk_test_lgb, "XGB": preds_uk_test_xgb,
              "EN": preds_uk_test_en, "DNN": preds_uk_test_dnn}
v9_ens_uk_test = apply_regime_weights(uk_v9_test, hours_test, v9_regime_uk)

# UK SR test prediction (enriched)
X_sr_uk_test_t2 = np.column_stack([t2_uk_test_preds[k] for k in uk_t2_names])
h_test = hours_test; d_test = dow_test
X_extra_test = np.column_stack([
    np.sin(2 * np.pi * h_test / 24), np.cos(2 * np.pi * h_test / 24),
    np.sin(2 * np.pi * d_test / 7), np.cos(2 * np.pi * d_test / 7),
])
X_sr_uk_test = np.hstack([X_sr_uk_test_t2, X_extra_test])
sr_correction_uk = meta_uk.predict(X_sr_uk_test)
preds_uk_test_sr = v9_ens_uk_test + sr_correction_uk

# UK final ensemble
uk_test_models = {"CB": preds_uk_test_cb, "LGB": preds_uk_test_lgb, "XGB": preds_uk_test_xgb,
                  "EN": preds_uk_test_en, "DNN": preds_uk_test_dnn,
                  "RidgeF": preds_uk_test_ridge, "SR": preds_uk_test_sr}
preds_uk_test = apply_regime_weights(uk_test_models, hours_test, uk_regime_weights)
preds_uk_test_hbc = preds_uk_test + np.array([hbc_uk_final.get(h, 0) for h in hours_test])
print(f"  UK test: mean={preds_uk_test_hbc.mean():.1f}, std={preds_uk_test_hbc.std():.1f}")


# ══════════════════════════════════════════════════════════════════════════
# 7. GENERATE SUBMISSION
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  7. Generate submission CSV")
print("=" * 90)

fr_q_low = np.percentile(train_fe["fr_spot"].dropna(), 0.1)
fr_q_high = np.percentile(train_fe["fr_spot"].dropna(), 99.9)
uk_q_low = np.percentile(train_fe["uk_spot"].dropna(), 0.1)
uk_q_high = np.percentile(train_fe["uk_spot"].dropna(), 99.9)

print(f"  FR clipping: [{fr_q_low:.1f}, {fr_q_high:.1f}]")
print(f"  UK clipping: [{uk_q_low:.1f}, {uk_q_high:.1f}]")

sub = pd.DataFrame({
    "id": test_fe.index,
    "fr_spot": np.clip(preds_fr_test_hbc, fr_q_low, fr_q_high),
    "uk_spot": np.clip(preds_uk_test_hbc, uk_q_low, uk_q_high),
})
sub.to_csv("outputs/submission_v11.csv", index=False)
sub.to_csv("outputs/submission.csv", index=False)
print(f"  submission_v11.csv — {len(sub)} rows")


# ══════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  FINAL SUMMARY — v11")
print("=" * 90)

print(f"\n  Validation (HBC):")
print(f"    FR: {rmse_fr_hbc:.4f}")
print(f"    UK: {rmse_uk_hbc:.4f}")
print(f"    SUM: {val_sum:.4f}")
print(f"    Gap vs leader: {val_sum - 23.14:.2f}")

print(f"\n  Architecture:")
print(f"    v9 models: CB + LGB + XGB + EN + DNN (per market)")
print(f"    + Ridge fondamentales")
print(f"    + Stacking Résiduel T2:")
print(f"      FR: {len(t2_fr)} models ({len(FR_T2_ALGOS)} algos × {len(fr_groups)} groups), alpha=1")
print(f"      UK: {len(t2_uk)} models ({len(UK_T2_ALGOS)} algos × {len(uk_groups)} groups + combos), enriched, alpha=100")
print(f"    Ensemble: regime-weighted (7 models) + HBC")

for rname in REGIMES:
    rw = fr_regime_weights.get(rname, {})
    rw_str = " / ".join(f"{nm}={rw.get(nm, 0):.1f}" for nm in ["CB","LGB","XGB","EN","DNN","RidgeF","SR"])
    print(f"    FR {rname:8s}: {rw_str}")

for rname in REGIMES:
    rw = uk_regime_weights.get(rname, {})
    rw_str = " / ".join(f"{nm}={rw.get(nm, 0):.1f}" for nm in ["CB","LGB","XGB","EN","DNN","RidgeF","SR"])
    print(f"    UK {rname:8s}: {rw_str}")

print(f"\n  Test predictions:")
print(f"    FR: mean={preds_fr_test_hbc.mean():.1f}, std={preds_fr_test_hbc.std():.1f}")
print(f"    UK: mean={preds_uk_test_hbc.mean():.1f}, std={preds_uk_test_hbc.std():.1f}")

print(f"\n  Submission: outputs/submission_v11.csv")
print(f"  Total time: {time.time() - t0:.0f}s")
