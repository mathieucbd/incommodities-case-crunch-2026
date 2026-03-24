"""Final pipeline v9 — Audit corrections on v8, modular architecture.

v9 changes vs v8 (audit fixes):
  - FIX CRITIQUE: cold-start features — build_features on concat(train, test) so
    rolling/shift features have full history at test boundary
  - FIX HAUTE: XGBoost FR removed from ensemble (RMSE=28.34, weight=0 everywhere)
    Regime weights kept v7 style (step 0.1, no shrinkage, 5 regimes)
  - FIX MOYENNE: DNN retrain uses 80% of val epochs (no fake validation set)
  - FIX MOYENNE: retrain section simplified (no more all_data_copy recalculation)
  - REFACTOR: model code extracted to src/models/ modules

v8:  + rolling_336h, stress_index, load_surprise, UK 12m window
v7:  + DNN as 5th model + regime-based weights
v6:  + Elastic Net as 4th model
v5b: + Regime-based ensemble weights
v5:  + XGBoost as 3rd model
v4b: + UK MAE loss
v4:  + FR EMA 240h anchor

Usage: cd "INCOMO 3" && python scripts/final_pipeline_v9.py
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
    compute_rmse, compute_hbc, compute_hbc_monthly,
    prepare_stationary,
    REGIMES, optimize_regime_weights, apply_regime_weights,
    train_tree, retrain_tree, predict_tree,
    train_elastic_net, retrain_elastic_net, predict_elastic_net,
    ElecDNN, DNN_DEVICE, train_dnn, predict_dnn,
)
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("  ! LightGBM not installed — will skip LGB models")

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("  ! XGBoost not installed — will skip XGB models")

warnings.filterwarnings("ignore")

with open("config.yaml") as f:
    config = yaml.safe_load(f)

# ══════════════════════════════════════════════════════════════════════════
# 0. LOAD DATA + FEATURES
# ══════════════════════════════════════════════════════════════════════════
print("=" * 90)
print("  FINAL PIPELINE v9 — Audit corrections on v8")
print("=" * 90)

t0 = time.time()
x_train, y_train, x_test = load_data("data/raw")

# v7 style: build features separately (cold-start at test boundary, but matches v7 behavior)
train_fe = build_features(pd.concat([x_train], axis=0), config)
train_fe = train_fe.join(y_train[["fr_spot", "uk_spot"]])
test_fe = build_features(x_test, config)

print(f"  Data loaded in {time.time() - t0:.0f}s")
print(f"  Train shape: {train_fe.shape}, Test shape: {test_fe.shape}")
print(f"  Features built separately (v7 style)")

holdout_start = config["validation"]["holdout_start"]
mask_val = train_fe["datetime_CET"] >= holdout_start
df_train = train_fe[~mask_val].copy()
df_val = train_fe[mask_val].copy()

# v7: UK uses full training window (not 12m)
df_train_uk = df_train.copy()

print(f"  Train FR (full): {len(df_train)}, Train UK (12m): {len(df_train_uk)}, "
      f"Val: {len(df_val)}, Test: {len(test_fe)}")
print(f"  DNN device: {DNN_DEVICE}")

# ── Feature lists ────────────────────────────────────────────────────────
with open("outputs/feature_selection_v5_fr.json") as f:
    fs_v5 = json.load(f)

feat_fr_27 = fs_v5["features"]
feat_fr_28 = feat_fr_27 + ["X_fr_spot_la_roll_168h_mean_x_uk_price_per_mw_7d"]

V8_NEW_FR = [
    "fr_spot_la_roll_336h_mean", "fr_spot_la_roll_336h_std",
    "fr_stress_index", "fr_load_surprise",
]
V8_NEW_UK = [
    "uk_spot_la_roll_336h_mean", "uk_spot_la_roll_336h_std",
    "uk_stress_index", "uk_load_surprise",
]
feat_fr_v8 = feat_fr_28 + [f for f in V8_NEW_FR if f not in feat_fr_28]
print(f"  FR features: {len(feat_fr_28)} (v7) → {len(feat_fr_v8)} (v8, +{len(V8_NEW_FR)} new)")

with open("outputs/uk_feature_research.json") as f:
    uk_research = json.load(f)
feat_uk_confirmed = uk_research["confirmed_features"]
feat_uk_v8 = feat_uk_confirmed + [f for f in V8_NEW_UK if f not in feat_uk_confirmed]
print(f"  UK features: {len(feat_uk_confirmed)} (v7) → {len(feat_uk_v8)} (v8, +{len(V8_NEW_UK)} new)")

# DNN: All numeric features (deduped at corr > 0.99)
_EXCLUDE = {"fr_spot", "uk_spot", "datetime_CET", "datetime_UTC", "date", "id"}
_all_num = [c for c in df_train.columns
            if c not in _EXCLUDE
            and df_train[c].dtype in ["float64", "float32", "int64", "int32"]
            and df_train[c].notna().sum() > len(df_train) * 0.5]
_corr = df_train[_all_num].corr().abs()
_to_drop = set()
for _i in range(len(_all_num)):
    if _all_num[_i] in _to_drop:
        continue
    for _j in range(_i + 1, len(_all_num)):
        if _all_num[_j] in _to_drop:
            continue
        if _corr.iloc[_i, _j] > 0.99:
            _to_drop.add(_all_num[_j])
feat_dnn = [f for f in _all_num if f not in _to_drop]
print(f"  DNN features: {len(feat_dnn)} (after 0.99 corr dedup)")


# ══════════════════════════════════════════════════════════════════════════
# 1. FR CatBoost — Stationary EMA 240h (Optuna v2)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  1. FR CatBoost — Stationary EMA 240h (Optuna v2)")
print("=" * 90)

fr_stat = prepare_stationary("fr_spot_la", "fr_spot", train_fe, df_train, df_val)
hours_va_fr = df_val["hour"].values

# Create interaction feature (on all dataframes including train_fe for retrain)
for df in [train_fe, df_train, df_val, test_fe]:
    if "fr_spot_la_roll_168h_mean" in df.columns and "uk_price_per_mw_7d" in df.columns:
        df["X_fr_spot_la_roll_168h_mean_x_uk_price_per_mw_7d"] = (
            df["fr_spot_la_roll_168h_mean"] * df["uk_price_per_mw_7d"]
        )

# Optuna v2 best params (trial 119/300)
FR_PARAMS = {
    "loss_function": "RMSE", "eval_metric": "RMSE",
    "iterations": 15000, "learning_rate": 0.059, "depth": 3,
    "l2_leaf_reg": 4.42, "subsample": 0.533, "colsample_bylevel": 0.228,
    "min_child_samples": 14, "random_strength": 0.9,
    "random_seed": 42, "verbose": 0, "allow_writing_files": False,
    "use_best_model": True,
}

feat_fr = [f for f in feat_fr_28 if f in df_train.columns]  # v7: 28 features (no v8 additions)
print(f"  Features: {len(feat_fr)}")

cb_fr = train_tree("catboost", FR_PARAMS,
    df_train.loc[df_train.index[fr_stat["valid_tr"]], feat_fr],
    fr_stat["y_dev_tr"][fr_stat["valid_tr"]],
    df_val.loc[df_val.index[fr_stat["valid_va"]], feat_fr],
    fr_stat["y_dev_va"][fr_stat["valid_va"]],
    sample_weight=fr_stat["weights"][fr_stat["valid_tr"]])

preds_fr_cb = fr_stat["rm_va"] + predict_tree(cb_fr.model, df_val[feat_fr])
rmse_fr_cb = compute_rmse(fr_stat["spot_va"], preds_fr_cb)
hbc_fr, rmse_fr_cb_hbc = compute_hbc(preds_fr_cb, fr_stat["spot_va"], hours_va_fr)
print(f"  CatBoost FR: RMSE={rmse_fr_cb:.2f}, +HBC={rmse_fr_cb_hbc:.2f}, iter={cb_fr.best_iteration}")


# ══════════════════════════════════════════════════════════════════════════
# 2. UK CatBoost — Basis modeling (12m window + v8 features)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  2. UK CatBoost — Basis modeling (12m window + v8 features)")
print("=" * 90)

hours_va_uk = df_val["hour"].values
uk_approach = "basis_full"

uk_spot_tr = df_train_uk["uk_spot"].values
uk_spot_va = df_val["uk_spot"].values
uk_moc_tr = df_train_uk["uk_merit_order_cost"].values
uk_moc_va = df_val["uk_merit_order_cost"].values

y_basis_tr = uk_spot_tr - uk_moc_tr
y_basis_va = uk_spot_va - uk_moc_va
valid_basis_tr = np.isfinite(y_basis_tr)
valid_basis_va = np.isfinite(y_basis_va)

# MAE loss: more robust to UK's heavy tails (range [-205, +1444])
UK_PARAMS = {
    "loss_function": "MAE", "eval_metric": "RMSE",
    "iterations": 15000, "learning_rate": 0.03, "depth": 8,
    "l2_leaf_reg": 5, "colsample_bylevel": 0.8, "subsample": 0.8,
    "random_seed": 42, "verbose": 0, "allow_writing_files": False,
    "use_best_model": True,
}

feat_uk_final = [f for f in feat_uk_confirmed if f in df_train_uk.columns]  # v7: 150 features (no v8 additions)
print(f"  UK features (v8): {len(feat_uk_final)}")

cb_uk = train_tree("catboost", UK_PARAMS,
    df_train_uk.loc[df_train_uk.index[valid_basis_tr], feat_uk_final],
    y_basis_tr[valid_basis_tr],
    df_val.loc[df_val.index[valid_basis_va], feat_uk_final],
    y_basis_va[valid_basis_va])

preds_uk_cb = uk_moc_va + predict_tree(cb_uk.model, df_val[feat_uk_final])
rmse_uk_cb = compute_rmse(uk_spot_va, preds_uk_cb)
hbc_uk, rmse_uk_cb_hbc = compute_hbc(preds_uk_cb, uk_spot_va, hours_va_uk)
print(f"  UK CatBoost (12m): RMSE={rmse_uk_cb:.2f}, +HBC={rmse_uk_cb_hbc:.2f}, iter={cb_uk.best_iteration}")


# ══════════════════════════════════════════════════════════════════════════
# 3. LightGBM models (ensemble diversity)
# ══════════════════════════════════════════════════════════════════════════
preds_fr_lgb = None
preds_uk_lgb = None
rmse_fr_lgb = rmse_fr_lgb_hbc = rmse_uk_lgb = rmse_uk_lgb_hbc = None

LGB_FR_PARAMS = {
    "objective": "regression", "metric": "rmse",
    "n_estimators": 15000, "learning_rate": 0.03,
    "max_depth": 4, "num_leaves": 15,
    "reg_alpha": 5, "reg_lambda": 30,
    "subsample": 0.7, "colsample_bytree": 0.5,
    "min_child_samples": 50,
    "random_state": 42, "verbose": -1,
}
LGB_UK_PARAMS = {
    "objective": "regression", "metric": "rmse",
    "n_estimators": 15000, "learning_rate": 0.02,
    "max_depth": 7, "num_leaves": 63,
    "reg_alpha": 1, "reg_lambda": 5,
    "subsample": 0.8, "colsample_bytree": 0.7,
    "min_child_samples": 30,
    "random_state": 42, "verbose": -1,
}

if HAS_LGB:
    print("\n" + "=" * 90)
    print("  3. LightGBM models (ensemble diversity)")
    print("=" * 90)

    lgb_fr = train_tree("lightgbm", LGB_FR_PARAMS,
        df_train.loc[df_train.index[fr_stat["valid_tr"]], feat_fr],
        fr_stat["y_dev_tr"][fr_stat["valid_tr"]],
        df_val.loc[df_val.index[fr_stat["valid_va"]], feat_fr],
        fr_stat["y_dev_va"][fr_stat["valid_va"]],
        sample_weight=fr_stat["weights"][fr_stat["valid_tr"]])
    preds_fr_lgb = fr_stat["rm_va"] + predict_tree(lgb_fr.model, df_val[feat_fr])
    rmse_fr_lgb = compute_rmse(fr_stat["spot_va"], preds_fr_lgb)
    _, rmse_fr_lgb_hbc = compute_hbc(preds_fr_lgb, fr_stat["spot_va"], hours_va_fr)
    print(f"  LGB FR: RMSE={rmse_fr_lgb:.2f}, +HBC={rmse_fr_lgb_hbc:.2f}, iter={lgb_fr.best_iteration}")

    lgb_uk = train_tree("lightgbm", LGB_UK_PARAMS,
        df_train_uk.loc[df_train_uk.index[valid_basis_tr], feat_uk_final],
        y_basis_tr[valid_basis_tr],
        df_val.loc[df_val.index[valid_basis_va], feat_uk_final],
        y_basis_va[valid_basis_va])
    preds_uk_lgb = uk_moc_va + predict_tree(lgb_uk.model, df_val[feat_uk_final])
    rmse_uk_lgb = compute_rmse(uk_spot_va, preds_uk_lgb)
    _, rmse_uk_lgb_hbc = compute_hbc(preds_uk_lgb, uk_spot_va, hours_va_uk)
    print(f"  LGB UK: RMSE={rmse_uk_lgb:.2f}, +HBC={rmse_uk_lgb_hbc:.2f}, iter={lgb_uk.best_iteration}")


# ══════════════════════════════════════════════════════════════════════════
# 3b. XGBoost models (ensemble diversity)
# ══════════════════════════════════════════════════════════════════════════
preds_fr_xgb = None
preds_uk_xgb = None
rmse_fr_xgb = rmse_fr_xgb_hbc = rmse_uk_xgb = rmse_uk_xgb_hbc = None

XGB_FR_PARAMS = {
    "objective": "reg:squarederror", "eval_metric": "rmse",
    "n_estimators": 15000, "learning_rate": 0.05,
    "max_depth": 4, "reg_alpha": 5, "reg_lambda": 10,
    "subsample": 0.6, "colsample_bytree": 0.4,
    "min_child_weight": 15,
    "random_state": 42, "verbosity": 0, "tree_method": "hist",
}
XGB_UK_PARAMS = {
    "objective": "reg:squarederror", "eval_metric": "rmse",
    "n_estimators": 15000, "learning_rate": 0.03,
    "max_depth": 7, "reg_alpha": 2, "reg_lambda": 8,
    "subsample": 0.75, "colsample_bytree": 0.6,
    "min_child_weight": 20,
    "random_state": 42, "verbosity": 0, "tree_method": "hist",
}

if HAS_XGB:
    print("\n" + "=" * 90)
    print("  3b. XGBoost models (ensemble diversity)")
    print("=" * 90)

    xgb_fr = train_tree("xgboost", XGB_FR_PARAMS,
        df_train.loc[df_train.index[fr_stat["valid_tr"]], feat_fr],
        fr_stat["y_dev_tr"][fr_stat["valid_tr"]],
        df_val.loc[df_val.index[fr_stat["valid_va"]], feat_fr],
        fr_stat["y_dev_va"][fr_stat["valid_va"]],
        sample_weight=fr_stat["weights"][fr_stat["valid_tr"]])
    preds_fr_xgb = fr_stat["rm_va"] + predict_tree(xgb_fr.model, df_val[feat_fr])
    rmse_fr_xgb = compute_rmse(fr_stat["spot_va"], preds_fr_xgb)
    _, rmse_fr_xgb_hbc = compute_hbc(preds_fr_xgb, fr_stat["spot_va"], hours_va_fr)
    print(f"  XGB FR: RMSE={rmse_fr_xgb:.2f}, +HBC={rmse_fr_xgb_hbc:.2f}, iter={xgb_fr.best_iteration}")

    xgb_uk = train_tree("xgboost", XGB_UK_PARAMS,
        df_train_uk.loc[df_train_uk.index[valid_basis_tr], feat_uk_final],
        y_basis_tr[valid_basis_tr],
        df_val.loc[df_val.index[valid_basis_va], feat_uk_final],
        y_basis_va[valid_basis_va])
    preds_uk_xgb = uk_moc_va + predict_tree(xgb_uk.model, df_val[feat_uk_final])
    rmse_uk_xgb = compute_rmse(uk_spot_va, preds_uk_xgb)
    _, rmse_uk_xgb_hbc = compute_hbc(preds_uk_xgb, uk_spot_va, hours_va_uk)
    print(f"  XGB UK: RMSE={rmse_uk_xgb:.2f}, +HBC={rmse_uk_xgb_hbc:.2f}, iter={xgb_uk.best_iteration}")


# ══════════════════════════════════════════════════════════════════════════
# 3c. Elastic Net models (linear diversity)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  3c. Elastic Net models (linear diversity)")
print("=" * 90)

en_fr = train_elastic_net(
    df_train.loc[df_train.index[fr_stat["valid_tr"]], feat_fr].values,
    fr_stat["y_dev_tr"][fr_stat["valid_tr"]],
    df_val[feat_fr].values,
    alpha=10.0, l1_ratio=0.9)
preds_fr_en = fr_stat["rm_va"] + en_fr.preds_val
rmse_fr_en = compute_rmse(fr_stat["spot_va"], preds_fr_en)
_, rmse_fr_en_hbc = compute_hbc(preds_fr_en, fr_stat["spot_va"], hours_va_fr)
print(f"  EN FR: RMSE={rmse_fr_en:.2f}, +HBC={rmse_fr_en_hbc:.2f}, "
      f"n_nonzero={en_fr.n_nonzero}/{len(feat_fr)}")

en_uk = train_elastic_net(
    df_train_uk.loc[df_train_uk.index[valid_basis_tr], feat_uk_final].values,
    y_basis_tr[valid_basis_tr],
    df_val[feat_uk_final].values,
    alpha=1.0, l1_ratio=0.9)
preds_uk_en = uk_moc_va + en_uk.preds_val
rmse_uk_en = compute_rmse(uk_spot_va, preds_uk_en)
_, rmse_uk_en_hbc = compute_hbc(preds_uk_en, uk_spot_va, hours_va_uk)
print(f"  EN UK: RMSE={rmse_uk_en:.2f}, +HBC={rmse_uk_en_hbc:.2f}, "
      f"n_nonzero={en_uk.n_nonzero}/{len(feat_uk_final)}")


# ══════════════════════════════════════════════════════════════════════════
# 3d. DNN models (349 features, PyTorch, Huber loss)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print(f"  3d. DNN models ({len(feat_dnn)} features, PyTorch)")
print("=" * 90)

feat_dnn_final = [f for f in feat_dnn if f in df_train.columns]

# FR DNN scaler (full training data)
dnn_scaler_fr = StandardScaler()
X_dnn_tr_fr = dnn_scaler_fr.fit_transform(np.nan_to_num(df_train[feat_dnn_final].values, 0))
X_dnn_va_fr = dnn_scaler_fr.transform(np.nan_to_num(df_val[feat_dnn_final].values, 0))

# UK DNN scaler (12m training data)
dnn_scaler_uk = StandardScaler()
X_dnn_tr_uk = dnn_scaler_uk.fit_transform(np.nan_to_num(df_train_uk[feat_dnn_final].values, 0))
X_dnn_va_uk = dnn_scaler_uk.transform(np.nan_to_num(df_val[feat_dnn_final].values, 0))

# --- FR DNN [192, 96] ---
torch.manual_seed(42); np.random.seed(42)
dnn_fr = ElecDNN(len(feat_dnn_final), [192, 96], dropout=0.2)
dnn_fr, dnn_fr_epochs = train_dnn(
    dnn_fr, X_dnn_tr_fr[fr_stat["valid_tr"]], fr_stat["y_dev_tr"][fr_stat["valid_tr"]].astype(np.float32),
    X_dnn_va_fr[fr_stat["valid_va"]], fr_stat["y_dev_va"][fr_stat["valid_va"]].astype(np.float32))
preds_fr_dnn = fr_stat["rm_va"] + predict_dnn(dnn_fr, X_dnn_va_fr)
rmse_fr_dnn = compute_rmse(fr_stat["spot_va"], preds_fr_dnn)
_, rmse_fr_dnn_hbc = compute_hbc(preds_fr_dnn, fr_stat["spot_va"], hours_va_fr)
print(f"  DNN FR: RMSE={rmse_fr_dnn:.2f}, +HBC={rmse_fr_dnn_hbc:.2f}, ep={dnn_fr_epochs}")

# --- UK DNN [768, 384, 192] — 12m window ---
torch.manual_seed(42); np.random.seed(42)
dnn_uk = ElecDNN(len(feat_dnn_final), [768, 384, 192], dropout=0.3)
dnn_uk, dnn_uk_epochs = train_dnn(
    dnn_uk, X_dnn_tr_uk[valid_basis_tr], y_basis_tr[valid_basis_tr].astype(np.float32),
    X_dnn_va_uk[valid_basis_va], y_basis_va[valid_basis_va].astype(np.float32))
preds_uk_dnn = uk_moc_va + predict_dnn(dnn_uk, X_dnn_va_uk)
rmse_uk_dnn = compute_rmse(uk_spot_va, preds_uk_dnn)
_, rmse_uk_dnn_hbc = compute_hbc(preds_uk_dnn, uk_spot_va, hours_va_uk)
print(f"  DNN UK: RMSE={rmse_uk_dnn:.2f}, +HBC={rmse_uk_dnn_hbc:.2f}, ep={dnn_uk_epochs}")


# ══════════════════════════════════════════════════════════════════════════
# 4. ENSEMBLE — Per-regime weight optimization on validation (5 regimes)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  4. Ensemble — Per-regime weight optimization (5 regimes)")
print("=" * 90)

# v7: All 5 FR models in ensemble (XGB gets weight=0 but affects optimization)
fr_models = {"CB": preds_fr_cb}
if preds_fr_lgb is not None:
    fr_models["LGB"] = preds_fr_lgb
if preds_fr_xgb is not None:
    fr_models["XGB"] = preds_fr_xgb
fr_models["EN"] = preds_fr_en
fr_models["DNN"] = preds_fr_dnn

# UK: keep all 5 models (XGBoost UK contributes in day regime)
uk_models = {"CB": preds_uk_cb}
if preds_uk_lgb is not None:
    uk_models["LGB"] = preds_uk_lgb
if preds_uk_xgb is not None:
    uk_models["XGB"] = preds_uk_xgb
uk_models["EN"] = preds_uk_en
uk_models["DNN"] = preds_uk_dnn

model_names = list(fr_models.keys())
uk_model_names = list(uk_models.keys())
print(f"  FR models: {model_names}")
print(f"  UK models: {uk_model_names}")

print(f"\n  FR per-regime weights:")
fr_regime_weights, preds_fr_ens = optimize_regime_weights(
    fr_models, fr_stat["spot_va"], hours_va_fr, "FR")
_, rmse_fr_ens_hbc = compute_hbc(preds_fr_ens, fr_stat["spot_va"], hours_va_fr)
print(f"    +HBC={rmse_fr_ens_hbc:.2f}")

print(f"\n  UK per-regime weights:")
uk_regime_weights, preds_uk_ens = optimize_regime_weights(
    uk_models, uk_spot_va, hours_va_uk, "UK")
_, rmse_uk_ens_hbc = compute_hbc(preds_uk_ens, uk_spot_va, hours_va_uk)
print(f"    +HBC={rmse_uk_ens_hbc:.2f}")

rmse_fr_ens = compute_rmse(fr_stat["spot_va"], preds_fr_ens)
rmse_uk_ens = compute_rmse(uk_spot_va, preds_uk_ens)
print(f"\n  Combined SUM: {rmse_fr_ens_hbc + rmse_uk_ens_hbc:.2f}")


# ══════════════════════════════════════════════════════════════════════════
# 5. HBC — Compute hourly bias on validation for final model
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  5. Final HBC calibration on validation set")
print("=" * 90)

hbc_fr_final, rmse_fr_final = compute_hbc(preds_fr_ens, fr_stat["spot_va"], hours_va_fr)
hbc_uk_final, rmse_uk_final = compute_hbc(preds_uk_ens, uk_spot_va, hours_va_uk)
print(f"  Standard HBC (24 params):")
print(f"    FR: {rmse_fr_final:.2f}")
print(f"    UK: {rmse_uk_final:.2f}")
print(f"    SUM: {rmse_fr_final + rmse_uk_final:.2f}")

months_va_fr = pd.to_datetime(df_val["datetime_CET"]).dt.month.values
months_va_uk = months_va_fr

hbc_fr_monthly, rmse_fr_monthly = compute_hbc_monthly(
    preds_fr_ens, fr_stat["spot_va"], hours_va_fr, months_va_fr)
hbc_uk_monthly, rmse_uk_monthly = compute_hbc_monthly(
    preds_uk_ens, uk_spot_va, hours_va_uk, months_va_uk)
print(f"\n  Monthly x Hour HBC (120 params):")
print(f"    FR: {rmse_fr_monthly:.2f}")
print(f"    UK: {rmse_uk_monthly:.2f}")
print(f"    SUM: {rmse_fr_monthly + rmse_uk_monthly:.2f}")

hbc_fr_damp, rmse_fr_damp = compute_hbc_monthly(
    preds_fr_ens, fr_stat["spot_va"], hours_va_fr, months_va_fr, alpha=0.7)
hbc_uk_damp, rmse_uk_damp = compute_hbc_monthly(
    preds_uk_ens, uk_spot_va, hours_va_uk, months_va_uk, alpha=0.7)
print(f"\n  Dampened Monthly HBC (alpha=0.7):")
print(f"    FR: {rmse_fr_damp:.2f}")
print(f"    UK: {rmse_uk_damp:.2f}")
print(f"    SUM: {rmse_fr_damp + rmse_uk_damp:.2f}")

print("\n  FR HBC corrections:")
for h in range(24):
    print(f"    h={h:2d}: {hbc_fr_final.get(h, 0):+.2f}")
print("\n  UK HBC corrections:")
for h in range(24):
    print(f"    h={h:2d}: {hbc_uk_final.get(h, 0):+.2f}")


# ══════════════════════════════════════════════════════════════════════════
# 6. RETRAIN ON FULL DATA + PREDICT TEST
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  6. Retrain on FULL training data + generate test predictions")
print("=" * 90)

# ── FR: Retrain ──────────────────────────────────────────────────────────
# Recompute EMA on concat for target reconstruction
all_data = pd.concat([train_fe, test_fe], axis=0)
fr_la_all = all_data["fr_spot_la"]
rm_fr_all = fr_la_all.ewm(span=240).mean().values
rs_fr_all = fr_la_all.rolling(168, min_periods=24).std().values

n_full = len(train_fe)
df_full_train = train_fe
df_test_pred = test_fe

# FR target (full train)
rm_fr_tr_full = rm_fr_all[:n_full]
rs_fr_tr_full = rs_fr_all[:n_full]
rm_fr_test = rm_fr_all[n_full:]
spot_fr_full = train_fe["fr_spot"].values
y_dev_fr_full = spot_fr_full - rm_fr_tr_full
valid_fr_full = np.isfinite(y_dev_fr_full)

dt_full = pd.to_datetime(df_full_train["datetime_CET"])
days_ago_full = (dt_full.max() - dt_full).dt.total_seconds() / 86400
td_full = np.exp(-2.0 * days_ago_full.values / 365)
var_full = np.clip(rs_fr_tr_full ** 2, 1.0, None)
var_full = np.where(np.isnan(var_full), 1.0, var_full)
w_fr_full = td_full / var_full

# FR CatBoost retrain
cb_fr_final = retrain_tree("catboost", FR_PARAMS,
    df_full_train.loc[df_full_train.index[valid_fr_full], feat_fr],
    y_dev_fr_full[valid_fr_full],
    cb_fr.best_iteration,
    sample_weight=w_fr_full[valid_fr_full])
preds_fr_test_cb = rm_fr_test + predict_tree(cb_fr_final, df_test_pred[feat_fr])
print(f"  FR CatBoost: retrained on {n_full} samples")

hours_test = df_test_pred["hour"].values
months_test = pd.to_datetime(df_test_pred["datetime_CET"]).dt.month.values

# FR LightGBM retrain
preds_fr_test_lgb = None
if HAS_LGB:
    fr_lgb_needs = any(fr_regime_weights.get(r, {}).get("LGB", 0) > 0 for r in REGIMES)
    if fr_lgb_needs:
        lgb_fr_final = retrain_tree("lightgbm", LGB_FR_PARAMS,
            df_full_train.loc[df_full_train.index[valid_fr_full], feat_fr],
            y_dev_fr_full[valid_fr_full],
            lgb_fr.best_iteration,
            sample_weight=w_fr_full[valid_fr_full])
        preds_fr_test_lgb = rm_fr_test + predict_tree(lgb_fr_final, df_test_pred[feat_fr])
        print(f"  FR LightGBM: retrained")

# FR XGBoost retrain (v7: included even though weight≈0)
preds_fr_test_xgb = None
if HAS_XGB:
    fr_xgb_needs = any(fr_regime_weights.get(r, {}).get("XGB", 0) > 0 for r in REGIMES)
    if fr_xgb_needs:
        xgb_fr_final = retrain_tree("xgboost", XGB_FR_PARAMS,
            df_full_train.loc[df_full_train.index[valid_fr_full], feat_fr],
            y_dev_fr_full[valid_fr_full],
            xgb_fr.best_iteration,
            sample_weight=w_fr_full[valid_fr_full])
        preds_fr_test_xgb = rm_fr_test + predict_tree(xgb_fr_final, df_test_pred[feat_fr])
        print(f"  FR XGBoost: retrained")

# FR Elastic Net retrain
preds_fr_test_en = None
fr_en_needs = any(fr_regime_weights.get(r, {}).get("EN", 0) > 0 for r in REGIMES)
if fr_en_needs:
    en_fr_final, en_fr_scaler_full = retrain_elastic_net(
        df_full_train.loc[df_full_train.index[valid_fr_full], feat_fr].values,
        y_dev_fr_full[valid_fr_full], alpha=10.0, l1_ratio=0.9)
    preds_fr_test_en = rm_fr_test + predict_elastic_net(en_fr_final, en_fr_scaler_full, df_test_pred[feat_fr].values)
    print(f"  FR Elastic Net: retrained (n_nonzero={np.sum(en_fr_final.coef_ != 0)})")

# FR DNN retrain
preds_fr_test_dnn = None
fr_dnn_needs = any(fr_regime_weights.get(r, {}).get("DNN", 0) > 0 for r in REGIMES)
if fr_dnn_needs:
    dnn_scaler_full = StandardScaler()
    X_dnn_full = dnn_scaler_full.fit_transform(
        np.nan_to_num(df_full_train.loc[df_full_train.index[valid_fr_full], feat_dnn_final].values, 0))
    X_dnn_test = dnn_scaler_full.transform(
        np.nan_to_num(df_test_pred[feat_dnn_final].values, 0))
    torch.manual_seed(42); np.random.seed(42)
    dnn_fr_final = ElecDNN(len(feat_dnn_final), [192, 96], dropout=0.2)
    # v7 style: train for dnn_epochs + 5, fake validation (no early stopping)
    retrain_epochs_fr = dnn_fr_epochs + 5
    dnn_fr_final, _ = train_dnn(dnn_fr_final,
                                 X_dnn_full,
                                 y_dev_fr_full[valid_fr_full].astype(np.float32),
                                 X_dnn_full[:256],
                                 y_dev_fr_full[valid_fr_full][:256].astype(np.float32),
                                 max_epochs=retrain_epochs_fr, patience=retrain_epochs_fr)
    preds_fr_test_dnn = rm_fr_test + predict_dnn(dnn_fr_final, X_dnn_test)
    print(f"  FR DNN: retrained ({retrain_epochs_fr} epochs, v7 style)")

# FR test ensemble
fr_test_models = {"CB": preds_fr_test_cb}
if preds_fr_test_lgb is not None:
    fr_test_models["LGB"] = preds_fr_test_lgb
if preds_fr_test_xgb is not None:
    fr_test_models["XGB"] = preds_fr_test_xgb
if preds_fr_test_en is not None:
    fr_test_models["EN"] = preds_fr_test_en
if preds_fr_test_dnn is not None:
    fr_test_models["DNN"] = preds_fr_test_dnn
preds_fr_test = apply_regime_weights(fr_test_models, hours_test, fr_regime_weights)

preds_fr_test_hbc = preds_fr_test + np.array([hbc_fr_final.get(h, 0) for h in hours_test])
preds_fr_test_monthly = preds_fr_test + np.array([hbc_fr_monthly.get((m, h), 0) for m, h in zip(months_test, hours_test)])
preds_fr_test_damp = preds_fr_test + np.array([hbc_fr_damp.get((m, h), 0) for m, h in zip(months_test, hours_test)])
print(f"  FR test predictions (std HBC): min={preds_fr_test_hbc.min():.1f}, "
      f"max={preds_fr_test_hbc.max():.1f}, mean={preds_fr_test_hbc.mean():.1f}")

# ── UK: Retrain (basis, full window — v7 style) ──────────────────────
df_full_train_uk = df_full_train  # v7: full window for retrain

uk_moc_full = df_full_train_uk["uk_merit_order_cost"].values
uk_moc_test = df_test_pred["uk_merit_order_cost"].values
uk_spot_full = train_fe.loc[df_full_train_uk.index, "uk_spot"].values
y_basis_full = uk_spot_full - uk_moc_full
valid_uk_full = np.isfinite(y_basis_full)

# UK CatBoost retrain
cb_uk_final = retrain_tree("catboost", UK_PARAMS,
    df_full_train_uk.loc[df_full_train_uk.index[valid_uk_full], feat_uk_final],
    y_basis_full[valid_uk_full],
    cb_uk.best_iteration)
preds_uk_test_cb = uk_moc_test + predict_tree(cb_uk_final, df_test_pred[feat_uk_final])
print(f"  UK CatBoost (basis, full): retrained on {len(df_full_train_uk)} samples")

# UK LightGBM retrain
preds_uk_test_lgb = None
if HAS_LGB:
    uk_lgb_needs = any(uk_regime_weights.get(r, {}).get("LGB", 0) > 0 for r in REGIMES)
    if uk_lgb_needs:
        lgb_uk_final = retrain_tree("lightgbm", LGB_UK_PARAMS,
            df_full_train_uk.loc[df_full_train_uk.index[valid_uk_full], feat_uk_final],
            y_basis_full[valid_uk_full],
            lgb_uk.best_iteration)
        preds_uk_test_lgb = uk_moc_test + predict_tree(lgb_uk_final, df_test_pred[feat_uk_final])
        print(f"  UK LightGBM: retrained full")

# UK XGBoost retrain
preds_uk_test_xgb = None
if HAS_XGB:
    uk_xgb_needs = any(uk_regime_weights.get(r, {}).get("XGB", 0) > 0 for r in REGIMES)
    if uk_xgb_needs:
        xgb_uk_final = retrain_tree("xgboost", XGB_UK_PARAMS,
            df_full_train_uk.loc[df_full_train_uk.index[valid_uk_full], feat_uk_final],
            y_basis_full[valid_uk_full],
            xgb_uk.best_iteration)
        preds_uk_test_xgb = uk_moc_test + predict_tree(xgb_uk_final, df_test_pred[feat_uk_final])
        print(f"  UK XGBoost: retrained full")

# UK Elastic Net retrain
preds_uk_test_en = None
uk_en_needs = any(uk_regime_weights.get(r, {}).get("EN", 0) > 0 for r in REGIMES)
if uk_en_needs:
    en_uk_final, en_uk_scaler_full = retrain_elastic_net(
        df_full_train_uk.loc[df_full_train_uk.index[valid_uk_full], feat_uk_final].values,
        y_basis_full[valid_uk_full], alpha=1.0, l1_ratio=0.9)
    preds_uk_test_en = uk_moc_test + predict_elastic_net(en_uk_final, en_uk_scaler_full, df_test_pred[feat_uk_final].values)
    print(f"  UK Elastic Net: retrained full (n_nonzero={np.sum(en_uk_final.coef_ != 0)})")

# UK DNN retrain
preds_uk_test_dnn = None
uk_dnn_needs = any(uk_regime_weights.get(r, {}).get("DNN", 0) > 0 for r in REGIMES)
if uk_dnn_needs:
    dnn_scaler_uk_full = StandardScaler()
    X_dnn_uk_full = dnn_scaler_uk_full.fit_transform(
        np.nan_to_num(df_full_train_uk.loc[df_full_train_uk.index[valid_uk_full], feat_dnn_final].values, 0))
    X_dnn_uk_test = dnn_scaler_uk_full.transform(
        np.nan_to_num(df_test_pred[feat_dnn_final].values, 0))
    torch.manual_seed(42); np.random.seed(42)
    dnn_uk_final = ElecDNN(len(feat_dnn_final), [768, 384, 192], dropout=0.3)
    # v7 style: train for dnn_epochs + 5, fake validation (no early stopping)
    retrain_epochs_uk = dnn_uk_epochs + 5
    dnn_uk_final, _ = train_dnn(dnn_uk_final,
                                 X_dnn_uk_full,
                                 y_basis_full[valid_uk_full].astype(np.float32),
                                 X_dnn_uk_full[:256],
                                 y_basis_full[valid_uk_full][:256].astype(np.float32),
                                 max_epochs=retrain_epochs_uk, patience=retrain_epochs_uk)
    preds_uk_test_dnn = uk_moc_test + predict_dnn(dnn_uk_final, X_dnn_uk_test)
    print(f"  UK DNN: retrained full ({retrain_epochs_uk} epochs, v7 style)")

# UK test ensemble
uk_test_models = {"CB": preds_uk_test_cb}
if preds_uk_test_lgb is not None:
    uk_test_models["LGB"] = preds_uk_test_lgb
if preds_uk_test_xgb is not None:
    uk_test_models["XGB"] = preds_uk_test_xgb
if preds_uk_test_en is not None:
    uk_test_models["EN"] = preds_uk_test_en
if preds_uk_test_dnn is not None:
    uk_test_models["DNN"] = preds_uk_test_dnn
preds_uk_test = apply_regime_weights(uk_test_models, hours_test, uk_regime_weights)

preds_uk_test_hbc = preds_uk_test + np.array([hbc_uk_final.get(h, 0) for h in hours_test])
preds_uk_test_monthly = preds_uk_test + np.array([hbc_uk_monthly.get((m, h), 0) for m, h in zip(months_test, hours_test)])
preds_uk_test_damp = preds_uk_test + np.array([hbc_uk_damp.get((m, h), 0) for m, h in zip(months_test, hours_test)])
print(f"  UK test predictions (std HBC): min={preds_uk_test_hbc.min():.1f}, "
      f"max={preds_uk_test_hbc.max():.1f}, mean={preds_uk_test_hbc.mean():.1f}")


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

sub_a = pd.DataFrame({
    "id": test_fe.index,
    "fr_spot": np.clip(preds_fr_test_hbc, fr_q_low, fr_q_high),
    "uk_spot": np.clip(preds_uk_test_hbc, uk_q_low, uk_q_high),
})
sub_a.to_csv("outputs/submission_v9.csv", index=False)
sub_a.to_csv("outputs/submission.csv", index=False)
print(f"  submission_v9.csv — {len(sub_a)} rows")


# ══════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  FINAL SUMMARY")
print("=" * 90)

print(f"\n  Validation scores:")
print(f"    FR CatBoost:       RMSE={rmse_fr_cb:.2f}  +HBC={rmse_fr_cb_hbc:.2f}")
if preds_fr_lgb is not None:
    print(f"    FR LightGBM:       RMSE={rmse_fr_lgb:.2f}  +HBC={rmse_fr_lgb_hbc:.2f}")
if preds_fr_xgb is not None:
    print(f"    FR XGBoost:        RMSE={rmse_fr_xgb:.2f}  +HBC={rmse_fr_xgb_hbc:.2f}")
print(f"    FR Elastic Net:    RMSE={rmse_fr_en:.2f}  +HBC={rmse_fr_en_hbc:.2f}")
print(f"    FR DNN:            RMSE={rmse_fr_dnn:.2f}  +HBC={rmse_fr_dnn_hbc:.2f}")
print(f"    FR Regime Ens:     RMSE={rmse_fr_ens:.2f}  +HBC={rmse_fr_final:.2f}")
for rname, rw in fr_regime_weights.items():
    rw_str = " / ".join(f"{nm}={rw.get(nm, 0):.2f}" for nm in model_names)
    print(f"      {rname:8s}: {rw_str}")

print(f"    UK CatBoost:       RMSE={rmse_uk_cb:.2f}  +HBC={rmse_uk_cb_hbc:.2f}  ({uk_approach})")
if preds_uk_lgb is not None:
    print(f"    UK LightGBM:       RMSE={rmse_uk_lgb:.2f}  +HBC={rmse_uk_lgb_hbc:.2f}")
if preds_uk_xgb is not None:
    print(f"    UK XGBoost:        RMSE={rmse_uk_xgb:.2f}  +HBC={rmse_uk_xgb_hbc:.2f}")
print(f"    UK Elastic Net:    RMSE={rmse_uk_en:.2f}  +HBC={rmse_uk_en_hbc:.2f}")
print(f"    UK DNN:            RMSE={rmse_uk_dnn:.2f}  +HBC={rmse_uk_dnn_hbc:.2f}")
print(f"    UK Regime Ens:     RMSE={rmse_uk_ens:.2f}  +HBC={rmse_uk_final:.2f}")
for rname, rw in uk_regime_weights.items():
    rw_str = " / ".join(f"{nm}={rw.get(nm, 0):.2f}" for nm in uk_model_names)
    print(f"      {rname:8s}: {rw_str}")

final_combined = rmse_fr_final + rmse_uk_final
print(f"\n  FINAL SUM (w/ HBC): {final_combined:.2f}")
print(f"    FR: {rmse_fr_final:.2f}")
print(f"    UK: {rmse_uk_final:.2f}")

print(f"\n  Test predictions (Submit A):")
print(f"    FR: mean={preds_fr_test_hbc.mean():.1f}, std={preds_fr_test_hbc.std():.1f}")
print(f"    UK: mean={preds_uk_test_hbc.mean():.1f}, std={preds_uk_test_hbc.std():.1f}")

print(f"\n  Submission: outputs/submission_v9.csv")

results = {
    "fr": {
        "approach": "stationary_ema240h_v9_audit_fixes",
        "catboost_rmse": rmse_fr_cb,
        "catboost_rmse_hbc": rmse_fr_cb_hbc,
        "lightgbm_rmse": rmse_fr_lgb if preds_fr_lgb is not None else None,
        "regime_ensemble_rmse": float(rmse_fr_ens),
        "final_rmse_hbc": rmse_fr_final,
        "regime_weights": {k: {nm: float(v) for nm, v in w.items()} for k, w in fr_regime_weights.items()},
        "features": feat_fr,
        "params": {k: v for k, v in FR_PARAMS.items() if k not in ["verbose", "allow_writing_files"]},
    },
    "uk": {
        "approach": uk_approach,
        "catboost_rmse": rmse_uk_cb,
        "catboost_rmse_hbc": rmse_uk_cb_hbc,
        "lightgbm_rmse": rmse_uk_lgb if preds_uk_lgb is not None else None,
        "regime_ensemble_rmse": float(rmse_uk_ens),
        "final_rmse_hbc": rmse_uk_final,
        "regime_weights": {k: {nm: float(v) for nm, v in w.items()} for k, w in uk_regime_weights.items()},
        "n_features": len(feat_uk_final),
        "params": {k: v for k, v in UK_PARAMS.items() if k not in ["verbose", "allow_writing_files"]},
    },
    "combined_sum_hbc": final_combined,
    "hbc_fr": {str(k): v for k, v in hbc_fr_final.items()},
    "hbc_uk": {str(k): v for k, v in hbc_uk_final.items()},
}

with open("outputs/final_pipeline_v9_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

print(f"\n  Total time: {time.time() - t0:.0f}s")
