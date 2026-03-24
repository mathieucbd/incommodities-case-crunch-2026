"""Final pipeline v8 — v7 + new features + UK 12m window.

v8 changes vs v7:
  - Added 7 new features: rolling_336h (4), stress_index (2), load_surprise (2)
    from feature_engineering.py (confirmed gains: FR -0.23, UK -0.17 in A/B test)
  - UK models trained on 12m window (Feb 2023 — Jan 2024) instead of full history
    (confirmed gain: UK -0.43 in multi-window test — 12m captures post-crisis regime)
  - FR models keep full training window (full is optimal for FR)

v7:  + DNN as 5th model + regime-based weights
v6:  + Elastic Net as 4th model
v5b: + Regime-based ensemble weights (5 hour groups)
v5:  + XGBoost as 3rd model
v4b: + UK MAE loss (robust to heavy tails)
v4:  + FR EMA 240h anchor

FR: Stationary target (EMA 240h) + CatBoost Optuna v2 + HBC + 7 new features
UK: Basis target (spot - merit_order_cost) + 12m window + 150+ features + HBC
Both: + LightGBM + XGBoost + Elastic Net + DNN for ensemble diversity
Ensemble: 5-model regime-weighted average optimized on validation

Usage: cd "INCOMO 3" && python scripts/final_pipeline_v8.py
"""

import sys, json, time, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, ".")
from src.data_loading import load_data, merge_train
from src.feature_engineering import build_features
from catboost import CatBoostRegressor, Pool
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
import yaml

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

# ── DNN setup ─────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    DNN_DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DNN_DEVICE = torch.device("cuda")
else:
    DNN_DEVICE = torch.device("cpu")


class ElecDNN(nn.Module):
    """Dense NN for electricity price forecasting (epftoolbox-inspired)."""
    def __init__(self, n_features, hidden_layers, dropout=0.2):
        super().__init__()
        layers = []
        in_dim = n_features
        for neurons in hidden_layers:
            layers.append(nn.Linear(in_dim, neurons))
            layers.append(nn.BatchNorm1d(neurons))
            layers.append(nn.LeakyReLU(0.01))
            layers.append(nn.Dropout(dropout))
            in_dim = neurons
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_dnn(model, X_tr, y_tr, X_va, y_va, lr=1e-3, wd=1e-4, bs=256,
              max_epochs=500, patience=30):
    """Train DNN with early stopping on validation Huber loss."""
    model = model.to(DNN_DEVICE)
    ds = TensorDataset(torch.FloatTensor(X_tr).to(DNN_DEVICE),
                       torch.FloatTensor(y_tr).to(DNN_DEVICE))
    loader = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=len(X_tr) % bs == 1)
    X_va_t = torch.FloatTensor(X_va).to(DNN_DEVICE)
    y_va_t = torch.FloatTensor(y_va).to(DNN_DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5, min_lr=1e-6)
    criterion = nn.HuberLoss(delta=5.0)
    best_loss, best_state, no_imp = float("inf"), None, 0

    for ep in range(max_epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = criterion(model(X_va_t), y_va_t).item()
        sched.step(vl)
        if vl < best_loss:
            best_loss = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
        if no_imp >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    return model, ep + 1


def predict_dnn(model, X):
    """Predict with DNN model."""
    model.eval()
    with torch.no_grad():
        return model(torch.FloatTensor(X).to(DNN_DEVICE)).cpu().numpy()


with open("config.yaml") as f:
    config = yaml.safe_load(f)

# ── Load data ─────────────────────────────────────────────────────────────
print("=" * 90)
print("  FINAL PIPELINE v8 — v7 + new features + UK 12m window")
print("=" * 90)

t0 = time.time()
x_train, y_train, x_test = load_data("data/raw")
train = merge_train(x_train, y_train)
train_fe = build_features(train, config)
test_fe = build_features(x_test, config)
print(f"  Data loaded in {time.time() - t0:.0f}s")
print(f"  Train shape: {train_fe.shape}, Test shape: {test_fe.shape}")

holdout_start = config["validation"]["holdout_start"]
mask_val = train_fe["datetime_CET"] >= holdout_start
df_train = train_fe[~mask_val].copy()
df_val = train_fe[mask_val].copy()

# v8: UK 12m window — train UK models on last 12 months only
# (confirmed -0.43 RMSE in multi-window test: post-crisis regime matches val better)
UK_12M_START = "2023-02-01"
df_train_uk = df_train[df_train["datetime_CET"] >= UK_12M_START].copy()

print(f"  Train FR (full): {len(df_train)}, Train UK (12m): {len(df_train_uk)}, "
      f"Val: {len(df_val)}, Test: {len(test_fe)}")
print(f"  DNN device: {DNN_DEVICE}")

# ── Feature lists ────────────────────────────────────────────────────────
with open("outputs/feature_selection_v5_fr.json") as f:
    fs_v5 = json.load(f)

feat_fr_27 = fs_v5["features"]
feat_fr_28 = feat_fr_27 + ["X_fr_spot_la_roll_168h_mean_x_uk_price_per_mw_7d"]

# v8: Add new confirmed features (from A/B test: combined -0.40 RMSE)
V8_NEW_FR = [
    "fr_spot_la_roll_336h_mean", "fr_spot_la_roll_336h_std",   # rolling 14d
    "fr_stress_index",                                          # thermal × (1 - renewable)
    "fr_load_surprise",                                         # load deviation from 7d mean
]
V8_NEW_UK = [
    "uk_spot_la_roll_336h_mean", "uk_spot_la_roll_336h_std",   # rolling 14d
    "uk_stress_index",                                          # thermal × (1 - renewable)
    "uk_load_surprise",                                         # load deviation from 7d mean
]
feat_fr_v8 = feat_fr_28 + [f for f in V8_NEW_FR if f not in feat_fr_28]
print(f"  FR features: {len(feat_fr_28)} (v7) → {len(feat_fr_v8)} (v8, +{len(V8_NEW_FR)} new)")

# UK: 150 confirmed features from UK feature research (basis SHAP + Boruta)
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
# HELPER: Prepare stationary target
# ══════════════════════════════════════════════════════════════════════════

def prepare_stationary(spot_col_la, spot_col, train_fe_full, df_tr, df_va):
    """Prepare stationary deviation target and weights."""
    la_col = train_fe_full[spot_col_la]
    roll_mean = la_col.ewm(span=240).mean()
    roll_std = la_col.rolling(168, min_periods=24).std()

    n_tr = len(df_tr)
    rm_tr = roll_mean.iloc[:n_tr].values
    rs_tr = roll_std.iloc[:n_tr].values
    rm_va = roll_mean.iloc[n_tr:n_tr + len(df_va)].values

    spot_tr = df_tr[spot_col].values
    spot_va = df_va[spot_col].values

    y_dev_tr = spot_tr - rm_tr
    y_dev_va = spot_va - rm_va
    valid_tr = np.isfinite(y_dev_tr)
    valid_va = np.isfinite(y_dev_va)

    dt = pd.to_datetime(df_tr["datetime_CET"])
    days_ago = (dt.max() - dt).dt.total_seconds() / 86400
    time_decay = np.exp(-2.0 * days_ago.values / 365)
    var_168h = np.clip(rs_tr ** 2, 1.0, None)
    var_168h = np.where(np.isnan(var_168h), 1.0, var_168h)
    w = time_decay / var_168h

    return {
        "y_dev_tr": y_dev_tr, "y_dev_va": y_dev_va,
        "valid_tr": valid_tr, "valid_va": valid_va,
        "weights": w, "rm_tr": rm_tr, "rm_va": rm_va,
        "spot_tr": spot_tr, "spot_va": spot_va,
    }


def compute_hbc(preds_spot, spot_va, hours_va):
    """Compute hourly bias correction."""
    errors = spot_va - preds_spot
    hb = {}
    for h in range(24):
        mask = hours_va == h
        if mask.sum() > 0:
            hb[h] = errors[mask].mean()
    corrected = preds_spot + np.array([hb.get(h, 0) for h in hours_va])
    rmse_hbc = np.sqrt(np.mean((spot_va - corrected) ** 2))
    return hb, rmse_hbc


def compute_hbc_monthly(preds_spot, spot_va, hours_va, months_va, alpha=1.0):
    """Compute monthly x hourly bias correction (120 params)."""
    errors = spot_va - preds_spot
    hb = {}
    for m in sorted(set(months_va)):
        for h in range(24):
            mask = (months_va == m) & (hours_va == h)
            if mask.sum() >= 5:
                hb[(m, h)] = alpha * errors[mask].mean()
    corrected = preds_spot + np.array([hb.get((m, h), 0) for m, h in zip(months_va, hours_va)])
    rmse_hbc = np.sqrt(np.mean((spot_va - corrected) ** 2))
    return hb, rmse_hbc


def compute_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


# ── Regime definitions ────────────────────────────────────────────────────
REGIMES = {
    "night":   [0, 1, 2, 3, 4, 5],
    "morning": [6, 7, 8, 9],
    "day":     [10, 11, 12, 13, 14, 15, 16],
    "peak":    [17, 18, 19, 20, 21],
    "late":    [22, 23],
}

HOUR_TO_REGIME = {}
for _rname, _hours in REGIMES.items():
    for _h in _hours:
        HOUR_TO_REGIME[_h] = _rname


def optimize_regime_weights(models_dict, actual, hours, label):
    """Per-regime weight optimization for 2-4 models. Returns dict of {regime: weights}."""
    names = list(models_dict.keys())
    n = len(names)
    regime_weights = {}
    ens_preds = np.zeros(len(actual))

    for rname, rhours in REGIMES.items():
        rmask = np.isin(hours, rhours)
        if rmask.sum() == 0:
            continue

        best = {"rmse": 999, "w": {names[0]: 1.0}}
        a = actual[rmask]
        p = {nm: models_dict[nm][rmask] for nm in names}

        if n == 1:
            best = {"rmse": compute_rmse(a, p[names[0]]), "w": {names[0]: 1.0}}
        elif n == 2:
            for w1 in np.arange(0.0, 1.05, 0.1):
                w2 = round(1.0 - w1, 1)
                e = w1 * p[names[0]] + w2 * p[names[1]]
                r = compute_rmse(a, e)
                if r < best["rmse"]:
                    best = {"rmse": r, "w": {names[0]: round(w1, 1), names[1]: w2}}
        elif n == 3:
            for w1 in np.arange(0.0, 1.05, 0.1):
                for w2 in np.arange(0.0, 1.05 - w1, 0.1):
                    w3 = round(1.0 - w1 - w2, 1)
                    if w3 < -0.01:
                        continue
                    e = w1 * p[names[0]] + w2 * p[names[1]] + w3 * p[names[2]]
                    r = compute_rmse(a, e)
                    if r < best["rmse"]:
                        best = {"rmse": r,
                                "w": {names[0]: round(w1, 1), names[1]: round(w2, 1), names[2]: w3}}
        elif n == 4:
            for w1 in np.arange(0.0, 1.05, 0.1):
                for w2 in np.arange(0.0, 1.05 - w1, 0.1):
                    for w3 in np.arange(0.0, 1.05 - w1 - w2, 0.1):
                        w4 = round(1.0 - w1 - w2 - w3, 1)
                        if w4 < -0.01:
                            continue
                        e = (w1 * p[names[0]] + w2 * p[names[1]] +
                             w3 * p[names[2]] + w4 * p[names[3]])
                        r = compute_rmse(a, e)
                        if r < best["rmse"]:
                            best = {"rmse": r,
                                    "w": {names[0]: round(w1, 1), names[1]: round(w2, 1),
                                          names[2]: round(w3, 1), names[3]: w4}}
        elif n == 5:
            for w1 in np.arange(0.0, 1.05, 0.1):
                for w2 in np.arange(0.0, 1.05 - w1, 0.1):
                    for w3 in np.arange(0.0, 1.05 - w1 - w2, 0.1):
                        for w4 in np.arange(0.0, 1.05 - w1 - w2 - w3, 0.1):
                            w5 = round(1.0 - w1 - w2 - w3 - w4, 1)
                            if w5 < -0.01:
                                continue
                            e = (w1 * p[names[0]] + w2 * p[names[1]] +
                                 w3 * p[names[2]] + w4 * p[names[3]] +
                                 w5 * p[names[4]])
                            r = compute_rmse(a, e)
                            if r < best["rmse"]:
                                best = {"rmse": r,
                                        "w": {names[0]: round(w1, 1), names[1]: round(w2, 1),
                                              names[2]: round(w3, 1), names[3]: round(w4, 1),
                                              names[4]: w5}}

        regime_weights[rname] = best["w"]
        ens_preds[rmask] = sum(best["w"].get(nm, 0) * p[nm] for nm in names)
        w_str = " / ".join(f"{nm}={best['w'].get(nm, 0):.1f}" for nm in names)
        print(f"    {rname:8s} (h={REGIMES[rname]}): {w_str}  RMSE={best['rmse']:.2f}  n={rmask.sum()}")

    total_rmse = compute_rmse(actual, ens_preds)
    print(f"  {label} regime ensemble: RMSE={total_rmse:.2f}")
    return regime_weights, ens_preds


def apply_regime_weights(models_dict, hours, regime_weights):
    """Apply pre-computed regime weights to test predictions."""
    names = list(models_dict.keys())
    n = len(list(models_dict.values())[0])
    ens = np.zeros(n)
    for i in range(n):
        h = hours[i]
        rname = HOUR_TO_REGIME.get(h, "day")
        w = regime_weights.get(rname, {names[0]: 1.0})
        ens[i] = sum(w.get(nm, 0) * models_dict[nm][i] for nm in names)
    return ens


# ══════════════════════════════════════════════════════════════════════════
# 1. FR MODEL — CatBoost (stationary deviation)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  1. FR CatBoost — Stationary EMA 240h (Optuna v2)")
print("=" * 90)

fr_stat = prepare_stationary("fr_spot_la", "fr_spot", train_fe, df_train, df_val)
hours_va_fr = df_val["hour"].values

# Create interaction feature
for df in [df_train, df_val, test_fe]:
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

feat_fr = [f for f in feat_fr_v8 if f in df_train.columns]
print(f"  Features: {len(feat_fr)}")

cb_fr = CatBoostRegressor(**FR_PARAMS)
cb_fr.fit(
    Pool(df_train.loc[df_train.index[fr_stat["valid_tr"]], feat_fr],
         fr_stat["y_dev_tr"][fr_stat["valid_tr"]],
         weight=fr_stat["weights"][fr_stat["valid_tr"]]),
    eval_set=Pool(df_val.loc[df_val.index[fr_stat["valid_va"]], feat_fr],
                  fr_stat["y_dev_va"][fr_stat["valid_va"]]),
    early_stopping_rounds=200, verbose=0,
)

preds_fr_dev = cb_fr.predict(df_val[feat_fr])
preds_fr_cb = fr_stat["rm_va"] + preds_fr_dev
rmse_fr_cb = compute_rmse(fr_stat["spot_va"], preds_fr_cb)
hbc_fr, rmse_fr_cb_hbc = compute_hbc(preds_fr_cb, fr_stat["spot_va"], hours_va_fr)

print(f"  CatBoost FR: RMSE={rmse_fr_cb:.2f}, +HBC={rmse_fr_cb_hbc:.2f}, "
      f"iter={cb_fr.get_best_iteration()}")


# ══════════════════════════════════════════════════════════════════════════
# 2. UK MODEL — CatBoost (basis: spot - merit_order_cost)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  2. UK CatBoost — Basis modeling (12m window + v8 features)")
print("=" * 90)

hours_va_uk = df_val["hour"].values
uk_approach = "basis_12m"

# v8: Use 12m window for UK training
uk_spot_tr = df_train_uk["uk_spot"].values
uk_spot_va = df_val["uk_spot"].values
uk_moc_tr = df_train_uk["uk_merit_order_cost"].values
uk_moc_va = df_val["uk_merit_order_cost"].values

y_basis_tr = uk_spot_tr - uk_moc_tr
y_basis_va = uk_spot_va - uk_moc_va
valid_basis_tr = np.isfinite(y_basis_tr)
valid_basis_va = np.isfinite(y_basis_va)

# UK: NO sample weights (research finding: weights hurt UK)
# MAE loss: more robust to UK's heavy tails (range [-205, +1444])
UK_PARAMS = {
    "loss_function": "MAE", "eval_metric": "RMSE",
    "iterations": 15000, "learning_rate": 0.03, "depth": 8,
    "l2_leaf_reg": 5, "colsample_bylevel": 0.8, "subsample": 0.8,
    "random_seed": 42, "verbose": 0, "allow_writing_files": False,
    "use_best_model": True,
}

feat_uk_final = [f for f in feat_uk_v8 if f in df_train_uk.columns]
print(f"  UK features (v8): {len(feat_uk_final)}")

cb_uk = CatBoostRegressor(**UK_PARAMS)
cb_uk.fit(
    Pool(df_train_uk.loc[df_train_uk.index[valid_basis_tr], feat_uk_final],
         y_basis_tr[valid_basis_tr]),
    eval_set=Pool(df_val.loc[df_val.index[valid_basis_va], feat_uk_final],
                  y_basis_va[valid_basis_va]),
    early_stopping_rounds=200, verbose=0,
)

preds_uk_basis = cb_uk.predict(df_val[feat_uk_final])
preds_uk_cb = uk_moc_va + preds_uk_basis
rmse_uk_cb = compute_rmse(uk_spot_va, preds_uk_cb)
hbc_uk, rmse_uk_cb_hbc = compute_hbc(preds_uk_cb, uk_spot_va, hours_va_uk)

print(f"  UK CatBoost (12m): RMSE={rmse_uk_cb:.2f}, +HBC={rmse_uk_cb_hbc:.2f}, "
      f"iter={cb_uk.get_best_iteration()}")


# ══════════════════════════════════════════════════════════════════════════
# 3. LightGBM MODELS (for ensemble diversity)
# ══════════════════════════════════════════════════════════════════════════
preds_fr_lgb = None
preds_uk_lgb = None

if HAS_LGB:
    print("\n" + "=" * 90)
    print("  3. LightGBM models (ensemble diversity)")
    print("=" * 90)

    # --- FR LightGBM ---
    LGB_FR_PARAMS = {
        "objective": "regression", "metric": "rmse",
        "n_estimators": 15000, "learning_rate": 0.03,
        "max_depth": 4, "num_leaves": 15,
        "reg_alpha": 5, "reg_lambda": 30,
        "subsample": 0.7, "colsample_bytree": 0.5,
        "min_child_samples": 50,
        "random_state": 42, "verbose": -1,
    }

    lgb_fr = lgb.LGBMRegressor(**LGB_FR_PARAMS)
    lgb_fr.fit(
        df_train.loc[df_train.index[fr_stat["valid_tr"]], feat_fr],
        fr_stat["y_dev_tr"][fr_stat["valid_tr"]],
        sample_weight=fr_stat["weights"][fr_stat["valid_tr"]],
        eval_set=[(df_val.loc[df_val.index[fr_stat["valid_va"]], feat_fr],
                    fr_stat["y_dev_va"][fr_stat["valid_va"]])],
        callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)],
    )

    preds_fr_dev_lgb = lgb_fr.predict(df_val[feat_fr])
    preds_fr_lgb = fr_stat["rm_va"] + preds_fr_dev_lgb
    rmse_fr_lgb = compute_rmse(fr_stat["spot_va"], preds_fr_lgb)
    _, rmse_fr_lgb_hbc = compute_hbc(preds_fr_lgb, fr_stat["spot_va"], hours_va_fr)
    print(f"  LGB FR: RMSE={rmse_fr_lgb:.2f}, +HBC={rmse_fr_lgb_hbc:.2f}, "
          f"iter={lgb_fr.best_iteration_}")

    # --- UK LightGBM (basis, no weights) ---
    LGB_UK_PARAMS = {
        "objective": "regression", "metric": "rmse",
        "n_estimators": 15000, "learning_rate": 0.02,
        "max_depth": 7, "num_leaves": 63,
        "reg_alpha": 1, "reg_lambda": 5,
        "subsample": 0.8, "colsample_bytree": 0.7,
        "min_child_samples": 30,
        "random_state": 42, "verbose": -1,
    }

    lgb_uk = lgb.LGBMRegressor(**LGB_UK_PARAMS)
    lgb_uk.fit(
        df_train_uk.loc[df_train_uk.index[valid_basis_tr], feat_uk_final],
        y_basis_tr[valid_basis_tr],
        eval_set=[(df_val.loc[df_val.index[valid_basis_va], feat_uk_final],
                    y_basis_va[valid_basis_va])],
        callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)],
    )
    preds_uk_basis_lgb = lgb_uk.predict(df_val[feat_uk_final])
    preds_uk_lgb = uk_moc_va + preds_uk_basis_lgb

    rmse_uk_lgb = compute_rmse(uk_spot_va, preds_uk_lgb)
    _, rmse_uk_lgb_hbc = compute_hbc(preds_uk_lgb, uk_spot_va, hours_va_uk)
    print(f"  LGB UK: RMSE={rmse_uk_lgb:.2f}, +HBC={rmse_uk_lgb_hbc:.2f}, "
          f"iter={lgb_uk.best_iteration_}")


# ══════════════════════════════════════════════════════════════════════════
# 3b. XGBoost MODELS (low error correlation → ensemble diversity)
# ══════════════════════════════════════════════════════════════════════════
preds_fr_xgb = None
preds_uk_xgb = None

if HAS_XGB:
    print("\n" + "=" * 90)
    print("  3b. XGBoost models (ensemble diversity)")
    print("=" * 90)

    # --- FR XGBoost ---
    XGB_FR_PARAMS = {
        "objective": "reg:squarederror", "eval_metric": "rmse",
        "n_estimators": 15000, "learning_rate": 0.05,
        "max_depth": 4, "reg_alpha": 5, "reg_lambda": 10,
        "subsample": 0.6, "colsample_bytree": 0.4,
        "min_child_weight": 15,
        "random_state": 42, "verbosity": 0, "tree_method": "hist",
    }

    xgb_fr = xgb.XGBRegressor(**XGB_FR_PARAMS)
    xgb_fr.fit(
        df_train.loc[df_train.index[fr_stat["valid_tr"]], feat_fr],
        fr_stat["y_dev_tr"][fr_stat["valid_tr"]],
        sample_weight=fr_stat["weights"][fr_stat["valid_tr"]],
        eval_set=[(df_val.loc[df_val.index[fr_stat["valid_va"]], feat_fr],
                    fr_stat["y_dev_va"][fr_stat["valid_va"]])],
        verbose=False,
    )

    preds_fr_dev_xgb = xgb_fr.predict(df_val[feat_fr])
    preds_fr_xgb = fr_stat["rm_va"] + preds_fr_dev_xgb
    rmse_fr_xgb = compute_rmse(fr_stat["spot_va"], preds_fr_xgb)
    _, rmse_fr_xgb_hbc = compute_hbc(preds_fr_xgb, fr_stat["spot_va"], hours_va_fr)
    xgb_fr_best_iter = xgb_fr.best_iteration if hasattr(xgb_fr, 'best_iteration') else XGB_FR_PARAMS["n_estimators"]
    print(f"  XGB FR: RMSE={rmse_fr_xgb:.2f}, +HBC={rmse_fr_xgb_hbc:.2f}, "
          f"iter={xgb_fr_best_iter}")

    # --- UK XGBoost (basis, no weights) ---
    XGB_UK_PARAMS = {
        "objective": "reg:squarederror", "eval_metric": "rmse",
        "n_estimators": 15000, "learning_rate": 0.03,
        "max_depth": 7, "reg_alpha": 2, "reg_lambda": 8,
        "subsample": 0.75, "colsample_bytree": 0.6,
        "min_child_weight": 20,
        "random_state": 42, "verbosity": 0, "tree_method": "hist",
    }

    xgb_uk = xgb.XGBRegressor(**XGB_UK_PARAMS)
    xgb_uk.fit(
        df_train_uk.loc[df_train_uk.index[valid_basis_tr], feat_uk_final],
        y_basis_tr[valid_basis_tr],
        eval_set=[(df_val.loc[df_val.index[valid_basis_va], feat_uk_final],
                    y_basis_va[valid_basis_va])],
        verbose=False,
    )
    preds_uk_basis_xgb = xgb_uk.predict(df_val[feat_uk_final])
    preds_uk_xgb = uk_moc_va + preds_uk_basis_xgb

    rmse_uk_xgb = compute_rmse(uk_spot_va, preds_uk_xgb)
    _, rmse_uk_xgb_hbc = compute_hbc(preds_uk_xgb, uk_spot_va, hours_va_uk)
    xgb_uk_best_iter = xgb_uk.best_iteration if hasattr(xgb_uk, 'best_iteration') else XGB_UK_PARAMS["n_estimators"]
    print(f"  XGB UK: RMSE={rmse_uk_xgb:.2f}, +HBC={rmse_uk_xgb_hbc:.2f}, "
          f"iter={xgb_uk_best_iter}")


# ══════════════════════════════════════════════════════════════════════════
# 3c. ELASTIC NET MODELS (linear diversity — low error correlation with trees)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  3c. Elastic Net models (linear diversity)")
print("=" * 90)

# --- FR Elastic Net (alpha=10, l1_ratio=0.9) ---
fr_scaler = StandardScaler()
X_fr_tr_scaled = fr_scaler.fit_transform(
    np.nan_to_num(df_train.loc[df_train.index[fr_stat["valid_tr"]], feat_fr].values, 0))
X_fr_va_scaled = fr_scaler.transform(
    np.nan_to_num(df_val[feat_fr].values, 0))

en_fr = ElasticNet(alpha=10.0, l1_ratio=0.9, max_iter=10000)
en_fr.fit(X_fr_tr_scaled, fr_stat["y_dev_tr"][fr_stat["valid_tr"]])
preds_fr_dev_en = en_fr.predict(X_fr_va_scaled)
preds_fr_en = fr_stat["rm_va"] + preds_fr_dev_en
rmse_fr_en = compute_rmse(fr_stat["spot_va"], preds_fr_en)
_, rmse_fr_en_hbc = compute_hbc(preds_fr_en, fr_stat["spot_va"], hours_va_fr)
print(f"  EN FR: RMSE={rmse_fr_en:.2f}, +HBC={rmse_fr_en_hbc:.2f}, "
      f"n_nonzero={np.sum(en_fr.coef_ != 0)}/{len(en_fr.coef_)}")

# --- UK Elastic Net (alpha=1, l1_ratio=0.9) — 12m window ---
uk_scaler = StandardScaler()
X_uk_tr_scaled = uk_scaler.fit_transform(
    np.nan_to_num(df_train_uk.loc[df_train_uk.index[valid_basis_tr], feat_uk_final].values, 0))
X_uk_va_scaled = uk_scaler.transform(
    np.nan_to_num(df_val[feat_uk_final].values, 0))

en_uk = ElasticNet(alpha=1.0, l1_ratio=0.9, max_iter=10000)
en_uk.fit(X_uk_tr_scaled, y_basis_tr[valid_basis_tr])
preds_uk_basis_en = en_uk.predict(X_uk_va_scaled)
preds_uk_en = uk_moc_va + preds_uk_basis_en
rmse_uk_en = compute_rmse(uk_spot_va, preds_uk_en)
_, rmse_uk_en_hbc = compute_hbc(preds_uk_en, uk_spot_va, hours_va_uk)
print(f"  EN UK: RMSE={rmse_uk_en:.2f}, +HBC={rmse_uk_en_hbc:.2f}, "
      f"n_nonzero={np.sum(en_uk.coef_ != 0)}/{len(en_uk.coef_)}")


# ══════════════════════════════════════════════════════════════════════════
# 3d. DNN MODELS (349 features, PyTorch, Huber loss — diverse from trees)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  3d. DNN models (349 features, PyTorch)")
print("=" * 90)

# v8: Separate scalers for FR (full) and UK (12m)
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
preds_fr_dev_dnn = predict_dnn(dnn_fr, X_dnn_va_fr)
preds_fr_dnn = fr_stat["rm_va"] + preds_fr_dev_dnn
rmse_fr_dnn = compute_rmse(fr_stat["spot_va"], preds_fr_dnn)
_, rmse_fr_dnn_hbc = compute_hbc(preds_fr_dnn, fr_stat["spot_va"], hours_va_fr)
print(f"  DNN FR: RMSE={rmse_fr_dnn:.2f}, +HBC={rmse_fr_dnn_hbc:.2f}, ep={dnn_fr_epochs}")

# --- UK DNN [768, 384, 192] — 12m window ---
torch.manual_seed(42); np.random.seed(42)
dnn_uk = ElecDNN(len(feat_dnn_final), [768, 384, 192], dropout=0.3)
dnn_uk, dnn_uk_epochs = train_dnn(
    dnn_uk, X_dnn_tr_uk[valid_basis_tr], y_basis_tr[valid_basis_tr].astype(np.float32),
    X_dnn_va_uk[valid_basis_va], y_basis_va[valid_basis_va].astype(np.float32))
preds_uk_dev_dnn = predict_dnn(dnn_uk, X_dnn_va_uk)
preds_uk_dnn = uk_moc_va + preds_uk_dev_dnn
rmse_uk_dnn = compute_rmse(uk_spot_va, preds_uk_dnn)
_, rmse_uk_dnn_hbc = compute_hbc(preds_uk_dnn, uk_spot_va, hours_va_uk)
print(f"  DNN UK: RMSE={rmse_uk_dnn:.2f}, +HBC={rmse_uk_dnn_hbc:.2f}, ep={dnn_uk_epochs}")


# ══════════════════════════════════════════════════════════════════════════
# 4. ENSEMBLE — Per-regime weight optimization on validation (5 models)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  4. Ensemble — Per-regime weight optimization (5 regimes × 5 models)")
print("=" * 90)

# Collect available model predictions
fr_models = {"CB": preds_fr_cb}
uk_models = {"CB": preds_uk_cb}
if preds_fr_lgb is not None:
    fr_models["LGB"] = preds_fr_lgb
    uk_models["LGB"] = preds_uk_lgb
if preds_fr_xgb is not None:
    fr_models["XGB"] = preds_fr_xgb
    uk_models["XGB"] = preds_uk_xgb
fr_models["EN"] = preds_fr_en
uk_models["EN"] = preds_uk_en
fr_models["DNN"] = preds_fr_dnn
uk_models["DNN"] = preds_uk_dnn

model_names = list(fr_models.keys())
print(f"  Models available: {model_names}")

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

# Use the best ensemble predictions (or CatBoost-only)
# --- Standard HBC (24 params) ---
hbc_fr_final, rmse_fr_final = compute_hbc(preds_fr_ens, fr_stat["spot_va"], hours_va_fr)
hbc_uk_final, rmse_uk_final = compute_hbc(preds_uk_ens, uk_spot_va, hours_va_uk)

print(f"  Standard HBC (24 params):")
print(f"    FR: {rmse_fr_final:.2f}")
print(f"    UK: {rmse_uk_final:.2f}")
print(f"    SUM: {rmse_fr_final + rmse_uk_final:.2f}")

# --- Monthly x Hour HBC (120 params) ---
months_va_fr = pd.to_datetime(df_val["datetime_CET"]).dt.month.values
months_va_uk = months_va_fr  # same dates

hbc_fr_monthly, rmse_fr_monthly = compute_hbc_monthly(
    preds_fr_ens, fr_stat["spot_va"], hours_va_fr, months_va_fr)
hbc_uk_monthly, rmse_uk_monthly = compute_hbc_monthly(
    preds_uk_ens, uk_spot_va, hours_va_uk, months_va_uk)

print(f"\n  Monthly x Hour HBC (120 params):")
print(f"    FR: {rmse_fr_monthly:.2f}")
print(f"    UK: {rmse_uk_monthly:.2f}")
print(f"    SUM: {rmse_fr_monthly + rmse_uk_monthly:.2f}")

# --- Dampened Monthly HBC (alpha=0.7) ---
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
# Rolling mean on ALL data (train + test)
all_data = pd.concat([train_fe, test_fe], axis=0)
fr_la_all = all_data["fr_spot_la"]
uk_la_all = all_data["uk_spot_la"]
rm_fr_all = fr_la_all.ewm(span=240).mean().values
rm_uk_all = uk_la_all.rolling(168, min_periods=24).mean().values  # unused but kept
rs_fr_all = fr_la_all.rolling(168, min_periods=24).std().values

n_full = len(train_fe)
n_test = len(test_fe)

# Create interaction feature on full data
all_data_copy = all_data.copy()
if "fr_spot_la_roll_168h_mean" in all_data_copy.columns and "uk_price_per_mw_7d" in all_data_copy.columns:
    all_data_copy["X_fr_spot_la_roll_168h_mean_x_uk_price_per_mw_7d"] = (
        all_data_copy["fr_spot_la_roll_168h_mean"] * all_data_copy["uk_price_per_mw_7d"]
    )

df_full_train = all_data_copy.iloc[:n_full].copy()
df_test_pred = all_data_copy.iloc[n_full:].copy()

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

# Train FR CatBoost on full data (no early stopping — use best_iter from val)
FR_FINAL = {**FR_PARAMS, "iterations": max(cb_fr.get_best_iteration() + 50, 500),
             "use_best_model": False}

print(f"  FR CatBoost: training on {n_full} samples, {FR_FINAL['iterations']} iterations")
cb_fr_final = CatBoostRegressor(**FR_FINAL)
cb_fr_final.fit(
    Pool(df_full_train.loc[df_full_train.index[valid_fr_full], feat_fr],
         y_dev_fr_full[valid_fr_full], weight=w_fr_full[valid_fr_full]),
    verbose=0,
)

preds_fr_test_cb = rm_fr_test + cb_fr_final.predict(df_test_pred[feat_fr])

# Hours for regime-based ensemble
hours_test = df_test_pred["hour"].values
months_test = pd.to_datetime(df_test_pred["datetime_CET"]).dt.month.values

# FR LightGBM on full data (needed by some regimes)
preds_fr_test_lgb = None
if HAS_LGB:
    fr_lgb_needs = any(fr_regime_weights.get(r, {}).get("LGB", 0) > 0 for r in REGIMES)
    if fr_lgb_needs:
        LGB_FR_FINAL = {**LGB_FR_PARAMS, "n_estimators": max(lgb_fr.best_iteration_ + 50, 500)}
        lgb_fr_final = lgb.LGBMRegressor(**LGB_FR_FINAL)
        lgb_fr_final.fit(
            df_full_train.loc[df_full_train.index[valid_fr_full], feat_fr],
            y_dev_fr_full[valid_fr_full],
            sample_weight=w_fr_full[valid_fr_full],
        )
        preds_fr_test_lgb = rm_fr_test + lgb_fr_final.predict(df_test_pred[feat_fr])
        print(f"  FR LightGBM: retrained ({LGB_FR_FINAL['n_estimators']} iters)")

# FR XGBoost on full data (needed by some regimes)
preds_fr_test_xgb = None
if HAS_XGB:
    fr_xgb_needs = any(fr_regime_weights.get(r, {}).get("XGB", 0) > 0 for r in REGIMES)
    if fr_xgb_needs:
        xgb_fr_iters = xgb_fr.best_iteration if hasattr(xgb_fr, 'best_iteration') else XGB_FR_PARAMS["n_estimators"]
        XGB_FR_FINAL = {**XGB_FR_PARAMS, "n_estimators": max(xgb_fr_iters + 50, 500)}
        xgb_fr_final = xgb.XGBRegressor(**XGB_FR_FINAL)
        xgb_fr_final.fit(
            df_full_train.loc[df_full_train.index[valid_fr_full], feat_fr],
            y_dev_fr_full[valid_fr_full],
            sample_weight=w_fr_full[valid_fr_full],
            verbose=False,
        )
        preds_fr_test_xgb = rm_fr_test + xgb_fr_final.predict(df_test_pred[feat_fr])
        print(f"  FR XGBoost: retrained ({XGB_FR_FINAL['n_estimators']} iters)")

# FR Elastic Net on full data
preds_fr_test_en = None
fr_en_needs = any(fr_regime_weights.get(r, {}).get("EN", 0) > 0 for r in REGIMES)
if fr_en_needs:
    fr_scaler_full = StandardScaler()
    X_fr_full_scaled = fr_scaler_full.fit_transform(
        np.nan_to_num(df_full_train.loc[df_full_train.index[valid_fr_full], feat_fr].values, 0))
    X_fr_test_scaled = fr_scaler_full.transform(
        np.nan_to_num(df_test_pred[feat_fr].values, 0))
    en_fr_final = ElasticNet(alpha=10.0, l1_ratio=0.9, max_iter=10000)
    en_fr_final.fit(X_fr_full_scaled, y_dev_fr_full[valid_fr_full])
    preds_fr_test_en = rm_fr_test + en_fr_final.predict(X_fr_test_scaled)
    print(f"  FR Elastic Net: retrained (n_nonzero={np.sum(en_fr_final.coef_ != 0)})")

# FR DNN on full data
preds_fr_test_dnn = None
fr_dnn_needs = any(fr_regime_weights.get(r, {}).get("DNN", 0) > 0 for r in REGIMES)
if fr_dnn_needs:
    # Scale DNN features on full data
    dnn_scaler_full = StandardScaler()
    X_dnn_full = dnn_scaler_full.fit_transform(
        np.nan_to_num(df_full_train.loc[df_full_train.index[valid_fr_full], feat_dnn_final].values, 0))
    X_dnn_test = dnn_scaler_full.transform(
        np.nan_to_num(df_test_pred[feat_dnn_final].values, 0))
    torch.manual_seed(42); np.random.seed(42)
    dnn_fr_final = ElecDNN(len(feat_dnn_final), [192, 96], dropout=0.2)
    # Train without early stopping: use val epochs as guide
    dnn_fr_final, _ = train_dnn(dnn_fr_final, X_dnn_full,
                                 y_dev_fr_full[valid_fr_full].astype(np.float32),
                                 X_dnn_full[:256], y_dev_fr_full[valid_fr_full][:256].astype(np.float32),
                                 max_epochs=dnn_fr_epochs + 5, patience=dnn_fr_epochs + 5)
    preds_fr_test_dnn = rm_fr_test + predict_dnn(dnn_fr_final, X_dnn_test)
    print(f"  FR DNN: retrained ({dnn_fr_epochs + 5} epochs)")

# FR ensemble — apply regime-specific weights
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

# Apply HBC (all 3 variants)
preds_fr_test_hbc = preds_fr_test + np.array([hbc_fr_final.get(h, 0) for h in hours_test])
preds_fr_test_monthly = preds_fr_test + np.array([hbc_fr_monthly.get((m, h), 0) for m, h in zip(months_test, hours_test)])
preds_fr_test_damp = preds_fr_test + np.array([hbc_fr_damp.get((m, h), 0) for m, h in zip(months_test, hours_test)])

print(f"  FR test predictions (std HBC): min={preds_fr_test_hbc.min():.1f}, "
      f"max={preds_fr_test_hbc.max():.1f}, mean={preds_fr_test_hbc.mean():.1f}")

# ── UK: Retrain (basis, 12m window, no weights) ────────────────────────
# v8: Use last 12 months of full training data for UK (matches val approach)
uk_12m_cutoff = pd.to_datetime(df_full_train["datetime_CET"].max()) - pd.DateOffset(months=12)
df_full_train_uk = df_full_train[df_full_train["datetime_CET"] >= uk_12m_cutoff].copy()
n_uk_12m = len(df_full_train_uk)

uk_moc_uk12m = df_full_train_uk["uk_merit_order_cost"].values
uk_moc_test = df_test_pred["uk_merit_order_cost"].values
uk_spot_uk12m = train_fe.loc[df_full_train_uk.index, "uk_spot"].values
y_basis_uk12m = uk_spot_uk12m - uk_moc_uk12m
valid_uk_12m = np.isfinite(y_basis_uk12m)

UK_FINAL = {**UK_PARAMS, "iterations": max(cb_uk.get_best_iteration() + 50, 500),
             "use_best_model": False}

print(f"  UK CatBoost (basis, 12m): training on {n_uk_12m} samples, {UK_FINAL['iterations']} iterations")
cb_uk_final = CatBoostRegressor(**UK_FINAL)
cb_uk_final.fit(
    Pool(df_full_train_uk.loc[df_full_train_uk.index[valid_uk_12m], feat_uk_final],
         y_basis_uk12m[valid_uk_12m]),
    verbose=0,
)

preds_uk_test_cb = uk_moc_test + cb_uk_final.predict(df_test_pred[feat_uk_final])

# UK LightGBM (12m window)
preds_uk_test_lgb = None
if HAS_LGB:
    uk_lgb_needs = any(uk_regime_weights.get(r, {}).get("LGB", 0) > 0 for r in REGIMES)
    if uk_lgb_needs:
        LGB_UK_FINAL = {**LGB_UK_PARAMS, "n_estimators": max(lgb_uk.best_iteration_ + 50, 500)}
        lgb_uk_final = lgb.LGBMRegressor(**LGB_UK_FINAL)
        lgb_uk_final.fit(
            df_full_train_uk.loc[df_full_train_uk.index[valid_uk_12m], feat_uk_final],
            y_basis_uk12m[valid_uk_12m],
        )
        preds_uk_test_lgb = uk_moc_test + lgb_uk_final.predict(df_test_pred[feat_uk_final])
        print(f"  UK LightGBM: retrained 12m ({LGB_UK_FINAL['n_estimators']} iters)")

# UK XGBoost (12m window)
preds_uk_test_xgb = None
if HAS_XGB:
    uk_xgb_needs = any(uk_regime_weights.get(r, {}).get("XGB", 0) > 0 for r in REGIMES)
    if uk_xgb_needs:
        xgb_uk_iters = xgb_uk.best_iteration if hasattr(xgb_uk, 'best_iteration') else XGB_UK_PARAMS["n_estimators"]
        XGB_UK_FINAL = {**XGB_UK_PARAMS, "n_estimators": max(xgb_uk_iters + 50, 500)}
        xgb_uk_final = xgb.XGBRegressor(**XGB_UK_FINAL)
        xgb_uk_final.fit(
            df_full_train_uk.loc[df_full_train_uk.index[valid_uk_12m], feat_uk_final],
            y_basis_uk12m[valid_uk_12m],
            verbose=False,
        )
        preds_uk_test_xgb = uk_moc_test + xgb_uk_final.predict(df_test_pred[feat_uk_final])
        print(f"  UK XGBoost: retrained 12m ({XGB_UK_FINAL['n_estimators']} iters)")

# UK Elastic Net (12m window)
preds_uk_test_en = None
uk_en_needs = any(uk_regime_weights.get(r, {}).get("EN", 0) > 0 for r in REGIMES)
if uk_en_needs:
    uk_scaler_full = StandardScaler()
    X_uk_full_scaled = uk_scaler_full.fit_transform(
        np.nan_to_num(df_full_train_uk.loc[df_full_train_uk.index[valid_uk_12m], feat_uk_final].values, 0))
    X_uk_test_scaled = uk_scaler_full.transform(
        np.nan_to_num(df_test_pred[feat_uk_final].values, 0))
    en_uk_final = ElasticNet(alpha=1.0, l1_ratio=0.9, max_iter=10000)
    en_uk_final.fit(X_uk_full_scaled, y_basis_uk12m[valid_uk_12m])
    preds_uk_test_en = uk_moc_test + en_uk_final.predict(X_uk_test_scaled)
    print(f"  UK Elastic Net: retrained 12m (n_nonzero={np.sum(en_uk_final.coef_ != 0)})")

# UK DNN (12m window)
preds_uk_test_dnn = None
uk_dnn_needs = any(uk_regime_weights.get(r, {}).get("DNN", 0) > 0 for r in REGIMES)
if uk_dnn_needs:
    dnn_scaler_uk_full = StandardScaler()
    X_dnn_uk_full = dnn_scaler_uk_full.fit_transform(
        np.nan_to_num(df_full_train_uk.loc[df_full_train_uk.index[valid_uk_12m], feat_dnn_final].values, 0))
    X_dnn_uk_test = dnn_scaler_uk_full.transform(
        np.nan_to_num(df_test_pred[feat_dnn_final].values, 0))
    torch.manual_seed(42); np.random.seed(42)
    dnn_uk_final = ElecDNN(len(feat_dnn_final), [768, 384, 192], dropout=0.3)
    dnn_uk_final, _ = train_dnn(dnn_uk_final, X_dnn_uk_full,
                                 y_basis_uk12m[valid_uk_12m].astype(np.float32),
                                 X_dnn_uk_full[:256], y_basis_uk12m[valid_uk_12m][:256].astype(np.float32),
                                 max_epochs=dnn_uk_epochs + 5, patience=dnn_uk_epochs + 5)
    preds_uk_test_dnn = uk_moc_test + predict_dnn(dnn_uk_final, X_dnn_uk_test)
    print(f"  UK DNN: retrained 12m ({dnn_uk_epochs + 5} epochs)")

# UK ensemble — apply regime-specific weights
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

# Apply HBC to UK (all 3 variants)
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

# Clipping: use train quantiles
fr_q_low = np.percentile(train_fe["fr_spot"].dropna(), 0.1)
fr_q_high = np.percentile(train_fe["fr_spot"].dropna(), 99.9)
uk_q_low = np.percentile(train_fe["uk_spot"].dropna(), 0.1)
uk_q_high = np.percentile(train_fe["uk_spot"].dropna(), 99.9)

print(f"  FR clipping: [{fr_q_low:.1f}, {fr_q_high:.1f}]")
print(f"  UK clipping: [{uk_q_low:.1f}, {uk_q_high:.1f}]")

# Submit A — Standard HBC (24 params) + retrain iters 500
sub_a = pd.DataFrame({
    "id": test_fe.index,
    "fr_spot": np.clip(preds_fr_test_hbc, fr_q_low, fr_q_high),
    "uk_spot": np.clip(preds_uk_test_hbc, uk_q_low, uk_q_high),
})
sub_a.to_csv("outputs/submission_v8.csv", index=False)
sub_a.to_csv("outputs/submission.csv", index=False)
print(f"  submission_v8.csv — {len(sub_a)} rows")


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
    rw_str = " / ".join(f"{nm}={rw.get(nm, 0):.1f}" for nm in model_names)
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
    rw_str = " / ".join(f"{nm}={rw.get(nm, 0):.1f}" for nm in model_names)
    print(f"      {rname:8s}: {rw_str}")

final_combined = rmse_fr_final + rmse_uk_final
print(f"\n  FINAL SUM (w/ HBC): {final_combined:.2f}")
print(f"    FR: {rmse_fr_final:.2f}")
print(f"    UK: {rmse_uk_final:.2f}")

print(f"\n  Test predictions (Submit A):")
print(f"    FR: mean={preds_fr_test_hbc.mean():.1f}, std={preds_fr_test_hbc.std():.1f}")
print(f"    UK: mean={preds_uk_test_hbc.mean():.1f}, std={preds_uk_test_hbc.std():.1f}")

print(f"\n  Submission: outputs/submission_v8.csv")

# Save results
results = {
    "fr": {
        "approach": "stationary_ema240h_v8_newfeats",
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

with open("outputs/final_pipeline_v8_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

print(f"\n  Total time: {time.time() - t0:.0f}s")
