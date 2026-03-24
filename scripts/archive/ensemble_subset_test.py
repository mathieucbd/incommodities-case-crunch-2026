"""Test which model subsets give the best ensemble.

Question: Do we need all 5 models or is a simpler combo better?
Risk: More models = more weight params = more overfitting on validation.
"""

import sys, yaml, warnings, time, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from catboost import CatBoostRegressor, Pool
import lightgbm as lgb
import xgboost as xgb

sys.path.insert(0, ".")
from src.data_loading import load_data, merge_train
from src.feature_engineering import build_features

warnings.filterwarnings("ignore")

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

with open("config.yaml") as f:
    config = yaml.safe_load(f)

t0 = time.time()
x_train, y_train, x_test = load_data("data/raw")
train = merge_train(x_train, y_train)
df = build_features(train, config)

holdout_start = config["validation"]["holdout_start"]
mask_val = (df["datetime_CET"] >= holdout_start).values
df_tr = df[~mask_val].copy()
df_va = df[mask_val].copy()

# Features
with open("outputs/feature_selection_v5_fr.json") as f:
    fs_v5 = json.load(f)
FR_FEAT = [f for f in fs_v5["features"] + ["X_fr_spot_la_roll_168h_mean_x_uk_price_per_mw_7d"]
           if f in df_tr.columns]

with open("outputs/uk_feature_research.json") as f:
    uk_research = json.load(f)
UK_FEAT = [f for f in uk_research["confirmed_features"] if f in df_tr.columns]

# DNN features
EXCLUDE = {"fr_spot", "uk_spot", "datetime_CET", "datetime_UTC", "date", "id"}
_all = [c for c in df_tr.columns if c not in EXCLUDE
        and df_tr[c].dtype in ["float64", "float32", "int64", "int32"]
        and df_tr[c].notna().sum() > len(df_tr) * 0.5]
_corr = df_tr[_all].corr().abs()
_drop = set()
for i in range(len(_all)):
    if _all[i] in _drop: continue
    for j in range(i+1, len(_all)):
        if _all[j] in _drop: continue
        if _corr.iloc[i,j] > 0.99: _drop.add(_all[j])
DNN_FEAT = [f for f in _all if f not in _drop]

# Interaction feature
for d in [df_tr, df_va]:
    if "fr_spot_la_roll_168h_mean" in d.columns and "uk_price_per_mw_7d" in d.columns:
        d["X_fr_spot_la_roll_168h_mean_x_uk_price_per_mw_7d"] = (
            d["fr_spot_la_roll_168h_mean"] * d["uk_price_per_mw_7d"])

# Targets
fr_la = df["fr_spot_la"].values
ema_fr = pd.Series(fr_la).ewm(span=240).mean().values
fr_rm_va = ema_fr[mask_val]
fr_spot_va = df_va["fr_spot"].values
fr_y_tr = df_tr["fr_spot"].values - ema_fr[~mask_val]
fr_y_va = fr_spot_va - fr_rm_va
fr_vt = np.isfinite(fr_y_tr) & np.isfinite(ema_fr[~mask_val])
fr_vv = np.isfinite(fr_y_va) & np.isfinite(fr_rm_va)

uk_moc_va = df_va["uk_merit_order_cost"].values
uk_spot_va = df_va["uk_spot"].values
uk_y_tr = df_tr["uk_spot"].values - df_tr["uk_merit_order_cost"].values
uk_y_va = uk_spot_va - uk_moc_va
uk_vt = np.isfinite(uk_y_tr)
uk_vv = np.isfinite(uk_y_va)

hours_va = df_va["hour"].values

# Sample weights FR
days_ago = (df_tr["datetime_CET"].max() - df_tr["datetime_CET"]).dt.total_seconds() / 86400
rs = df_tr["fr_spot_la"].rolling(168, min_periods=24).std().fillna(df_tr["fr_spot_la"].std())
sw = np.exp(-2 * days_ago.values / 365) / np.clip(rs.values ** 2, 1, None)
sw[~fr_vt] = 0

REGIMES = {
    "night": [0,1,2,3,4,5], "morning": [6,7,8,9],
    "day": [10,11,12,13,14,15,16], "peak": [17,18,19,20,21], "late": [22,23],
}
HOUR_TO_REGIME = {}
for r, hs in REGIMES.items():
    for h in hs: HOUR_TO_REGIME[h] = r

def compute_rmse(a, p): return np.sqrt(np.mean((a - p) ** 2))

def compute_hbc(preds, actual, hours):
    errors = actual - preds
    hbc = {h: errors[hours == h].mean() for h in range(24) if (hours == h).sum() > 0}
    corrected = preds + np.array([hbc.get(h, 0) for h in hours])
    return np.sqrt(np.mean((actual - corrected) ** 2))


def optimize_weights(preds_dict, actual, hours):
    """Regime weight optimization for 1-5 models."""
    names = list(preds_dict.keys())
    n = len(names)
    regime_w = {}
    ens = np.zeros(len(actual))

    for rname, rhours in REGIMES.items():
        m = np.isin(hours, rhours)
        if m.sum() == 0: continue
        a = actual[m]
        p = {nm: preds_dict[nm][m] for nm in names}
        best_rmse, best_w = 999, {names[0]: 1.0}

        if n == 1:
            best_rmse = compute_rmse(a, p[names[0]])
            best_w = {names[0]: 1.0}
        elif n == 2:
            for w1 in np.arange(0.0, 1.05, 0.1):
                w2 = round(1.0 - w1, 1)
                e = w1 * p[names[0]] + w2 * p[names[1]]
                r = compute_rmse(a, e)
                if r < best_rmse:
                    best_rmse = r
                    best_w = {names[0]: round(w1,1), names[1]: w2}
        elif n == 3:
            for w1 in np.arange(0.0, 1.05, 0.1):
                for w2 in np.arange(0.0, 1.05-w1, 0.1):
                    w3 = round(1.0-w1-w2, 1)
                    if w3 < -0.01: continue
                    e = w1*p[names[0]] + w2*p[names[1]] + w3*p[names[2]]
                    r = compute_rmse(a, e)
                    if r < best_rmse:
                        best_rmse = r
                        best_w = {names[0]: round(w1,1), names[1]: round(w2,1), names[2]: w3}
        elif n == 4:
            for w1 in np.arange(0.0, 1.05, 0.1):
                for w2 in np.arange(0.0, 1.05-w1, 0.1):
                    for w3 in np.arange(0.0, 1.05-w1-w2, 0.1):
                        w4 = round(1.0-w1-w2-w3, 1)
                        if w4 < -0.01: continue
                        e = w1*p[names[0]] + w2*p[names[1]] + w3*p[names[2]] + w4*p[names[3]]
                        r = compute_rmse(a, e)
                        if r < best_rmse:
                            best_rmse = r
                            best_w = {names[0]: round(w1,1), names[1]: round(w2,1),
                                      names[2]: round(w3,1), names[3]: w4}
        elif n == 5:
            for w1 in np.arange(0.0, 1.05, 0.1):
                for w2 in np.arange(0.0, 1.05-w1, 0.1):
                    for w3 in np.arange(0.0, 1.05-w1-w2, 0.1):
                        for w4 in np.arange(0.0, 1.05-w1-w2-w3, 0.1):
                            w5 = round(1.0-w1-w2-w3-w4, 1)
                            if w5 < -0.01: continue
                            e = (w1*p[names[0]] + w2*p[names[1]] + w3*p[names[2]] +
                                 w4*p[names[3]] + w5*p[names[4]])
                            r = compute_rmse(a, e)
                            if r < best_rmse:
                                best_rmse = r
                                best_w = {names[0]: round(w1,1), names[1]: round(w2,1),
                                          names[2]: round(w3,1), names[3]: round(w4,1), names[4]: w5}

        regime_w[rname] = best_w
        ens[m] = sum(best_w.get(nm, 0) * p[nm] for nm in names)

    return regime_w, ens


# ── DNN class ─────────────────────────────────────────────────────────
class ElecDNN(nn.Module):
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

def train_dnn(model, X_tr, y_tr, X_va, y_va, lr=1e-3, wd=1e-4, bs=256, max_ep=500, pat=30):
    model = model.to(DEVICE)
    ds = TensorDataset(torch.FloatTensor(X_tr).to(DEVICE), torch.FloatTensor(y_tr).to(DEVICE))
    loader = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=len(X_tr)%bs==1)
    Xv = torch.FloatTensor(X_va).to(DEVICE)
    yv = torch.FloatTensor(y_va).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5, min_lr=1e-6)
    crit = nn.HuberLoss(delta=5.0)
    bl, bs_state, ni = float("inf"), None, 0
    for ep in range(max_ep):
        model.train()
        for xb, yb in loader:
            opt.zero_grad(); l = crit(model(xb), yb); l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0); opt.step()
        model.eval()
        with torch.no_grad(): vl = crit(model(Xv), yv).item()
        sched.step(vl)
        if vl < bl: bl = vl; bs_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}; ni = 0
        else: ni += 1
        if ni >= pat: break
    model.load_state_dict(bs_state); model.eval()
    return model

def pred_dnn(model, X):
    model.eval()
    with torch.no_grad(): return model(torch.FloatTensor(X).to(DEVICE)).cpu().numpy()


# ══════════════════════════════════════════════════════════════════════
#  TRAIN ALL 5 MODELS
# ══════════════════════════════════════════════════════════════════════
print("=" * 90)
print("  Training all 5 models...")
print("=" * 90)

# CB FR
cb_p = config.get("catboost_params_fr_optuna_v2", config.get("catboost_params_fr", {}))
cb_fr = CatBoostRegressor(**{**cb_p, "verbose": 0, "iterations": 15000,
    "learning_rate": 0.059, "depth": 3, "l2_leaf_reg": 4.42,
    "subsample": 0.533, "colsample_bylevel": 0.228, "min_child_samples": 14,
    "random_strength": 0.9, "use_best_model": True, "allow_writing_files": False})
cb_fr.fit(Pool(df_tr[FR_FEAT].values[fr_vt], fr_y_tr[fr_vt], weight=sw[fr_vt]),
          eval_set=Pool(df_va[FR_FEAT].values, fr_y_va), early_stopping_rounds=200, verbose=0)
p_fr = {"CB": fr_rm_va + cb_fr.predict(df_va[FR_FEAT].values)}

# CB UK
uk_p = {"loss_function": "MAE", "eval_metric": "RMSE", "iterations": 15000,
    "learning_rate": 0.03, "depth": 8, "l2_leaf_reg": 5, "colsample_bylevel": 0.8,
    "subsample": 0.8, "random_seed": 42, "verbose": 0, "allow_writing_files": False, "use_best_model": True}
cb_uk = CatBoostRegressor(**uk_p)
cb_uk.fit(Pool(df_tr[UK_FEAT].values[uk_vt], uk_y_tr[uk_vt]),
          eval_set=Pool(df_va[UK_FEAT].values, uk_y_va), early_stopping_rounds=200, verbose=0)
p_uk = {"CB": uk_moc_va + cb_uk.predict(df_va[UK_FEAT].values)}

# LGB
lgb_fr_p = {"objective": "regression", "metric": "rmse", "n_estimators": 15000, "learning_rate": 0.03,
    "max_depth": 4, "num_leaves": 15, "reg_alpha": 5, "reg_lambda": 30,
    "subsample": 0.7, "colsample_bytree": 0.5, "min_child_samples": 50, "random_state": 42, "verbose": -1}
lgb_fr = lgb.LGBMRegressor(**lgb_fr_p)
lgb_fr.fit(df_tr[FR_FEAT].values[fr_vt], fr_y_tr[fr_vt], sample_weight=sw[fr_vt],
           eval_set=[(df_va[FR_FEAT].values[fr_vv], fr_y_va[fr_vv])],
           callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
p_fr["LGB"] = fr_rm_va + lgb_fr.predict(df_va[FR_FEAT].values)

lgb_uk_p = {"objective": "regression", "metric": "rmse", "n_estimators": 15000, "learning_rate": 0.02,
    "max_depth": 7, "num_leaves": 63, "reg_alpha": 1, "reg_lambda": 5,
    "subsample": 0.8, "colsample_bytree": 0.7, "min_child_samples": 30, "random_state": 42, "verbose": -1}
lgb_uk = lgb.LGBMRegressor(**lgb_uk_p)
lgb_uk.fit(df_tr[UK_FEAT].values[uk_vt], uk_y_tr[uk_vt],
           eval_set=[(df_va[UK_FEAT].values[uk_vv], uk_y_va[uk_vv])],
           callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
p_uk["LGB"] = uk_moc_va + lgb_uk.predict(df_va[UK_FEAT].values)

# XGB
xgb_fr = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=15000, learning_rate=0.05,
    max_depth=4, reg_alpha=5, reg_lambda=10, subsample=0.6, colsample_bytree=0.4,
    min_child_weight=15, random_state=42, verbosity=0, tree_method="hist")
xgb_fr.fit(df_tr[FR_FEAT].values[fr_vt], fr_y_tr[fr_vt], sample_weight=sw[fr_vt],
           eval_set=[(df_va[FR_FEAT].values[fr_vv], fr_y_va[fr_vv])], verbose=False)
p_fr["XGB"] = fr_rm_va + xgb_fr.predict(df_va[FR_FEAT].values)

xgb_uk = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=15000, learning_rate=0.03,
    max_depth=7, reg_alpha=2, reg_lambda=8, subsample=0.75, colsample_bytree=0.6,
    min_child_weight=20, random_state=42, verbosity=0, tree_method="hist")
xgb_uk.fit(df_tr[UK_FEAT].values[uk_vt], uk_y_tr[uk_vt],
           eval_set=[(df_va[UK_FEAT].values[uk_vv], uk_y_va[uk_vv])], verbose=False)
p_uk["XGB"] = uk_moc_va + xgb_uk.predict(df_va[UK_FEAT].values)

# EN
sc_fr = StandardScaler()
Xf_tr = sc_fr.fit_transform(np.nan_to_num(df_tr[FR_FEAT].values[fr_vt], 0))
Xf_va = sc_fr.transform(np.nan_to_num(df_va[FR_FEAT].values, 0))
en_fr = ElasticNet(alpha=10, l1_ratio=0.9, max_iter=10000)
en_fr.fit(Xf_tr, fr_y_tr[fr_vt])
p_fr["EN"] = fr_rm_va + en_fr.predict(Xf_va)

sc_uk = StandardScaler()
Xu_tr = sc_uk.fit_transform(np.nan_to_num(df_tr[UK_FEAT].values[uk_vt], 0))
Xu_va = sc_uk.transform(np.nan_to_num(df_va[UK_FEAT].values, 0))
en_uk = ElasticNet(alpha=1, l1_ratio=0.9, max_iter=10000)
en_uk.fit(Xu_tr, uk_y_tr[uk_vt])
p_uk["EN"] = uk_moc_va + en_uk.predict(Xu_va)

# DNN
dsc = StandardScaler()
Xd_tr = dsc.fit_transform(np.nan_to_num(df_tr[DNN_FEAT].values, 0))
Xd_va = dsc.transform(np.nan_to_num(df_va[DNN_FEAT].values, 0))

torch.manual_seed(42); np.random.seed(42)
dnn_fr = train_dnn(ElecDNN(len(DNN_FEAT), [192, 96], 0.2),
                   Xd_tr[fr_vt], fr_y_tr[fr_vt].astype(np.float32),
                   Xd_va[fr_vv], fr_y_va[fr_vv].astype(np.float32))
p_fr["DNN"] = fr_rm_va + pred_dnn(dnn_fr, Xd_va)

torch.manual_seed(42); np.random.seed(42)
dnn_uk = train_dnn(ElecDNN(len(DNN_FEAT), [768, 384, 192], 0.3),
                   Xd_tr[uk_vt], uk_y_tr[uk_vt].astype(np.float32),
                   Xd_va[uk_vv], uk_y_va[uk_vv].astype(np.float32))
p_uk["DNN"] = uk_moc_va + pred_dnn(dnn_uk, Xd_va)

print("  All models trained.\n")

# Print individual scores
print("  Individual model scores (+HBC):")
for nm in ["CB", "LGB", "XGB", "EN", "DNN"]:
    fr_hbc = compute_hbc(p_fr[nm], fr_spot_va, hours_va)
    uk_hbc = compute_hbc(p_uk[nm], uk_spot_va, hours_va)
    print(f"    {nm:4s}:  FR={fr_hbc:.2f}  UK={uk_hbc:.2f}  SUM={fr_hbc+uk_hbc:.2f}")

# ══════════════════════════════════════════════════════════════════════
#  TEST ALL SUBSETS
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*90}")
print("  ENSEMBLE SUBSET COMPARISON")
print(f"{'='*90}")

SUBSETS = [
    # 2-model combos
    ("DNN+EN", ["DNN", "EN"]),
    ("DNN+CB", ["DNN", "CB"]),
    ("CB+EN", ["CB", "EN"]),
    # 3-model combos
    ("DNN+EN+CB", ["CB", "EN", "DNN"]),
    ("DNN+EN+LGB", ["LGB", "EN", "DNN"]),
    ("DNN+CB+LGB", ["CB", "LGB", "DNN"]),
    ("CB+LGB+EN", ["CB", "LGB", "EN"]),
    # 4-model combos
    ("CB+LGB+EN+DNN", ["CB", "LGB", "EN", "DNN"]),
    ("CB+LGB+XGB+EN", ["CB", "LGB", "XGB", "EN"]),
    ("CB+LGB+XGB+DNN", ["CB", "LGB", "XGB", "DNN"]),
    # 5-model
    ("ALL 5", ["CB", "LGB", "XGB", "EN", "DNN"]),
]

results = []
for label, models in SUBSETS:
    t1 = time.time()
    fr_sub = {nm: p_fr[nm] for nm in models}
    uk_sub = {nm: p_uk[nm] for nm in models}

    fr_rw, fr_ens = optimize_weights(fr_sub, fr_spot_va, hours_va)
    uk_rw, uk_ens = optimize_weights(uk_sub, uk_spot_va, hours_va)

    fr_hbc = compute_hbc(fr_ens, fr_spot_va, hours_va)
    uk_hbc = compute_hbc(uk_ens, uk_spot_va, hours_va)
    total = fr_hbc + uk_hbc
    elapsed = time.time() - t1

    results.append({"label": label, "n": len(models), "fr": fr_hbc, "uk": uk_hbc,
                    "sum": total, "time": elapsed, "fr_w": fr_rw, "uk_w": uk_rw})
    print(f"  {label:22s} ({len(models)}m):  FR={fr_hbc:.2f}  UK={uk_hbc:.2f}  SUM={total:.2f}  ({elapsed:.1f}s)")

# Sort by SUM
print(f"\n{'='*90}")
print("  RANKING BY SUM")
print(f"{'='*90}")
for i, r in enumerate(sorted(results, key=lambda x: x["sum"])):
    marker = " <<<" if i == 0 else ""
    print(f"  {i+1}. {r['label']:22s} ({r['n']}m):  FR={r['fr']:.2f}  UK={r['uk']:.2f}  SUM={r['sum']:.2f}{marker}")

# Show weights for top 3
print(f"\n{'='*90}")
print("  TOP 3 — Regime weights detail")
print(f"{'='*90}")
for r in sorted(results, key=lambda x: x["sum"])[:3]:
    print(f"\n  {r['label']}  SUM={r['sum']:.2f}")
    for mkt, rw in [("FR", r["fr_w"]), ("UK", r["uk_w"])]:
        print(f"    {mkt}:")
        for regime, w in rw.items():
            w_str = " / ".join(f"{k}={v:.1f}" for k, v in w.items())
            print(f"      {regime:10s}: {w_str}")

print(f"\n  Total time: {time.time() - t0:.0f}s")
