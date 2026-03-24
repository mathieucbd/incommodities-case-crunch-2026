"""Test best ensemble PER MARKET independently.

FR and UK may benefit from different model combinations.
Tests all possible subsets (2^5 - 1 = 31 combos per market).
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
from itertools import combinations

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

days_ago = (df_tr["datetime_CET"].max() - df_tr["datetime_CET"]).dt.total_seconds() / 86400
rs = df_tr["fr_spot_la"].rolling(168, min_periods=24).std().fillna(df_tr["fr_spot_la"].std())
sw = np.exp(-2 * days_ago.values / 365) / np.clip(rs.values ** 2, 1, None)
sw[~fr_vt] = 0

REGIMES = {
    "night": [0,1,2,3,4,5], "morning": [6,7,8,9],
    "day": [10,11,12,13,14,15,16], "peak": [17,18,19,20,21], "late": [22,23],
}

def compute_rmse(a, p): return np.sqrt(np.mean((a - p) ** 2))

def compute_hbc(preds, actual, hours):
    errors = actual - preds
    hbc = {h: errors[hours == h].mean() for h in range(24) if (hours == h).sum() > 0}
    corrected = preds + np.array([hbc.get(h, 0) for h in hours])
    return np.sqrt(np.mean((actual - corrected) ** 2))

def optimize_weights(preds_dict, actual, hours):
    names = list(preds_dict.keys())
    n = len(names)
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
                if r < best_rmse: best_rmse = r; best_w = {names[0]: round(w1,1), names[1]: w2}
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

        ens[m] = sum(best_w.get(nm, 0) * p[nm] for nm in names)

    return ens


# ── DNN ───────────────────────────────────────────────────────────────
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
    def forward(self, x): return self.net(x).squeeze(-1)

def train_dnn(model, X_tr, y_tr, X_va, y_va, lr=1e-3, wd=1e-4, bs=256, max_ep=500, pat=30):
    model = model.to(DEVICE)
    ds = TensorDataset(torch.FloatTensor(X_tr).to(DEVICE), torch.FloatTensor(y_tr).to(DEVICE))
    loader = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=len(X_tr)%bs==1)
    Xv = torch.FloatTensor(X_va).to(DEVICE); yv = torch.FloatTensor(y_va).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5, min_lr=1e-6)
    crit = nn.HuberLoss(delta=5.0)
    bl, bst, ni = float("inf"), None, 0
    for ep in range(max_ep):
        model.train()
        for xb, yb in loader:
            opt.zero_grad(); l = crit(model(xb), yb); l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0); opt.step()
        model.eval()
        with torch.no_grad(): vl = crit(model(Xv), yv).item()
        sched.step(vl)
        if vl < bl: bl = vl; bst = {k: v.cpu().clone() for k, v in model.state_dict().items()}; ni = 0
        else: ni += 1
        if ni >= pat: break
    model.load_state_dict(bst); model.eval(); return model

def pred_dnn(model, X):
    model.eval()
    with torch.no_grad(): return model(torch.FloatTensor(X).to(DEVICE)).cpu().numpy()


# ══════════════════════════════════════════════════════════════════════
print("=" * 90)
print("  Training all 5 models...")
print("=" * 90)

# CB
cb_fr = CatBoostRegressor(loss_function="RMSE", eval_metric="RMSE", iterations=15000,
    learning_rate=0.059, depth=3, l2_leaf_reg=4.42, subsample=0.533, colsample_bylevel=0.228,
    min_child_samples=14, random_strength=0.9, random_seed=42, verbose=0, allow_writing_files=False,
    use_best_model=True)
cb_fr.fit(Pool(df_tr[FR_FEAT].values[fr_vt], fr_y_tr[fr_vt], weight=sw[fr_vt]),
          eval_set=Pool(df_va[FR_FEAT].values, fr_y_va), early_stopping_rounds=200, verbose=0)
p_fr = {"CB": fr_rm_va + cb_fr.predict(df_va[FR_FEAT].values)}

cb_uk = CatBoostRegressor(loss_function="MAE", eval_metric="RMSE", iterations=15000,
    learning_rate=0.03, depth=8, l2_leaf_reg=5, colsample_bylevel=0.8, subsample=0.8,
    random_seed=42, verbose=0, allow_writing_files=False, use_best_model=True)
cb_uk.fit(Pool(df_tr[UK_FEAT].values[uk_vt], uk_y_tr[uk_vt]),
          eval_set=Pool(df_va[UK_FEAT].values, uk_y_va), early_stopping_rounds=200, verbose=0)
p_uk = {"CB": uk_moc_va + cb_uk.predict(df_va[UK_FEAT].values)}

# LGB
lgb_fr = lgb.LGBMRegressor(objective="regression", metric="rmse", n_estimators=15000, learning_rate=0.03,
    max_depth=4, num_leaves=15, reg_alpha=5, reg_lambda=30, subsample=0.7, colsample_bytree=0.5,
    min_child_samples=50, random_state=42, verbose=-1)
lgb_fr.fit(df_tr[FR_FEAT].values[fr_vt], fr_y_tr[fr_vt], sample_weight=sw[fr_vt],
           eval_set=[(df_va[FR_FEAT].values[fr_vv], fr_y_va[fr_vv])],
           callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
p_fr["LGB"] = fr_rm_va + lgb_fr.predict(df_va[FR_FEAT].values)

lgb_uk = lgb.LGBMRegressor(objective="regression", metric="rmse", n_estimators=15000, learning_rate=0.02,
    max_depth=7, num_leaves=63, reg_alpha=1, reg_lambda=5, subsample=0.8, colsample_bytree=0.7,
    min_child_samples=30, random_state=42, verbose=-1)
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
p_fr["EN"] = fr_rm_va + ElasticNet(alpha=10, l1_ratio=0.9, max_iter=10000).fit(
    sc_fr.fit_transform(np.nan_to_num(df_tr[FR_FEAT].values[fr_vt], 0)), fr_y_tr[fr_vt]).predict(
    sc_fr.transform(np.nan_to_num(df_va[FR_FEAT].values, 0)))

sc_uk = StandardScaler()
p_uk["EN"] = uk_moc_va + ElasticNet(alpha=1, l1_ratio=0.9, max_iter=10000).fit(
    sc_uk.fit_transform(np.nan_to_num(df_tr[UK_FEAT].values[uk_vt], 0)), uk_y_tr[uk_vt]).predict(
    sc_uk.transform(np.nan_to_num(df_va[UK_FEAT].values, 0)))

# DNN
dsc = StandardScaler()
Xd_tr = dsc.fit_transform(np.nan_to_num(df_tr[DNN_FEAT].values, 0))
Xd_va = dsc.transform(np.nan_to_num(df_va[DNN_FEAT].values, 0))

torch.manual_seed(42); np.random.seed(42)
p_fr["DNN"] = fr_rm_va + pred_dnn(train_dnn(
    ElecDNN(len(DNN_FEAT), [192, 96], 0.2),
    Xd_tr[fr_vt], fr_y_tr[fr_vt].astype(np.float32),
    Xd_va[fr_vv], fr_y_va[fr_vv].astype(np.float32)), Xd_va)

torch.manual_seed(42); np.random.seed(42)
p_uk["DNN"] = uk_moc_va + pred_dnn(train_dnn(
    ElecDNN(len(DNN_FEAT), [768, 384, 192], 0.3),
    Xd_tr[uk_vt], uk_y_tr[uk_vt].astype(np.float32),
    Xd_va[uk_vv], uk_y_va[uk_vv].astype(np.float32)), Xd_va)

print("  Done.\n")

ALL_MODELS = ["CB", "LGB", "XGB", "EN", "DNN"]

# Individual scores
print("  Individual +HBC:")
for nm in ALL_MODELS:
    fr_h = compute_hbc(p_fr[nm], fr_spot_va, hours_va)
    uk_h = compute_hbc(p_uk[nm], uk_spot_va, hours_va)
    print(f"    {nm:4s}:  FR={fr_h:.2f}  UK={uk_h:.2f}")

# ══════════════════════════════════════════════════════════════════════
#  ALL SUBSETS PER MARKET
# ══════════════════════════════════════════════════════════════════════
def all_subsets(models):
    """Generate all non-empty subsets."""
    subs = []
    for r in range(1, len(models) + 1):
        for combo in combinations(models, r):
            subs.append(list(combo))
    return subs

subsets = all_subsets(ALL_MODELS)
print(f"\n  Testing {len(subsets)} subsets per market...")

# FR
print(f"\n{'='*90}")
print("  FR — All subsets ranked")
print(f"{'='*90}")
fr_results = []
for sub in subsets:
    preds_sub = {nm: p_fr[nm] for nm in sub}
    ens = optimize_weights(preds_sub, fr_spot_va, hours_va)
    hbc = compute_hbc(ens, fr_spot_va, hours_va)
    fr_results.append({"models": "+".join(sub), "n": len(sub), "hbc": hbc})

fr_sorted = sorted(fr_results, key=lambda x: x["hbc"])
for i, r in enumerate(fr_sorted[:15]):
    print(f"  {i+1:2d}. {r['models']:30s} ({r['n']}m)  +HBC={r['hbc']:.2f}")

# UK
print(f"\n{'='*90}")
print("  UK — All subsets ranked")
print(f"{'='*90}")
uk_results = []
for sub in subsets:
    preds_sub = {nm: p_uk[nm] for nm in sub}
    ens = optimize_weights(preds_sub, uk_spot_va, hours_va)
    hbc = compute_hbc(ens, uk_spot_va, hours_va)
    uk_results.append({"models": "+".join(sub), "n": len(sub), "hbc": hbc})

uk_sorted = sorted(uk_results, key=lambda x: x["hbc"])
for i, r in enumerate(uk_sorted[:15]):
    print(f"  {i+1:2d}. {r['models']:30s} ({r['n']}m)  +HBC={r['hbc']:.2f}")

# ══════════════════════════════════════════════════════════════════════
#  BEST COMBO: pick best FR + best UK independently
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*90}")
print("  OPTIMAL MIX — Best per market")
print(f"{'='*90}")

best_fr = fr_sorted[0]
best_uk = uk_sorted[0]
combined = best_fr["hbc"] + best_uk["hbc"]

print(f"\n  Best FR: {best_fr['models']} ({best_fr['n']}m)  +HBC={best_fr['hbc']:.2f}")
print(f"  Best UK: {best_uk['models']} ({best_uk['n']}m)  +HBC={best_uk['hbc']:.2f}")
print(f"  Combined SUM: {combined:.2f}")

# Compare with same-ensemble approaches
print(f"\n  vs uniform ensemble choices:")
for fr_r in fr_sorted:
    for uk_r in uk_sorted:
        if fr_r["models"] == uk_r["models"]:
            s = fr_r["hbc"] + uk_r["hbc"]
            if s < combined + 0.5:
                print(f"    {fr_r['models']:30s}  FR={fr_r['hbc']:.2f}  UK={uk_r['hbc']:.2f}  SUM={s:.2f}")

# Top 10 mixed combos
print(f"\n  Top 10 mixed combos (FR best × UK best):")
mixed = []
for fr_r in fr_sorted[:10]:
    for uk_r in uk_sorted[:10]:
        mixed.append({
            "fr_models": fr_r["models"], "uk_models": uk_r["models"],
            "fr": fr_r["hbc"], "uk": uk_r["hbc"], "sum": fr_r["hbc"] + uk_r["hbc"],
        })
mixed_sorted = sorted(mixed, key=lambda x: x["sum"])
for i, m in enumerate(mixed_sorted[:10]):
    same = " (same)" if m["fr_models"] == m["uk_models"] else ""
    print(f"  {i+1:2d}. FR={m['fr']:.2f} [{m['fr_models']}]  "
          f"UK={m['uk']:.2f} [{m['uk_models']}]  SUM={m['sum']:.2f}{same}")

print(f"\n  Total time: {time.time() - t0:.0f}s")
