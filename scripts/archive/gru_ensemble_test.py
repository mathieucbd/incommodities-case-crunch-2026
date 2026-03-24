"""GRU ensemble test — Does GRU add value to the 5-model ensemble?

Tests:
  1. Error correlation between GRU and existing 5 models
  2. 5-model swap: Replace XGB with GRU → CB + LGB + GRU + EN + DNN
  3. 6-model: CB + LGB + XGB + EN + DNN + GRU

Best GRU configs from sweep:
  FR: seq12_h128  → RMSE+HBC=17.80
  UK: seq24_h128_head128 → RMSE+HBC=10.66
"""

import sys, yaml, warnings, time, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from catboost import CatBoostRegressor
import lightgbm as lgb
import xgboost as xgb
from scipy.optimize import minimize

sys.path.insert(0, ".")
from src.data_loading import load_data, merge_train
from src.feature_engineering import build_features

warnings.filterwarnings("ignore")

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"  Device: {DEVICE}")

with open("config.yaml") as f:
    config = yaml.safe_load(f)

print("=" * 90)
print("  GRU ENSEMBLE TEST — Does GRU add value?")
print("=" * 90)

t0 = time.time()
x_train, y_train, x_test = load_data("data/raw")
train = merge_train(x_train, y_train)
df = build_features(train, config)

holdout_start = config["validation"]["holdout_start"]
mask_val = (df["datetime_CET"] >= holdout_start).values
df_tr = df[~mask_val].copy()
df_va = df[mask_val].copy()

# ── Feature sets ──────────────────────────────────────────────────────
with open("outputs/feature_selection_v5_fr.json") as f:
    fs_v5 = json.load(f)
FR_TREE_FEAT = [f for f in fs_v5["features"] if f in df_tr.columns]

with open("outputs/uk_feature_research.json") as f:
    uk_research = json.load(f)
UK_TREE_FEAT = [f for f in uk_research["confirmed_features"] if f in df_tr.columns]

# DNN/GRU features (350 deduped)
EXCLUDE = {"fr_spot", "uk_spot", "datetime_CET", "datetime_UTC", "date", "id"}
ALL_NUM = [c for c in df_tr.columns
           if c not in EXCLUDE
           and df_tr[c].dtype in ["float64", "float32", "int64", "int32"]
           and df_tr[c].notna().sum() > len(df_tr) * 0.5]
corr_matrix = df_tr[ALL_NUM].corr().abs()
to_drop = set()
for i in range(len(ALL_NUM)):
    if ALL_NUM[i] in to_drop: continue
    for j in range(i + 1, len(ALL_NUM)):
        if ALL_NUM[j] in to_drop: continue
        if corr_matrix.iloc[i, j] > 0.99: to_drop.add(ALL_NUM[j])
DNN_FEAT = [f for f in ALL_NUM if f not in to_drop]
print(f"  DNN/GRU features: {len(DNN_FEAT)}")

# ── Targets ───────────────────────────────────────────────────────────
fr_spot_la_full = df["fr_spot_la"].values
ema_fr = pd.Series(fr_spot_la_full).ewm(span=240).mean().values
fr_anchor_va = ema_fr[mask_val]
fr_spot_va = df_va["fr_spot"].values
fr_y_tr = df_tr["fr_spot"].values - ema_fr[~mask_val]
fr_y_va = fr_spot_va - fr_anchor_va

uk_moc_tr = df_tr["uk_merit_order_cost"].values
uk_moc_va = df_va["uk_merit_order_cost"].values
uk_spot_va = df_va["uk_spot"].values
uk_y_tr = df_tr["uk_spot"].values - uk_moc_tr
uk_y_va = uk_spot_va - uk_moc_va

hours_va = df_va["hour"].values

fr_valid_tr = np.isfinite(fr_y_tr) & np.isfinite(ema_fr[~mask_val])
fr_valid_va = np.isfinite(fr_y_va) & np.isfinite(fr_anchor_va)
uk_valid_tr = np.isfinite(uk_y_tr)
uk_valid_va = np.isfinite(uk_y_va)


# ── Metrics ───────────────────────────────────────────────────────────
def compute_rmse(actual, preds):
    return np.sqrt(np.mean((actual - preds) ** 2))

def compute_hbc(preds, actual, hours):
    errors = actual - preds
    hbc = {h: float(errors[hours == h].mean()) for h in range(24) if (hours == h).sum() > 0}
    corrected = preds + np.array([hbc.get(h, 0) for h in hours])
    return hbc, np.sqrt(np.mean((actual - corrected) ** 2))


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

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ── GRU ───────────────────────────────────────────────────────────────
class ElecGRU(nn.Module):
    def __init__(self, n_features, hidden_size=128, num_layers=1,
                 bidirectional=False, dropout=0.2, head_size=64):
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_features, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        gru_out = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Linear(gru_out, head_size),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            nn.Linear(head_size, 1),
        )

    def forward(self, x):
        out, _ = self.gru(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


def build_sequences(X, y, valid_mask, seq_len):
    sequences, targets, valid_idx = [], [], []
    for i in range(seq_len, len(X)):
        if not valid_mask[i]:
            continue
        seq = X[i - seq_len:i]
        if np.any(np.isnan(seq)):
            continue
        sequences.append(seq)
        targets.append(y[i])
        valid_idx.append(i)
    return np.array(sequences), np.array(targets), np.array(valid_idx)


def train_nn(model, X_tr, y_tr, X_va, y_va, lr=1e-3, wd=1e-4, bs=256,
             max_epochs=500, patience=30):
    model = model.to(DEVICE)
    ds = TensorDataset(torch.FloatTensor(X_tr).to(DEVICE),
                       torch.FloatTensor(y_tr).to(DEVICE))
    loader = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=len(X_tr) % bs == 1)
    X_va_t = torch.FloatTensor(X_va).to(DEVICE)
    y_va_t = torch.FloatTensor(y_va).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10 if not isinstance(model, ElecGRU) else 8,
                                                        factor=0.5, min_lr=1e-6)
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


def predict_nn(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.FloatTensor(X).to(DEVICE)).cpu().numpy()


# ── Regime weights ────────────────────────────────────────────────────
REGIMES = {
    "night":   [0, 1, 2, 3, 4, 5],
    "morning": [6, 7, 8, 9],
    "day":     [10, 11, 12, 13, 14, 15, 16],
    "peak":    [17, 18, 19, 20, 21],
    "late":    [22, 23],
}


def optimize_regime_weights_grid(preds_dict, actual, hours):
    """Grid search for 4-5 models."""
    names = list(preds_dict.keys())
    n = len(names)
    p = preds_dict

    regime_weights = {}
    for regime, hours_list in REGIMES.items():
        mask = np.isin(hours, hours_list)
        if mask.sum() == 0: continue
        a = actual[mask]
        best_rmse, best_w = 999, {}

        if n == 5:
            for w1 in np.arange(0.0, 1.05, 0.1):
                for w2 in np.arange(0.0, 1.05 - w1, 0.1):
                    for w3 in np.arange(0.0, 1.05 - w1 - w2, 0.1):
                        for w4 in np.arange(0.0, 1.05 - w1 - w2 - w3, 0.1):
                            w5 = round(1.0 - w1 - w2 - w3 - w4, 1)
                            if w5 < -0.01: continue
                            e = sum(w * p[names[i]][mask] for i, w in enumerate([w1, w2, w3, w4, w5]))
                            rmse = np.sqrt(np.mean((a - e) ** 2))
                            if rmse < best_rmse:
                                best_rmse = rmse
                                best_w = dict(zip(names, [w1, w2, w3, w4, w5]))
        elif n == 4:
            for w1 in np.arange(0.0, 1.05, 0.1):
                for w2 in np.arange(0.0, 1.05 - w1, 0.1):
                    for w3 in np.arange(0.0, 1.05 - w1 - w2, 0.1):
                        w4 = round(1.0 - w1 - w2 - w3, 1)
                        if w4 < -0.01: continue
                        e = sum(w * p[names[i]][mask] for i, w in enumerate([w1, w2, w3, w4]))
                        rmse = np.sqrt(np.mean((a - e) ** 2))
                        if rmse < best_rmse:
                            best_rmse = rmse
                            best_w = dict(zip(names, [w1, w2, w3, w4]))

        regime_weights[regime] = {"weights": best_w, "rmse": round(best_rmse, 2)}
    return regime_weights


def optimize_regime_weights_scipy(preds_dict, actual, hours):
    """Scipy optimize for 6+ models."""
    names = list(preds_dict.keys())
    n = len(names)
    p_arrays = [preds_dict[name] for name in names]

    regime_weights = {}
    for regime, hours_list in REGIMES.items():
        mask = np.isin(hours, hours_list)
        if mask.sum() == 0: continue
        a = actual[mask]
        p_masked = [pa[mask] for pa in p_arrays]

        def objective(w):
            # Last weight = 1 - sum(others)
            w_full = np.append(w, 1.0 - np.sum(w))
            if w_full[-1] < -0.05:
                return 1e6
            pred = sum(w_full[i] * p_masked[i] for i in range(n))
            return np.sqrt(np.mean((a - pred) ** 2))

        # Multi-start
        best_result = None
        for _ in range(20):
            w0 = np.random.dirichlet(np.ones(n))[:-1]
            bounds = [(0, 1)] * (n - 1)
            res = minimize(objective, w0, method='L-BFGS-B', bounds=bounds)
            if best_result is None or res.fun < best_result.fun:
                best_result = res

        w_opt = np.append(best_result.x, 1.0 - np.sum(best_result.x))
        w_opt = np.maximum(w_opt, 0)
        w_opt /= w_opt.sum()
        best_w = {names[i]: round(float(w_opt[i]), 3) for i in range(n)}
        regime_weights[regime] = {"weights": best_w, "rmse": round(best_result.fun, 2)}

    return regime_weights


def apply_regime_ensemble(preds_dict, hours, regime_weights):
    result = np.zeros(len(hours))
    for regime, hours_list in REGIMES.items():
        mask = np.isin(hours, hours_list)
        if mask.sum() == 0: continue
        w = regime_weights[regime]["weights"]
        for name, weight in w.items():
            result[mask] += weight * preds_dict[name][mask]
    return result


# ══════════════════════════════════════════════════════════════════════
#  1. TRAIN ALL MODELS (5 existing + GRU)
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*90}")
print("  1. Training all models")
print(f"{'='*90}")

# Scale features for DNN/GRU
dnn_scaler = StandardScaler()
X_dnn_tr = dnn_scaler.fit_transform(np.nan_to_num(df_tr[DNN_FEAT].values, 0))
X_dnn_va = dnn_scaler.transform(np.nan_to_num(df_va[DNN_FEAT].values, 0))
n_feat = X_dnn_tr.shape[1]

# ── FR Models ─────────────────────────────────────────────────────────
print("\n  --- FR CatBoost ---")
cb_params_fr = config.get("catboost_params_fr_optuna_v2", config.get("catboost_params_fr", {}))
days_ago = (df_tr["datetime_CET"].max() - df_tr["datetime_CET"]).dt.total_seconds() / 86400
roll_std = df_tr["fr_spot_la"].rolling(168, min_periods=24).std().fillna(df_tr["fr_spot_la"].std())
sample_w = np.exp(-2 * days_ago.values / 365) / np.clip(roll_std.values ** 2, 1, None)
sample_w[~fr_valid_tr] = 0

cb_fr = CatBoostRegressor(**{**cb_params_fr, "verbose": 0})
cb_fr.fit(df_tr[FR_TREE_FEAT].values[fr_valid_tr], fr_y_tr[fr_valid_tr],
          sample_weight=sample_w[fr_valid_tr],
          eval_set=(df_va[FR_TREE_FEAT].values, fr_y_va))
preds_fr_cb = fr_anchor_va + cb_fr.predict(df_va[FR_TREE_FEAT].values)
_, rmse_fr_cb = compute_hbc(preds_fr_cb, fr_spot_va, hours_va)
print(f"    RMSE+HBC: {rmse_fr_cb:.2f}")

print("  --- FR LightGBM ---")
lgb_params_fr = config.get("lightgbm_params_fr", {})
lgb_params_fr_clean = {k: v for k, v in lgb_params_fr.items() if k != "n_estimators"}
ds_tr_lgb = lgb.Dataset(df_tr[FR_TREE_FEAT].values[fr_valid_tr], fr_y_tr[fr_valid_tr],
                        weight=sample_w[fr_valid_tr])
ds_va_lgb = lgb.Dataset(df_va[FR_TREE_FEAT].values, fr_y_va, reference=ds_tr_lgb)
lgb_fr = lgb.train(lgb_params_fr_clean, ds_tr_lgb,
                    num_boost_round=lgb_params_fr.get("n_estimators", 5000),
                    valid_sets=[ds_va_lgb], callbacks=[lgb.early_stopping(50, verbose=False)])
preds_fr_lgb = fr_anchor_va + lgb_fr.predict(df_va[FR_TREE_FEAT].values)
_, rmse_fr_lgb = compute_hbc(preds_fr_lgb, fr_spot_va, hours_va)
print(f"    RMSE+HBC: {rmse_fr_lgb:.2f}")

print("  --- FR XGBoost ---")
xgb_params_fr = config.get("xgboost_params_fr", {})
xgb_params_fr_clean = {k: v for k, v in xgb_params_fr.items() if k != "n_estimators"}
xgb_params_fr_clean["device"] = "cpu"
dtrain_fr = xgb.DMatrix(df_tr[FR_TREE_FEAT].values[fr_valid_tr], fr_y_tr[fr_valid_tr],
                        weight=sample_w[fr_valid_tr])
dval_fr = xgb.DMatrix(df_va[FR_TREE_FEAT].values, fr_y_va)
xgb_fr = xgb.train(xgb_params_fr_clean, dtrain_fr,
                    num_boost_round=xgb_params_fr.get("n_estimators", 15000),
                    evals=[(dval_fr, "val")], early_stopping_rounds=50, verbose_eval=False)
preds_fr_xgb = fr_anchor_va + xgb_fr.predict(dval_fr)
_, rmse_fr_xgb = compute_hbc(preds_fr_xgb, fr_spot_va, hours_va)
print(f"    RMSE+HBC: {rmse_fr_xgb:.2f}")

print("  --- FR Elastic Net ---")
fr_scaler_en = StandardScaler()
X_fr_en_tr = fr_scaler_en.fit_transform(np.nan_to_num(df_tr[FR_TREE_FEAT].values[fr_valid_tr], 0))
X_fr_en_va = fr_scaler_en.transform(np.nan_to_num(df_va[FR_TREE_FEAT].values, 0))
en_fr = ElasticNet(alpha=10.0, l1_ratio=0.9, max_iter=10000)
en_fr.fit(X_fr_en_tr, fr_y_tr[fr_valid_tr])
preds_fr_en = fr_anchor_va + en_fr.predict(X_fr_en_va)
_, rmse_fr_en = compute_hbc(preds_fr_en, fr_spot_va, hours_va)
print(f"    RMSE+HBC: {rmse_fr_en:.2f}")

print("  --- FR DNN [192, 96] ---")
torch.manual_seed(42); np.random.seed(42)
dnn_fr = ElecDNN(len(DNN_FEAT), [192, 96], dropout=0.2)
dnn_fr, ep_fr = train_nn(dnn_fr, X_dnn_tr[fr_valid_tr], fr_y_tr[fr_valid_tr].astype(np.float32),
                          X_dnn_va[fr_valid_va], fr_y_va[fr_valid_va].astype(np.float32))
preds_fr_dnn = fr_anchor_va + predict_nn(dnn_fr, X_dnn_va)
_, rmse_fr_dnn = compute_hbc(preds_fr_dnn, fr_spot_va, hours_va)
print(f"    RMSE+HBC: {rmse_fr_dnn:.2f}  (ep={ep_fr})")

print("  --- FR GRU seq12_h128 ---")
# Build sequences for FR GRU (best config: seq_len=12, hidden=128)
FR_GRU_SEQ = 12
seq_fr_tr, tgt_fr_tr, idx_fr_tr = build_sequences(X_dnn_tr, fr_y_tr, fr_valid_tr, FR_GRU_SEQ)
seq_fr_va, tgt_fr_va, idx_fr_va = build_sequences(X_dnn_va, fr_y_va, fr_valid_va, FR_GRU_SEQ)
print(f"    Sequences: tr={len(seq_fr_tr)}, va={len(seq_fr_va)}")

torch.manual_seed(42); np.random.seed(42)
gru_fr = ElecGRU(n_feat, hidden_size=128, num_layers=1, bidirectional=False, dropout=0.2, head_size=64)
gru_fr, ep_gru_fr = train_nn(gru_fr, seq_fr_tr, tgt_fr_tr.astype(np.float32),
                               seq_fr_va, tgt_fr_va.astype(np.float32),
                               patience=25, max_epochs=300)
# GRU only has predictions for idx_fr_va positions
# For ensemble, we need full coverage. Fallback to DNN for positions without GRU.
preds_fr_gru_dev = np.full(len(X_dnn_va), np.nan)
gru_fr_preds_seq = predict_nn(gru_fr, seq_fr_va)
preds_fr_gru_dev[idx_fr_va] = gru_fr_preds_seq
# Fill gaps with DNN predictions
dnn_fr_dev = predict_nn(dnn_fr, X_dnn_va)
gru_gaps = np.isnan(preds_fr_gru_dev)
preds_fr_gru_dev[gru_gaps] = dnn_fr_dev[gru_gaps]
preds_fr_gru = fr_anchor_va + preds_fr_gru_dev
_, rmse_fr_gru = compute_hbc(preds_fr_gru, fr_spot_va, hours_va)
print(f"    RMSE+HBC: {rmse_fr_gru:.2f}  (ep={ep_gru_fr}, gaps={gru_gaps.sum()})")

del dnn_fr
if DEVICE.type == "mps": torch.mps.empty_cache()

# ── UK Models ─────────────────────────────────────────────────────────
print("\n  --- UK CatBoost ---")
cb_params_uk = config.get("catboost_params_uk", {})
cb_uk = CatBoostRegressor(**{**cb_params_uk, "verbose": 0})
cb_uk.fit(df_tr[UK_TREE_FEAT].values[uk_valid_tr], uk_y_tr[uk_valid_tr],
          eval_set=(df_va[UK_TREE_FEAT].values, uk_y_va))
preds_uk_cb = uk_moc_va + cb_uk.predict(df_va[UK_TREE_FEAT].values)
_, rmse_uk_cb = compute_hbc(preds_uk_cb, uk_spot_va, hours_va)
print(f"    RMSE+HBC: {rmse_uk_cb:.2f}")

print("  --- UK LightGBM ---")
lgb_params_uk = config.get("lightgbm_params_uk", {})
lgb_params_uk_clean = {k: v for k, v in lgb_params_uk.items() if k != "n_estimators"}
ds_tr_uk = lgb.Dataset(df_tr[UK_TREE_FEAT].values[uk_valid_tr], uk_y_tr[uk_valid_tr])
ds_va_uk = lgb.Dataset(df_va[UK_TREE_FEAT].values, uk_y_va, reference=ds_tr_uk)
lgb_uk = lgb.train(lgb_params_uk_clean, ds_tr_uk,
                   num_boost_round=lgb_params_uk.get("n_estimators", 5000),
                   valid_sets=[ds_va_uk], callbacks=[lgb.early_stopping(50, verbose=False)])
preds_uk_lgb = uk_moc_va + lgb_uk.predict(df_va[UK_TREE_FEAT].values)
_, rmse_uk_lgb = compute_hbc(preds_uk_lgb, uk_spot_va, hours_va)
print(f"    RMSE+HBC: {rmse_uk_lgb:.2f}")

print("  --- UK XGBoost ---")
xgb_params_uk = config.get("xgboost_params_uk", {})
xgb_params_uk_clean = {k: v for k, v in xgb_params_uk.items() if k != "n_estimators"}
xgb_params_uk_clean["device"] = "cpu"
dtrain_uk = xgb.DMatrix(df_tr[UK_TREE_FEAT].values[uk_valid_tr], uk_y_tr[uk_valid_tr])
dval_uk = xgb.DMatrix(df_va[UK_TREE_FEAT].values, uk_y_va)
xgb_uk = xgb.train(xgb_params_uk_clean, dtrain_uk,
                    num_boost_round=xgb_params_uk.get("n_estimators", 15000),
                    evals=[(dval_uk, "val")], early_stopping_rounds=50, verbose_eval=False)
preds_uk_xgb = uk_moc_va + xgb_uk.predict(dval_uk)
_, rmse_uk_xgb = compute_hbc(preds_uk_xgb, uk_spot_va, hours_va)
print(f"    RMSE+HBC: {rmse_uk_xgb:.2f}")

print("  --- UK Elastic Net ---")
uk_scaler_en = StandardScaler()
X_uk_en_tr = uk_scaler_en.fit_transform(np.nan_to_num(df_tr[UK_TREE_FEAT].values[uk_valid_tr], 0))
X_uk_en_va = uk_scaler_en.transform(np.nan_to_num(df_va[UK_TREE_FEAT].values, 0))
en_uk = ElasticNet(alpha=1.0, l1_ratio=0.9, max_iter=10000)
en_uk.fit(X_uk_en_tr, uk_y_tr[uk_valid_tr])
preds_uk_en = uk_moc_va + en_uk.predict(X_uk_en_va)
_, rmse_uk_en = compute_hbc(preds_uk_en, uk_spot_va, hours_va)
print(f"    RMSE+HBC: {rmse_uk_en:.2f}")

print("  --- UK DNN [768, 384, 192] ---")
torch.manual_seed(42); np.random.seed(42)
dnn_uk = ElecDNN(len(DNN_FEAT), [768, 384, 192], dropout=0.3)
dnn_uk, ep_uk = train_nn(dnn_uk, X_dnn_tr[uk_valid_tr], uk_y_tr[uk_valid_tr].astype(np.float32),
                          X_dnn_va[uk_valid_va], uk_y_va[uk_valid_va].astype(np.float32))
preds_uk_dnn = uk_moc_va + predict_nn(dnn_uk, X_dnn_va)
_, rmse_uk_dnn = compute_hbc(preds_uk_dnn, uk_spot_va, hours_va)
print(f"    RMSE+HBC: {rmse_uk_dnn:.2f}  (ep={ep_uk})")

print("  --- UK GRU seq24_h128_head128 ---")
UK_GRU_SEQ = 24
seq_uk_tr, tgt_uk_tr, idx_uk_tr = build_sequences(X_dnn_tr, uk_y_tr, uk_valid_tr, UK_GRU_SEQ)
seq_uk_va, tgt_uk_va, idx_uk_va = build_sequences(X_dnn_va, uk_y_va, uk_valid_va, UK_GRU_SEQ)
print(f"    Sequences: tr={len(seq_uk_tr)}, va={len(seq_uk_va)}")

torch.manual_seed(42); np.random.seed(42)
gru_uk = ElecGRU(n_feat, hidden_size=128, num_layers=1, bidirectional=False, dropout=0.2, head_size=128)
gru_uk, ep_gru_uk = train_nn(gru_uk, seq_uk_tr, tgt_uk_tr.astype(np.float32),
                               seq_uk_va, tgt_uk_va.astype(np.float32),
                               patience=25, max_epochs=300)
preds_uk_gru_dev = np.full(len(X_dnn_va), np.nan)
gru_uk_preds_seq = predict_nn(gru_uk, seq_uk_va)
preds_uk_gru_dev[idx_uk_va] = gru_uk_preds_seq
dnn_uk_dev = predict_nn(dnn_uk, X_dnn_va)
gru_gaps_uk = np.isnan(preds_uk_gru_dev)
preds_uk_gru_dev[gru_gaps_uk] = dnn_uk_dev[gru_gaps_uk]
preds_uk_gru = uk_moc_va + preds_uk_gru_dev
_, rmse_uk_gru = compute_hbc(preds_uk_gru, uk_spot_va, hours_va)
print(f"    RMSE+HBC: {rmse_uk_gru:.2f}  (ep={ep_gru_uk}, gaps={gru_gaps_uk.sum()})")

del dnn_uk, gru_fr, gru_uk
if DEVICE.type == "mps": torch.mps.empty_cache()


# ══════════════════════════════════════════════════════════════════════
#  2. ERROR CORRELATION ANALYSIS
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*90}")
print("  2. Error correlation analysis (including GRU)")
print(f"{'='*90}")

for market, preds, spot_va_m in [
    ("FR", {"CB": preds_fr_cb, "LGB": preds_fr_lgb, "XGB": preds_fr_xgb,
            "EN": preds_fr_en, "DNN": preds_fr_dnn, "GRU": preds_fr_gru}, fr_spot_va),
    ("UK", {"CB": preds_uk_cb, "LGB": preds_uk_lgb, "XGB": preds_uk_xgb,
            "EN": preds_uk_en, "DNN": preds_uk_dnn, "GRU": preds_uk_gru}, uk_spot_va),
]:
    errors = {name: spot_va_m - p for name, p in preds.items()}
    names = list(errors.keys())
    print(f"\n  {market} error correlations:")
    header = "         " + "  ".join(f"{n:>6s}" for n in names)
    print(header)
    for n1 in names:
        row = f"    {n1:>4s} "
        for n2 in names:
            if n1 == n2:
                row += "     - "
            else:
                c = np.corrcoef(errors[n1], errors[n2])[0, 1]
                row += f"  {c:.3f}"
        print(row)


# ══════════════════════════════════════════════════════════════════════
#  3. ENSEMBLE COMPARISON
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*90}")
print("  3. Ensemble comparison")
print(f"{'='*90}")

for market, actual, hrs, preds_all in [
    ("FR", fr_spot_va, hours_va, {
        "CB": preds_fr_cb, "LGB": preds_fr_lgb, "XGB": preds_fr_xgb,
        "EN": preds_fr_en, "DNN": preds_fr_dnn, "GRU": preds_fr_gru}),
    ("UK", uk_spot_va, hours_va, {
        "CB": preds_uk_cb, "LGB": preds_uk_lgb, "XGB": preds_uk_xgb,
        "EN": preds_uk_en, "DNN": preds_uk_dnn, "GRU": preds_uk_gru}),
]:
    print(f"\n  {market}:")

    configs = {
        "5-model (current)":  {k: v for k, v in preds_all.items() if k != "GRU"},
        "5-model (GRU swap)": {k: v for k, v in preds_all.items() if k != "XGB"},
        "6-model (all)":      preds_all,
    }

    for label, preds_dict in configs.items():
        n = len(preds_dict)
        if n <= 5:
            rw = optimize_regime_weights_grid(preds_dict, actual, hrs)
        else:
            rw = optimize_regime_weights_scipy(preds_dict, actual, hrs)
        ens = apply_regime_ensemble(preds_dict, hrs, rw)
        _, rmse_hbc = compute_hbc(ens, actual, hrs)
        print(f"    {label:25s}  +HBC={rmse_hbc:.2f}")
        for r, info in rw.items():
            w_str = " / ".join(f"{k}={v}" for k, v in info["weights"].items())
            print(f"      {r:10s}: {w_str}  RMSE={info['rmse']}")


# ── Final summary ────────────────────────────────────────────────────
print(f"\n{'='*90}")
print("  FINAL SUMMARY")
print(f"{'='*90}")

results = {}
for label_key, combo in [
    ("5-model (current)", ["CB", "LGB", "XGB", "EN", "DNN"]),
    ("5-model (GRU swap)", ["CB", "LGB", "GRU", "EN", "DNN"]),
    ("6-model (all)", ["CB", "LGB", "XGB", "EN", "DNN", "GRU"]),
]:
    fr_preds = {k: {"CB": preds_fr_cb, "LGB": preds_fr_lgb, "XGB": preds_fr_xgb,
                     "EN": preds_fr_en, "DNN": preds_fr_dnn, "GRU": preds_fr_gru}[k] for k in combo}
    uk_preds = {k: {"CB": preds_uk_cb, "LGB": preds_uk_lgb, "XGB": preds_uk_xgb,
                     "EN": preds_uk_en, "DNN": preds_uk_dnn, "GRU": preds_uk_gru}[k] for k in combo}

    n = len(combo)
    if n <= 5:
        rw_fr = optimize_regime_weights_grid(fr_preds, fr_spot_va, hours_va)
        rw_uk = optimize_regime_weights_grid(uk_preds, uk_spot_va, hours_va)
    else:
        rw_fr = optimize_regime_weights_scipy(fr_preds, fr_spot_va, hours_va)
        rw_uk = optimize_regime_weights_scipy(uk_preds, uk_spot_va, hours_va)

    ens_fr = apply_regime_ensemble(fr_preds, hours_va, rw_fr)
    ens_uk = apply_regime_ensemble(uk_preds, hours_va, rw_uk)
    _, r_fr = compute_hbc(ens_fr, fr_spot_va, hours_va)
    _, r_uk = compute_hbc(ens_uk, uk_spot_va, hours_va)
    s = r_fr + r_uk
    results[label_key] = {"FR": r_fr, "UK": r_uk, "SUM": s}
    print(f"  {label_key:25s}  FR={r_fr:.2f}  UK={r_uk:.2f}  SUM={s:.2f}")

best_label = min(results, key=lambda k: results[k]["SUM"])
print(f"\n  BEST: {best_label}  SUM={results[best_label]['SUM']:.2f}")

print(f"\n  Total time: {time.time() - t0:.0f}s")
