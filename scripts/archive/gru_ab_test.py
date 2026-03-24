"""GRU A/B test — Sequence model for electricity price forecasting.

Key idea: All our current models treat each hour independently.
A GRU sees sequences of 24-48h and captures temporal dynamics:
  - "prices rising for 6h" → momentum
  - "just transitioned from night to peak" → regime shifts
  - Autocorrelation patterns missed by static features

Tests:
  - Sequence lengths: 12, 24, 48h
  - GRU sizes: 64, 128, 256 hidden
  - Layers: 1, 2
  - Bidirectional vs unidirectional
  - With/without feature head (GRU output → small Dense → prediction)
"""

import sys, yaml, warnings, time, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, ".")
from src.data_loading import load_data, merge_train
from src.feature_engineering import build_features

warnings.filterwarnings("ignore")

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"  Device: {DEVICE}")

with open("config.yaml") as f:
    config = yaml.safe_load(f)

print("=" * 90)
print("  GRU A/B TEST — Temporal sequence model")
print("=" * 90)

t0 = time.time()
x_train, y_train, x_test = load_data("data/raw")
train = merge_train(x_train, y_train)
df = build_features(train, config)

holdout_start = config["validation"]["holdout_start"]
mask_val = (df["datetime_CET"] >= holdout_start).values
df_tr = df[~mask_val].copy()
df_va = df[mask_val].copy()

# DNN features (349 deduped)
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
FEATURES = [f for f in _all if f not in _drop]
print(f"  Features: {len(FEATURES)}")

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


# ── Sequence builder ──────────────────────────────────────────────────
def build_sequences(X, y, valid_mask, seq_len):
    """Build (seq_len, n_features) sequences. Target = last timestep."""
    sequences, targets, valid_idx = [], [], []
    for i in range(seq_len, len(X)):
        if not valid_mask[i]:
            continue
        # Check all timesteps in window are present (no gaps)
        seq = X[i - seq_len:i]
        if np.any(np.isnan(seq)):
            continue
        sequences.append(seq)
        targets.append(y[i])
        valid_idx.append(i)
    return np.array(sequences), np.array(targets), np.array(valid_idx)


# ── GRU Model ─────────────────────────────────────────────────────────
class ElecGRU(nn.Module):
    def __init__(self, n_features, hidden_size=128, num_layers=1,
                 bidirectional=False, dropout=0.2, head_size=64):
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
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
        # x: (batch, seq_len, n_features)
        out, _ = self.gru(x)
        # Take last timestep output
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


def train_gru(model, X_tr, y_tr, X_va, y_va, lr=1e-3, wd=1e-4, bs=256,
              max_epochs=300, patience=25):
    model = model.to(DEVICE)
    ds = TensorDataset(torch.FloatTensor(X_tr).to(DEVICE),
                       torch.FloatTensor(y_tr).to(DEVICE))
    loader = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=len(X_tr) % bs == 1)
    Xv = torch.FloatTensor(X_va).to(DEVICE)
    yv = torch.FloatTensor(y_va).to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=8, factor=0.5, min_lr=1e-6)
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
            vl = criterion(model(Xv), yv).item()
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


def predict_gru(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.FloatTensor(X).to(DEVICE)).cpu().numpy()


def compute_rmse(a, p):
    return np.sqrt(np.mean((a - p) ** 2))

def compute_hbc(preds, actual, hours):
    errors = actual - preds
    hbc = {h: errors[hours == h].mean() for h in range(24) if (hours == h).sum() > 0}
    corrected = preds + np.array([hbc.get(h, 0) for h in hours])
    return np.sqrt(np.mean((actual - corrected) ** 2))


# ── Scale features ────────────────────────────────────────────────────
feat = [f for f in FEATURES if f in df_tr.columns]
scaler = StandardScaler()
X_tr_flat = scaler.fit_transform(np.nan_to_num(df_tr[feat].values, 0))
X_va_flat = scaler.transform(np.nan_to_num(df_va[feat].values, 0))
n_feat = X_tr_flat.shape[1]
print(f"  Scaled features: {n_feat}")

# ── Configs ───────────────────────────────────────────────────────────
CONFIGS = [
    # Sequence length sweep
    {"name": "seq12_h128",   "seq": 12, "hidden": 128, "layers": 1, "bidir": False, "dp": 0.2, "head": 64},
    {"name": "seq24_h128",   "seq": 24, "hidden": 128, "layers": 1, "bidir": False, "dp": 0.2, "head": 64},
    {"name": "seq48_h128",   "seq": 48, "hidden": 128, "layers": 1, "bidir": False, "dp": 0.2, "head": 64},
    # Hidden size sweep (seq=24)
    {"name": "seq24_h64",    "seq": 24, "hidden": 64,  "layers": 1, "bidir": False, "dp": 0.2, "head": 32},
    {"name": "seq24_h256",   "seq": 24, "hidden": 256, "layers": 1, "bidir": False, "dp": 0.2, "head": 128},
    # 2-layer GRU
    {"name": "seq24_h128_2L","seq": 24, "hidden": 128, "layers": 2, "bidir": False, "dp": 0.2, "head": 64},
    {"name": "seq24_h256_2L","seq": 24, "hidden": 256, "layers": 2, "bidir": False, "dp": 0.2, "head": 128},
    # Bidirectional
    {"name": "seq24_h128_bi","seq": 24, "hidden": 128, "layers": 1, "bidir": True,  "dp": 0.2, "head": 64},
    {"name": "seq48_h128_bi","seq": 48, "hidden": 128, "layers": 1, "bidir": True,  "dp": 0.2, "head": 64},
    # Dropout sweep
    {"name": "seq24_h128_dp1","seq": 24,"hidden": 128, "layers": 1, "bidir": False, "dp": 0.1, "head": 64},
    {"name": "seq24_h128_dp3","seq": 24,"hidden": 128, "layers": 1, "bidir": False, "dp": 0.3, "head": 64},
    # Larger head
    {"name": "seq24_h128_head128","seq": 24, "hidden": 128, "layers": 1, "bidir": False, "dp": 0.2, "head": 128},
]

# ── Run sweep ─────────────────────────────────────────────────────────
results_all = []

for market, y_tr, y_va, vt, vv, anchor_va, spot_va in [
    ("FR", fr_y_tr, fr_y_va, fr_vt, fr_vv, fr_rm_va, fr_spot_va),
    ("UK", uk_y_tr, uk_y_va, uk_vt, uk_vv, uk_moc_va, uk_spot_va),
]:
    print(f"\n{'='*90}")
    print(f"  {market} — GRU SWEEP")
    print(f"{'='*90}")

    best = {"rmse_hbc": 999}

    for cfg in CONFIGS:
        t1 = time.time()
        seq_len = cfg["seq"]

        # Build sequences for train and val
        seq_tr, tgt_tr, idx_tr = build_sequences(X_tr_flat, y_tr, vt, seq_len)
        seq_va, tgt_va, idx_va = build_sequences(X_va_flat, y_va, vv, seq_len)

        if len(seq_tr) < 100 or len(seq_va) < 100:
            print(f"    {cfg['name']:25s}  SKIP (too few sequences: tr={len(seq_tr)}, va={len(seq_va)})")
            continue

        torch.manual_seed(42); np.random.seed(42)

        model = ElecGRU(
            n_features=n_feat,
            hidden_size=cfg["hidden"],
            num_layers=cfg["layers"],
            bidirectional=cfg["bidir"],
            dropout=cfg["dp"],
            head_size=cfg["head"],
        )

        model, epochs = train_gru(model, seq_tr, tgt_tr.astype(np.float32),
                                   seq_va, tgt_va.astype(np.float32),
                                   lr=1e-3, wd=1e-4, bs=256)

        preds_dev = predict_gru(model, seq_va)
        preds_spot = anchor_va[idx_va] + preds_dev
        actual = spot_va[idx_va]
        hrs = hours_va[idx_va]
        rmse = compute_rmse(actual, preds_spot)
        rmse_hbc = compute_hbc(preds_spot, actual, hrs)
        n_params = sum(p.numel() for p in model.parameters())

        elapsed = time.time() - t1
        result = {
            "market": market, "config": cfg["name"], "rmse": round(rmse, 2),
            "rmse_hbc": round(rmse_hbc, 2), "epochs": epochs,
            "params": n_params, "time": round(elapsed, 1),
            "n_train": len(seq_tr), "n_val": len(seq_va),
        }
        results_all.append(result)

        flag = " ***" if rmse_hbc < best["rmse_hbc"] else ""
        print(f"    {cfg['name']:25s}  RMSE={rmse:.2f}  +HBC={rmse_hbc:.2f}  "
              f"ep={epochs:3d}  params={n_params:>8,}  tr={len(seq_tr)}  {elapsed:.1f}s{flag}")

        if rmse_hbc < best["rmse_hbc"]:
            best = result.copy()

        del model
        if DEVICE.type == "mps":
            torch.mps.empty_cache()

    print(f"\n  {market} BEST: {best['config']}  RMSE={best.get('rmse','?')}  +HBC={best['rmse_hbc']:.2f}")


# ── Summary ───────────────────────────────────────────────────────────
print(f"\n{'='*90}")
print(f"  FINAL SUMMARY")
print(f"{'='*90}")

for market in ["FR", "UK"]:
    mr = [r for r in results_all if r["market"] == market]
    top = sorted(mr, key=lambda x: x["rmse_hbc"])[:8]
    print(f"\n  {market} top 8:")
    for r in top:
        print(f"    {r['config']:25s}  RMSE={r['rmse']}  +HBC={r['rmse_hbc']}  "
              f"ep={r['epochs']}  params={r['params']:,}")

with open("outputs/gru_ab_test.json", "w") as f:
    json.dump(results_all, f, indent=2)

print(f"\n  Total time: {time.time() - t0:.0f}s")
