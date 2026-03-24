"""GAT-GRU A/B test — Graph Attention + Temporal model.

Architecture: Features are grouped into "nodes" (market groups).
GAT learns attention between feature groups at each timestep.
GRU captures temporal dynamics on the GAT-enriched representations.

Feature groups (nodes):
  - FR price signals (spot_la, lags, rolling stats)
  - UK price signals
  - FR fundamentals (nuclear, hydro, solar, wind, demand)
  - UK fundamentals
  - Commodity (gas, coal, carbon, oil)
  - Weather (temperature, wind speed)
  - Calendar (hour, dow, month, holidays)
  - Cross-market (interconnectors, spreads)

Each node = aggregated features of that group.
GAT learns: "for predicting FR price, attend to gas and nuclear more than UK weather"
"""

import sys, yaml, warnings, time, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
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
print("  GAT-GRU A/B TEST — Graph Attention + Temporal")
print("=" * 90)

t0 = time.time()
x_train, y_train, x_test = load_data("data/raw")
train = merge_train(x_train, y_train)
df = build_features(train, config)

holdout_start = config["validation"]["holdout_start"]
mask_val = (df["datetime_CET"] >= holdout_start).values
df_tr = df[~mask_val].copy()
df_va = df[mask_val].copy()

# ── Feature groups (nodes in graph) ───────────────────────────────────
EXCLUDE = {"fr_spot", "uk_spot", "datetime_CET", "datetime_UTC", "date", "id"}
all_cols = [c for c in df_tr.columns if c not in EXCLUDE
            and df_tr[c].dtype in ["float64", "float32", "int64", "int32"]
            and df_tr[c].notna().sum() > len(df_tr) * 0.5]

def classify_feature(name):
    n = name.lower()
    if any(k in n for k in ["hour", "dow", "month", "weekend", "holiday", "day_of", "is_"]):
        return "calendar"
    if any(k in n for k in ["fr_spot", "fr_price", "fr_da_"]) and "uk" not in n:
        return "fr_price"
    if any(k in n for k in ["uk_spot", "uk_price", "uk_da_", "uk_n2ex"]) and "fr" not in n:
        return "uk_price"
    if any(k in n for k in ["nuclear", "hydro", "solar", "wind", "thermal", "biomass",
                             "demand", "load", "capacity", "generation", "scarcity"]):
        if "fr" in n: return "fr_fund"
        if "uk" in n: return "uk_fund"
        return "fr_fund"  # default
    if any(k in n for k in ["gas", "coal", "carbon", "co2", "oil", "brent", "ttf", "eua"]):
        return "commodity"
    if any(k in n for k in ["temp", "weather", "wind_speed", "irradiance", "cloud"]):
        return "weather"
    if any(k in n for k in ["interconn", "spread", "cross", "flow", "ifa", "eleclink"]):
        return "cross_market"
    if "merit" in n or "moc" in n or "spark" in n or "margin" in n:
        return "commodity"
    if "fr" in n:
        return "fr_price"
    if "uk" in n:
        return "uk_price"
    return "other"

groups = {}
for col in all_cols:
    g = classify_feature(col)
    groups.setdefault(g, []).append(col)

# Remove tiny groups, merge "other" into largest
if "other" in groups and len(groups["other"]) < 5:
    groups.setdefault("commodity", []).extend(groups.pop("other", []))
elif "other" in groups:
    pass  # keep as separate node

GROUP_NAMES = sorted(groups.keys())
print(f"  Feature groups ({len(GROUP_NAMES)} nodes):")
for g in GROUP_NAMES:
    print(f"    {g:15s}: {len(groups[g])} features")

# ── Targets ───────────────────────────────────────────────────────────
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


# ── Scale each group independently ───────────────────────────────────
group_scalers = {}
group_data_tr = {}
group_data_va = {}

for g in GROUP_NAMES:
    cols = groups[g]
    sc = StandardScaler()
    group_data_tr[g] = sc.fit_transform(np.nan_to_num(df_tr[cols].values, 0))
    group_data_va[g] = sc.transform(np.nan_to_num(df_va[cols].values, 0))
    group_scalers[g] = sc


# ── Sequence builder ──────────────────────────────────────────────────
def build_sequences_grouped(group_data, y, valid_mask, seq_len, group_names):
    """Build sequences: (n_samples, seq_len, n_groups, max_group_size)."""
    max_size = max(group_data[g].shape[1] for g in group_names)
    n_groups = len(group_names)
    sequences = []
    targets = []
    valid_idx = []

    for i in range(seq_len, len(y)):
        if not valid_mask[i]:
            continue
        # Build grouped sequence
        seq = np.zeros((seq_len, n_groups, max_size))
        for gi, g in enumerate(group_names):
            data = group_data[g]
            gs = data.shape[1]
            seq[:, gi, :gs] = data[i - seq_len:i]
        sequences.append(seq)
        targets.append(y[i])
        valid_idx.append(i)

    return np.array(sequences), np.array(targets), np.array(valid_idx)


# ── GAT Layer ─────────────────────────────────────────────────────────
class GraphAttentionLayer(nn.Module):
    """Multi-head attention between feature group nodes."""
    def __init__(self, in_dim, out_dim, n_heads=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = out_dim // n_heads
        assert out_dim % n_heads == 0

        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a_src = nn.Parameter(torch.randn(n_heads, self.head_dim))
        self.a_dst = nn.Parameter(torch.randn(n_heads, self.head_dim))
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        # x: (batch, n_nodes, in_dim)
        B, N, _ = x.shape
        h = self.W(x)  # (B, N, out_dim)
        h = h.view(B, N, self.n_heads, self.head_dim)  # (B, N, H, D)

        # Attention scores
        src_score = (h * self.a_src).sum(-1)  # (B, N, H)
        dst_score = (h * self.a_dst).sum(-1)  # (B, N, H)

        # Pairwise attention: e_ij = LeakyReLU(a_src*h_i + a_dst*h_j)
        attn = src_score.unsqueeze(2) + dst_score.unsqueeze(1)  # (B, N, N, H)
        attn = self.leaky_relu(attn)
        attn = F.softmax(attn, dim=2)  # normalize over source nodes
        attn = self.dropout(attn)

        # Aggregate: h'_i = sum_j attn_ij * h_j
        h_perm = h.permute(0, 2, 1, 3)  # (B, H, N, D)
        attn_perm = attn.permute(0, 3, 1, 2)  # (B, H, N, N)
        out = torch.matmul(attn_perm, h_perm)  # (B, H, N, D)
        out = out.permute(0, 2, 1, 3).reshape(B, N, -1)  # (B, N, out_dim)

        return out


class GATGRU(nn.Module):
    """GAT-GRU: Graph attention on feature groups + GRU for temporal."""
    def __init__(self, n_groups, max_group_size, gat_dim=64, n_heads=4,
                 gru_hidden=128, gru_layers=1, dropout=0.2, head_size=64):
        super().__init__()
        # Project each group to same dimension
        self.group_proj = nn.Linear(max_group_size, gat_dim)

        # GAT layer
        self.gat = GraphAttentionLayer(gat_dim, gat_dim, n_heads=n_heads, dropout=dropout)
        self.gat_norm = nn.LayerNorm(gat_dim)

        # Flatten GAT output for GRU input
        gru_input_size = n_groups * gat_dim

        # GRU
        self.gru = nn.GRU(
            input_size=gru_input_size,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0,
        )

        # Prediction head
        self.head = nn.Sequential(
            nn.Linear(gru_hidden, head_size),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            nn.Linear(head_size, 1),
        )

    def forward(self, x):
        # x: (batch, seq_len, n_groups, max_group_size)
        B, T, G, F = x.shape

        # Project each group
        x_proj = self.group_proj(x.reshape(B * T, G, F))  # (B*T, G, gat_dim)

        # Apply GAT
        x_gat = self.gat(x_proj)
        x_gat = self.gat_norm(x_gat + x_proj)  # residual + norm

        # Flatten groups for GRU
        x_flat = x_gat.reshape(B, T, -1)  # (B, T, G * gat_dim)

        # GRU
        gru_out, _ = self.gru(x_flat)
        last = gru_out[:, -1, :]

        return self.head(last).squeeze(-1)


def train_model(model, X_tr, y_tr, X_va, y_va, lr=1e-3, wd=1e-4, bs=256,
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


def predict_model(model, X):
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


# ── Configs ───────────────────────────────────────────────────────────
max_group_size = max(len(groups[g]) for g in GROUP_NAMES)
n_groups = len(GROUP_NAMES)
print(f"\n  Graph: {n_groups} nodes, max group size: {max_group_size}")

CONFIGS = [
    # Baseline GAT-GRU
    {"name": "s24_gat64_gru128",  "seq": 24, "gat_dim": 64,  "heads": 4, "gru_h": 128, "gru_l": 1, "dp": 0.2, "head": 64},
    {"name": "s24_gat32_gru128",  "seq": 24, "gat_dim": 32,  "heads": 4, "gru_h": 128, "gru_l": 1, "dp": 0.2, "head": 64},
    {"name": "s24_gat64_gru256",  "seq": 24, "gat_dim": 64,  "heads": 4, "gru_h": 256, "gru_l": 1, "dp": 0.2, "head": 128},
    # Sequence length
    {"name": "s12_gat64_gru128",  "seq": 12, "gat_dim": 64,  "heads": 4, "gru_h": 128, "gru_l": 1, "dp": 0.2, "head": 64},
    {"name": "s48_gat64_gru128",  "seq": 48, "gat_dim": 64,  "heads": 4, "gru_h": 128, "gru_l": 1, "dp": 0.2, "head": 64},
    # More heads
    {"name": "s24_gat64_h8_gru128","seq": 24,"gat_dim": 64,  "heads": 8, "gru_h": 128, "gru_l": 1, "dp": 0.2, "head": 64},
    # 2-layer GRU
    {"name": "s24_gat64_gru128_2L","seq": 24,"gat_dim": 64,  "heads": 4, "gru_h": 128, "gru_l": 2, "dp": 0.2, "head": 64},
    # Lower dropout
    {"name": "s24_gat64_gru128_dp1","seq":24,"gat_dim": 64,  "heads": 4, "gru_h": 128, "gru_l": 1, "dp": 0.1, "head": 64},
    # Larger
    {"name": "s24_gat128_gru256", "seq": 24, "gat_dim": 128, "heads": 4, "gru_h": 256, "gru_l": 1, "dp": 0.25,"head": 128},
]

# ── Run sweep ─────────────────────────────────────────────────────────
results_all = []

for market, y_tr, y_va, vt, vv, anchor_va, spot_va in [
    ("FR", fr_y_tr, fr_y_va, fr_vt, fr_vv, fr_rm_va, fr_spot_va),
    ("UK", uk_y_tr, uk_y_va, uk_vt, uk_vv, uk_moc_va, uk_spot_va),
]:
    print(f"\n{'='*90}")
    print(f"  {market} — GAT-GRU SWEEP")
    print(f"{'='*90}")

    best = {"rmse_hbc": 999}

    for cfg in CONFIGS:
        t1 = time.time()
        seq_len = cfg["seq"]

        seq_tr, tgt_tr, idx_tr = build_sequences_grouped(
            group_data_tr, y_tr, vt, seq_len, GROUP_NAMES)
        seq_va, tgt_va, idx_va = build_sequences_grouped(
            group_data_va, y_va, vv, seq_len, GROUP_NAMES)

        if len(seq_tr) < 100 or len(seq_va) < 100:
            print(f"    {cfg['name']:30s}  SKIP")
            continue

        torch.manual_seed(42); np.random.seed(42)

        model = GATGRU(
            n_groups=n_groups, max_group_size=max_group_size,
            gat_dim=cfg["gat_dim"], n_heads=cfg["heads"],
            gru_hidden=cfg["gru_h"], gru_layers=cfg["gru_l"],
            dropout=cfg["dp"], head_size=cfg["head"],
        )

        model, epochs = train_model(model, seq_tr, tgt_tr.astype(np.float32),
                                     seq_va, tgt_va.astype(np.float32),
                                     lr=1e-3, wd=1e-4, bs=128)

        preds_dev = predict_model(model, seq_va)
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
        }
        results_all.append(result)

        flag = " ***" if rmse_hbc < best["rmse_hbc"] else ""
        print(f"    {cfg['name']:30s}  RMSE={rmse:.2f}  +HBC={rmse_hbc:.2f}  "
              f"ep={epochs:3d}  params={n_params:>8,}  {elapsed:.1f}s{flag}")

        if rmse_hbc < best["rmse_hbc"]:
            best = result.copy()

        del model
        if DEVICE.type == "mps": torch.mps.empty_cache()

    print(f"\n  {market} BEST: {best.get('config','?')}  +HBC={best['rmse_hbc']:.2f}")

# Summary
print(f"\n{'='*90}")
print(f"  SUMMARY")
print(f"{'='*90}")
for market in ["FR", "UK"]:
    mr = [r for r in results_all if r["market"] == market]
    top = sorted(mr, key=lambda x: x["rmse_hbc"])[:6]
    print(f"\n  {market} top 6:")
    for r in top:
        print(f"    {r['config']:30s}  RMSE={r['rmse']}  +HBC={r['rmse_hbc']}  "
              f"ep={r['epochs']}  params={r['params']:,}")

with open("outputs/gat_gru_ab_test.json", "w") as f:
    json.dump(results_all, f, indent=2)

print(f"\n  Total time: {time.time() - t0:.0f}s")
