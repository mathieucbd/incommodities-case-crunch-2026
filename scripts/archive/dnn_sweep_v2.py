"""DNN sweep v2 — Fine-grained architecture & regularization search.

Based on v1 findings:
  - Huber loss dominates MSE everywhere → stick with Huber
  - FR prefers 2L (med), UK prefers 3L (large)
  - LeakyReLU / GELU good, SELU terrible

This sweep tests:
  - Layer widths: narrower and wider around best configs
  - 4-layer networks for UK
  - Dropout rates: 0.1, 0.15, 0.2, 0.25, 0.3
  - Learning rates: 5e-4, 1e-3, 2e-3
  - Weight decay: 1e-5, 1e-4, 1e-3
  - Batch sizes: 128, 256, 512
  - Skip/residual connections
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

# ── Device ────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"  Device: {DEVICE}")

# ── Data ──────────────────────────────────────────────────────────────
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

EXCLUDE = {"fr_spot", "uk_spot", "datetime_CET", "datetime_UTC", "date", "id"}
ALL_FEATURES = [c for c in df_tr.columns
                if c not in EXCLUDE
                and df_tr[c].dtype in ["float64", "float32", "int64", "int32"]
                and df_tr[c].notna().sum() > len(df_tr) * 0.5]

corr_matrix = df_tr[ALL_FEATURES].corr().abs()
to_drop = set()
for i in range(len(ALL_FEATURES)):
    if ALL_FEATURES[i] in to_drop:
        continue
    for j in range(i + 1, len(ALL_FEATURES)):
        if ALL_FEATURES[j] in to_drop:
            continue
        if corr_matrix.iloc[i, j] > 0.99:
            to_drop.add(ALL_FEATURES[j])
FEATURES = [f for f in ALL_FEATURES if f not in to_drop]
print(f"  Features: {len(FEATURES)} after dedup")

# Targets
fr_spot_la_full = df["fr_spot_la"].values
ema_fr = pd.Series(fr_spot_la_full).ewm(span=240).mean().values
fr_anchor_va = ema_fr[mask_val]
fr_spot_va = df_va["fr_spot"].values
fr_y_tr = df_tr["fr_spot"].values - ema_fr[~mask_val]
fr_y_va = fr_spot_va - fr_anchor_va

uk_moc_va = df_va["uk_merit_order_cost"].values
uk_spot_va = df_va["uk_spot"].values
uk_y_tr = df_tr["uk_spot"].values - df_tr["uk_merit_order_cost"].values
uk_y_va = uk_spot_va - uk_moc_va

hours_va = df_va["hour"].values

fr_valid_tr = np.isfinite(fr_y_tr) & np.isfinite(ema_fr[~mask_val])
fr_valid_va = np.isfinite(fr_y_va) & np.isfinite(fr_anchor_va)
uk_valid_tr = np.isfinite(uk_y_tr)
uk_valid_va = np.isfinite(uk_y_va)


# ── Models ────────────────────────────────────────────────────────────
class ElecDNN(nn.Module):
    def __init__(self, n_features, hidden_layers, activation="leaky_relu",
                 dropout=0.2, batch_norm=True):
        super().__init__()
        layers = []
        in_dim = n_features
        for neurons in hidden_layers:
            layers.append(nn.Linear(in_dim, neurons))
            if batch_norm:
                layers.append(nn.BatchNorm1d(neurons))
            if activation == "leaky_relu":
                layers.append(nn.LeakyReLU(0.01))
            elif activation == "gelu":
                layers.append(nn.GELU())
            elif activation == "relu":
                layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = neurons
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class ResidualDNN(nn.Module):
    """DNN with skip connections between blocks."""
    def __init__(self, n_features, hidden_layers, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(n_features, hidden_layers[0])
        self.blocks = nn.ModuleList()
        for i, neurons in enumerate(hidden_layers):
            in_dim = hidden_layers[i - 1] if i > 0 else hidden_layers[0]
            block = nn.Sequential(
                nn.Linear(in_dim, neurons),
                nn.BatchNorm1d(neurons),
                nn.LeakyReLU(0.01),
                nn.Dropout(dropout),
            )
            self.blocks.append(block)
            # Skip projection if dims don't match
            if in_dim != neurons:
                self.blocks.append(nn.Linear(in_dim, neurons))
            else:
                self.blocks.append(None)
        self.output = nn.Linear(hidden_layers[-1], 1)

    def forward(self, x):
        x = self.input_proj(x)
        for i in range(0, len(self.blocks), 2):
            block = self.blocks[i]
            skip_proj = self.blocks[i + 1]
            residual = x
            x = block(x)
            if skip_proj is not None:
                residual = skip_proj(residual)
            if x.shape == residual.shape:
                x = x + residual
        return self.output(x).squeeze(-1)


def train_dnn(model, X_tr, y_tr, X_va, y_va, lr=1e-3, weight_decay=1e-4,
              batch_size=256, max_epochs=500, patience=30):
    model = model.to(DEVICE)
    train_ds = TensorDataset(
        torch.FloatTensor(X_tr).to(DEVICE),
        torch.FloatTensor(y_tr).to(DEVICE),
    )
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                        drop_last=len(X_tr) % batch_size == 1)
    X_va_t = torch.FloatTensor(X_va).to(DEVICE)
    y_va_t = torch.FloatTensor(y_va).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5, min_lr=1e-6)
    criterion = nn.HuberLoss(delta=5.0)

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_va_t), y_va_t).item()
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    return model, epoch + 1


def predict_dnn(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.FloatTensor(X).to(DEVICE)).cpu().numpy()


def compute_rmse(actual, preds):
    return np.sqrt(np.mean((actual - preds) ** 2))

def apply_hbc(preds, actual, hours):
    errors = actual - preds
    hbc = {h: float(errors[hours == h].mean()) for h in range(24) if (hours == h).sum() > 0}
    corrected = preds + np.array([hbc.get(h, 0) for h in hours])
    return np.sqrt(np.mean((actual - corrected) ** 2))


# ── Prepare data ─────────────────────────────────────────────────────
feat = [f for f in FEATURES if f in df_tr.columns]
scaler = StandardScaler()
X_tr_all = scaler.fit_transform(np.nan_to_num(df_tr[feat].values, 0))
X_va_all = scaler.transform(np.nan_to_num(df_va[feat].values, 0))
n_feat = X_tr_all.shape[1]

# ══════════════════════════════════════════════════════════════════════
#  SWEEP CONFIGS — market-specific
# ══════════════════════════════════════════════════════════════════════

# FR: 2-layer works best. Test widths, dropout, LR, weight_decay.
FR_CONFIGS = [
    # Width variations (2L, huber, leaky_relu)
    {"name": "2L_192_96",     "layers": [192, 96],      "dp": 0.2, "lr": 1e-3, "wd": 1e-4, "bs": 256},
    {"name": "2L_256_128",    "layers": [256, 128],      "dp": 0.2, "lr": 1e-3, "wd": 1e-4, "bs": 256},  # baseline
    {"name": "2L_384_192",    "layers": [384, 192],      "dp": 0.2, "lr": 1e-3, "wd": 1e-4, "bs": 256},
    {"name": "2L_512_256",    "layers": [512, 256],      "dp": 0.2, "lr": 1e-3, "wd": 1e-4, "bs": 256},
    {"name": "2L_256_64",     "layers": [256, 64],       "dp": 0.2, "lr": 1e-3, "wd": 1e-4, "bs": 256},
    # Dropout variations
    {"name": "2L_256_dp10",   "layers": [256, 128],      "dp": 0.1, "lr": 1e-3, "wd": 1e-4, "bs": 256},
    {"name": "2L_256_dp15",   "layers": [256, 128],      "dp": 0.15,"lr": 1e-3, "wd": 1e-4, "bs": 256},
    {"name": "2L_256_dp25",   "layers": [256, 128],      "dp": 0.25,"lr": 1e-3, "wd": 1e-4, "bs": 256},
    {"name": "2L_256_dp30",   "layers": [256, 128],      "dp": 0.3, "lr": 1e-3, "wd": 1e-4, "bs": 256},
    # LR variations
    {"name": "2L_256_lr5e4",  "layers": [256, 128],      "dp": 0.2, "lr": 5e-4, "wd": 1e-4, "bs": 256},
    {"name": "2L_256_lr2e3",  "layers": [256, 128],      "dp": 0.2, "lr": 2e-3, "wd": 1e-4, "bs": 256},
    # Weight decay variations
    {"name": "2L_256_wd1e5",  "layers": [256, 128],      "dp": 0.2, "lr": 1e-3, "wd": 1e-5, "bs": 256},
    {"name": "2L_256_wd1e3",  "layers": [256, 128],      "dp": 0.2, "lr": 1e-3, "wd": 1e-3, "bs": 256},
    # Batch size variations
    {"name": "2L_256_bs128",  "layers": [256, 128],      "dp": 0.2, "lr": 1e-3, "wd": 1e-4, "bs": 128},
    {"name": "2L_256_bs512",  "layers": [256, 128],      "dp": 0.2, "lr": 1e-3, "wd": 1e-4, "bs": 512},
    # Residual (same width for skip connections)
    {"name": "res_256_256",   "layers": [256, 256],      "dp": 0.2, "lr": 1e-3, "wd": 1e-4, "bs": 256, "residual": True},
    {"name": "res_256x3",     "layers": [256, 256, 256], "dp": 0.2, "lr": 1e-3, "wd": 1e-4, "bs": 256, "residual": True},
]

# UK: 3-layer works best, larger is better. Test deeper and wider.
UK_CONFIGS = [
    # Width variations (3L)
    {"name": "3L_256_128_64",  "layers": [256, 128, 64],   "dp": 0.2, "lr": 1e-3, "wd": 1e-4, "bs": 256},
    {"name": "3L_384_192_96",  "layers": [384, 192, 96],   "dp": 0.2, "lr": 1e-3, "wd": 1e-4, "bs": 256},
    {"name": "3L_512_256_128", "layers": [512, 256, 128],  "dp": 0.2, "lr": 1e-3, "wd": 1e-4, "bs": 256},
    {"name": "3L_512_256_128_dp30", "layers": [512, 256, 128], "dp": 0.3, "lr": 1e-3, "wd": 1e-4, "bs": 256},  # baseline best
    {"name": "3L_768_384_192", "layers": [768, 384, 192],  "dp": 0.3, "lr": 1e-3, "wd": 1e-4, "bs": 256},
    # 4-layer
    {"name": "4L_512_256_128_64", "layers": [512, 256, 128, 64], "dp": 0.25, "lr": 1e-3, "wd": 1e-4, "bs": 256},
    {"name": "4L_384_256_128_64", "layers": [384, 256, 128, 64], "dp": 0.2,  "lr": 1e-3, "wd": 1e-4, "bs": 256},
    # Dropout variations (on best 3L)
    {"name": "3L_512_dp15",    "layers": [512, 256, 128],  "dp": 0.15, "lr": 1e-3, "wd": 1e-4, "bs": 256},
    {"name": "3L_512_dp20",    "layers": [512, 256, 128],  "dp": 0.2,  "lr": 1e-3, "wd": 1e-4, "bs": 256},
    {"name": "3L_512_dp25",    "layers": [512, 256, 128],  "dp": 0.25, "lr": 1e-3, "wd": 1e-4, "bs": 256},
    {"name": "3L_512_dp35",    "layers": [512, 256, 128],  "dp": 0.35, "lr": 1e-3, "wd": 1e-4, "bs": 256},
    # LR variations
    {"name": "3L_512_lr5e4",   "layers": [512, 256, 128],  "dp": 0.3, "lr": 5e-4, "wd": 1e-4, "bs": 256},
    {"name": "3L_512_lr2e3",   "layers": [512, 256, 128],  "dp": 0.3, "lr": 2e-3, "wd": 1e-4, "bs": 256},
    # Weight decay
    {"name": "3L_512_wd1e5",   "layers": [512, 256, 128],  "dp": 0.3, "lr": 1e-3, "wd": 1e-5, "bs": 256},
    {"name": "3L_512_wd1e3",   "layers": [512, 256, 128],  "dp": 0.3, "lr": 1e-3, "wd": 1e-3, "bs": 256},
    # Batch size
    {"name": "3L_512_bs128",   "layers": [512, 256, 128],  "dp": 0.3, "lr": 1e-3, "wd": 1e-4, "bs": 128},
    {"name": "3L_512_bs512",   "layers": [512, 256, 128],  "dp": 0.3, "lr": 1e-3, "wd": 1e-4, "bs": 512},
    # Residual
    {"name": "res_256x3",      "layers": [256, 256, 256],  "dp": 0.2, "lr": 1e-3, "wd": 1e-4, "bs": 256, "residual": True},
    {"name": "res_384x3",      "layers": [384, 384, 384],  "dp": 0.25,"lr": 1e-3, "wd": 1e-4, "bs": 256, "residual": True},
]

# ── Run sweep ─────────────────────────────────────────────────────────
print(f"\n{'='*90}")
print(f"  DNN SWEEP v2 — Fine-grained architecture search")
print(f"{'='*90}")

results_all = []

for market, configs, y_tr, y_va, v_tr, v_va, anchor_va, spot_va in [
    ("FR", FR_CONFIGS, fr_y_tr, fr_y_va, fr_valid_tr, fr_valid_va, fr_anchor_va, fr_spot_va),
    ("UK", UK_CONFIGS, uk_y_tr, uk_y_va, uk_valid_tr, uk_valid_va, uk_moc_va, uk_spot_va),
]:
    print(f"\n{'='*90}")
    print(f"  {market} — {len(configs)} configs")
    print(f"{'='*90}")

    X_tr = X_tr_all[v_tr]
    X_va = X_va_all[v_va]
    yt = y_tr[v_tr].astype(np.float32)
    yv = y_va[v_va].astype(np.float32)
    hrs = hours_va[v_va]
    actual = spot_va[v_va]
    anch = anchor_va[v_va]

    best = {"rmse_hbc": 999}

    for cfg in configs:
        t1 = time.time()
        torch.manual_seed(42)
        np.random.seed(42)

        if cfg.get("residual"):
            model = ResidualDNN(n_feat, cfg["layers"], dropout=cfg["dp"])
        else:
            model = ElecDNN(n_feat, cfg["layers"], dropout=cfg["dp"], batch_norm=True)

        model, epochs = train_dnn(
            model, X_tr, yt, X_va, yv,
            lr=cfg["lr"], weight_decay=cfg["wd"],
            batch_size=cfg["bs"], max_epochs=500, patience=30,
        )

        preds_dev = predict_dnn(model, X_va)
        preds_spot = anch + preds_dev
        rmse = compute_rmse(actual, preds_spot)
        rmse_hbc = apply_hbc(preds_spot, actual, hrs)
        n_params = sum(p.numel() for p in model.parameters())

        elapsed = time.time() - t1
        result = {
            "market": market, "config": cfg["name"], "rmse": round(rmse, 2),
            "rmse_hbc": round(rmse_hbc, 2), "epochs": epochs,
            "params": n_params, "time": round(elapsed, 1),
            **{k: v for k, v in cfg.items() if k != "name"},
        }
        results_all.append(result)

        flag = " ***" if rmse_hbc < best["rmse_hbc"] else ""
        print(f"    {cfg['name']:25s}  RMSE={rmse:.2f}  +HBC={rmse_hbc:.2f}  "
              f"ep={epochs:3d}  params={n_params:>8,}  {elapsed:.1f}s{flag}")

        if rmse_hbc < best["rmse_hbc"]:
            best = result.copy()

        del model
        if DEVICE.type == "mps":
            torch.mps.empty_cache()

    print(f"\n  {market} BEST: {best['config']}  RMSE={best['rmse']:.2f}  +HBC={best['rmse_hbc']:.2f}")

# ── Summary ───────────────────────────────────────────────────────────
print(f"\n{'='*90}")
print(f"  FINAL SUMMARY")
print(f"{'='*90}")

for market in ["FR", "UK"]:
    mr = [r for r in results_all if r["market"] == market]
    top = sorted(mr, key=lambda x: x["rmse_hbc"])[:8]
    print(f"\n  {market} top 8:")
    for r in top:
        print(f"    {r['config']:25s}  RMSE={r['rmse']:.2f}  +HBC={r['rmse_hbc']:.2f}  "
              f"ep={r['epochs']}  params={r['params']:,}  "
              f"dp={r.get('dp','-')} lr={r.get('lr','-')} wd={r.get('wd','-')} bs={r.get('bs','-')}")

with open("outputs/dnn_sweep_v2.json", "w") as f:
    json.dump(results_all, f, indent=2, default=str)

print(f"\n  Total time: {time.time() - t0:.0f}s")
