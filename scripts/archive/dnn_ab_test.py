"""DNN A/B test — PyTorch adaptation of epftoolbox DNN for electricity price forecasting.

Uses ALL 350 deduped numeric features (vs 28/150 for trees).
DNN with dropout + L2 regularization handles high-dimensional input.
Tests multiple architectures, activations, and losses.
MPS (Apple Silicon GPU) supported.
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

# ── Data loading ──────────────────────────────────────────────────────
with open("config.yaml") as f:
    config = yaml.safe_load(f)

print("=" * 90)
print("  DNN A/B TEST — PyTorch (epftoolbox-inspired)")
print("=" * 90)

t0 = time.time()
x_train, y_train, x_test = load_data("data/raw")
train = merge_train(x_train, y_train)
df = build_features(train, config)

holdout_start = config["validation"]["holdout_start"]
mask_val = (df["datetime_CET"] >= holdout_start).values

df_tr = df[~mask_val].copy()
df_va = df[mask_val].copy()

# ── All numeric features (350 deduped) ────────────────────────────────
EXCLUDE = {"fr_spot", "uk_spot", "datetime_CET", "datetime_UTC", "date", "id"}
ALL_FEATURES = [c for c in df_tr.columns
                if c not in EXCLUDE
                and df_tr[c].dtype in ["float64", "float32", "int64", "int32"]
                and df_tr[c].notna().sum() > len(df_tr) * 0.5]

# Correlation dedup > 0.99
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
print(f"  Features: {len(ALL_FEATURES)} → {len(FEATURES)} after 0.99 dedup")

# ── Targets (stationary) ─────────────────────────────────────────────
# FR: spot - EMA(240h)
fr_spot_la_full = df["fr_spot_la"].values
ema_fr = pd.Series(fr_spot_la_full).ewm(span=240).mean().values
fr_anchor_va = ema_fr[mask_val]
fr_spot_va = df_va["fr_spot"].values
fr_y_tr = df_tr["fr_spot"].values - ema_fr[~mask_val]
fr_y_va = fr_spot_va - fr_anchor_va

# UK: spot - merit_order_cost
uk_moc_va = df_va["uk_merit_order_cost"].values
uk_spot_va = df_va["uk_spot"].values
uk_y_tr = df_tr["uk_spot"].values - df_tr["uk_merit_order_cost"].values
uk_y_va = uk_spot_va - uk_moc_va

hours_va = df_va["hour"].values

# Valid masks
fr_valid_tr = np.isfinite(fr_y_tr) & np.isfinite(ema_fr[~mask_val])
fr_valid_va = np.isfinite(fr_y_va) & np.isfinite(fr_anchor_va)
uk_valid_tr = np.isfinite(uk_y_tr)
uk_valid_va = np.isfinite(uk_y_va)


# ── DNN Model ────────────────────────────────────────────────────────
class ElecDNN(nn.Module):
    """Dense NN inspired by epftoolbox, adapted for hourly prediction."""

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
            elif activation == "selu":
                layers.append(nn.SELU())
            elif activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "gelu":
                layers.append(nn.GELU())

            if dropout > 0:
                if activation == "selu":
                    layers.append(nn.AlphaDropout(dropout))
                else:
                    layers.append(nn.Dropout(dropout))

            in_dim = neurons

        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_dnn(model, X_tr, y_tr, X_va, y_va, lr=1e-3, weight_decay=1e-4,
              batch_size=256, max_epochs=500, patience=30, loss_fn="mse"):
    """Train with early stopping on validation loss."""
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

    if loss_fn == "mse":
        criterion = nn.MSELoss()
    elif loss_fn == "huber":
        criterion = nn.HuberLoss(delta=5.0)
    elif loss_fn == "mae":
        criterion = nn.L1Loss()

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

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_va_t)
            val_loss = criterion(val_pred, y_va_t).item()

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
        X_t = torch.FloatTensor(X).to(DEVICE)
        return model(X_t).cpu().numpy()


# ── Metrics ───────────────────────────────────────────────────────────
def compute_rmse(actual, preds):
    return np.sqrt(np.mean((actual - preds) ** 2))


def apply_hbc(preds, actual, hours):
    errors = actual - preds
    hbc = {h: float(errors[hours == h].mean()) for h in range(24) if (hours == h).sum() > 0}
    corrected = preds + np.array([hbc.get(h, 0) for h in hours])
    return np.sqrt(np.mean((actual - corrected) ** 2))


def error_correlation(e1, e2):
    return np.corrcoef(e1, e2)[0, 1]


# ── Prepare data ─────────────────────────────────────────────────────
feat = [f for f in FEATURES if f in df_tr.columns]
scaler = StandardScaler()
X_tr_all = scaler.fit_transform(np.nan_to_num(df_tr[feat].values, 0))
X_va_all = scaler.transform(np.nan_to_num(df_va[feat].values, 0))
n_feat = X_tr_all.shape[1]
print(f"  Scaled features: {n_feat}")
print(f"  Train: {X_tr_all.shape[0]}, Val: {X_va_all.shape[0]}")

# ── Architecture sweep ───────────────────────────────────────────────
CONFIGS = [
    {"name": "small_2L",    "layers": [128, 64],       "dropout": 0.2, "act": "leaky_relu", "bn": True},
    {"name": "med_2L",      "layers": [256, 128],      "dropout": 0.2, "act": "leaky_relu", "bn": True},
    {"name": "med_3L",      "layers": [256, 128, 64],  "dropout": 0.2, "act": "leaky_relu", "bn": True},
    {"name": "large_3L",    "layers": [512, 256, 128],  "dropout": 0.3, "act": "leaky_relu", "bn": True},
    {"name": "selu_3L",     "layers": [256, 128, 64],  "dropout": 0.1, "act": "selu",       "bn": False},
    {"name": "gelu_3L",     "layers": [256, 128, 64],  "dropout": 0.2, "act": "gelu",       "bn": True},
]

LOSS_FNS = ["mse", "huber"]

# Also load tree model predictions for error correlation
# (we'll compute these inline)

results_all = []

for market, y_tr, y_va, v_tr, v_va, anchor_va, spot_va in [
    ("FR", fr_y_tr, fr_y_va, fr_valid_tr, fr_valid_va, fr_anchor_va, fr_spot_va),
    ("UK", uk_y_tr, uk_y_va, uk_valid_tr, uk_valid_va, uk_moc_va, uk_spot_va),
]:
    print(f"\n{'='*90}")
    print(f"  {market} — DNN SWEEP ({n_feat} features)")
    print(f"{'='*90}")

    X_tr = X_tr_all[v_tr]
    X_va = X_va_all[v_va]
    yt = y_tr[v_tr].astype(np.float32)
    yv = y_va[v_va].astype(np.float32)
    hrs = hours_va[v_va]
    actual = spot_va[v_va]
    anch = anchor_va[v_va]

    best_overall = {"rmse_hbc": 999}

    for cfg in CONFIGS:
        for loss_fn in LOSS_FNS:
            label = f"{cfg['name']}_{loss_fn}"
            t1 = time.time()

            torch.manual_seed(42)
            np.random.seed(42)

            model = ElecDNN(
                n_features=n_feat,
                hidden_layers=cfg["layers"],
                activation=cfg["act"],
                dropout=cfg["dropout"],
                batch_norm=cfg["bn"],
            )

            model, epochs = train_dnn(
                model, X_tr, yt, X_va, yv,
                lr=1e-3, weight_decay=1e-4,
                batch_size=256, max_epochs=500, patience=30,
                loss_fn=loss_fn,
            )

            preds_dev = predict_dnn(model, X_va)
            preds_spot = anch + preds_dev
            rmse = compute_rmse(actual, preds_spot)
            rmse_hbc = apply_hbc(preds_spot, actual, hrs)
            n_params = sum(p.numel() for p in model.parameters())

            elapsed = time.time() - t1
            result = {
                "market": market, "config": label, "rmse": rmse,
                "rmse_hbc": rmse_hbc, "epochs": epochs, "params": n_params,
                "time": elapsed,
            }
            results_all.append(result)

            flag = " ***" if rmse_hbc < best_overall["rmse_hbc"] else ""
            print(f"    {label:25s}  RMSE={rmse:.2f}  +HBC={rmse_hbc:.2f}  "
                  f"ep={epochs:3d}  params={n_params:,}  {elapsed:.1f}s{flag}")

            if rmse_hbc < best_overall["rmse_hbc"]:
                best_overall = result.copy()
                best_overall["preds_dev"] = preds_dev
                best_overall["model_state"] = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }

            del model
            if DEVICE.type == "mps":
                torch.mps.empty_cache()

    print(f"\n  {market} BEST: {best_overall['config']}  "
          f"RMSE={best_overall['rmse']:.2f}  +HBC={best_overall['rmse_hbc']:.2f}")

# ── Summary ───────────────────────────────────────────────────────────
print(f"\n{'='*90}")
print(f"  SUMMARY")
print(f"{'='*90}")

for market in ["FR", "UK"]:
    market_results = [r for r in results_all if r["market"] == market]
    top5 = sorted(market_results, key=lambda x: x["rmse_hbc"])[:5]
    print(f"\n  {market} top 5:")
    for r in top5:
        print(f"    {r['config']:25s}  RMSE={r['rmse']:.2f}  +HBC={r['rmse_hbc']:.2f}  "
              f"ep={r['epochs']}  params={r['params']:,}")

# Save results
with open("outputs/dnn_ab_test.json", "w") as f:
    json.dump([{k: v for k, v in r.items() if k not in ("preds_dev", "model_state")}
               for r in results_all], f, indent=2, default=str)

print(f"\n  Total time: {time.time() - t0:.0f}s")
