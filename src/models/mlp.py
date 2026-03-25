"""Hybrid Linear + MLP model for electricity price forecasting.

Based on: "Electricity Price Forecasting: Bridging Linear Models, Neural Networks
and Online Learning" (arXiv:2601.02856, January 2026).

The HybridLinearMLP adds a direct linear pathway to a standard MLP, allowing the
model to learn both linear and non-linear components of the price signal simultaneously.
This is particularly useful for DAM prices where a large share of variance is explained
by linear relationships (gas price, load, renewables) but residuals are non-linear.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin
from torch.utils.data import DataLoader, TensorDataset

# ── Device selection (mirrors dnn.py convention) ──────────────────────────────
if torch.backends.mps.is_available():
    MLP_DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    MLP_DEVICE = torch.device("cuda")
else:
    MLP_DEVICE = torch.device("cpu")


# ══════════════════════════════════════════════════════════════════════════════
# 1. PyTorch Module
# ══════════════════════════════════════════════════════════════════════════════

class _ResidualBlock(nn.Module):
    """Single residual block: projection shortcut + BN → act → dropout → linear."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float, act: str) -> None:
        super().__init__()
        activation = nn.GELU() if act == "gelu" else nn.LeakyReLU(negative_slope=0.01)
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            activation,
            nn.Dropout(dropout),
        )
        self.shortcut = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + self.shortcut(x)


class HybridLinearMLP(nn.Module):
    """Dual-pathway architecture: linear shortcut + non-linear MLP branch.

    forward(x) = linear_out + mlp_out

    The linear branch provides a strong inductive bias for the dominant linear
    relationships in electricity markets, while the MLP captures residual
    non-linearity (regime changes, interactions, spikes).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (256, 128, 64),
        dropout: float = 0.2,
        activation: str = "gelu",
        residual: bool = True,
    ) -> None:
        super().__init__()

        # Linear branch: single affine map, no activation
        self.linear_branch = nn.Linear(input_dim, 1)

        # Non-linear branch
        if residual:
            layers: list[nn.Module] = []
            prev_dim = input_dim
            for hdim in hidden_dims:
                layers.append(_ResidualBlock(prev_dim, hdim, dropout, activation))
                prev_dim = hdim
            layers.append(nn.Linear(prev_dim, 1))
            self.mlp_branch = nn.Sequential(*layers)
        else:
            act_fn = nn.GELU if activation == "gelu" else lambda: nn.LeakyReLU(negative_slope=0.01)
            layers = []
            prev_dim = input_dim
            for i, hdim in enumerate(hidden_dims):
                layers.append(nn.Linear(prev_dim, hdim))
                layers.append(nn.BatchNorm1d(hdim))
                layers.append(act_fn())
                if i < len(hidden_dims) - 1:
                    layers.append(nn.Dropout(dropout))
                prev_dim = hdim
            layers.append(nn.Linear(prev_dim, 1))
            self.mlp_branch = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self) -> None:
        """Explicit weight initialization for stable training."""
        for m in self.mlp_branch.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.01, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # Linear branch: Xavier for stable initial contribution
        nn.init.xavier_normal_(self.linear_branch.weight)
        nn.init.zeros_(self.linear_branch.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        linear_out = self.linear_branch(x)   # (B, 1)
        mlp_out = self.mlp_branch(x)          # (B, 1)
        return (linear_out + mlp_out).squeeze(-1)  # (B,)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Scikit-Learn compatible wrapper
# ══════════════════════════════════════════════════════════════════════════════

class HybridMLPRegressor(BaseEstimator, RegressorMixin):
    """Scikit-Learn wrapper for HybridLinearMLP.

    Fits on (X_train, y_train) numpy arrays, returns numpy predictions.
    Uses AdamW + early stopping on an internal validation split.
    All tensor operations run on MLP_DEVICE (CUDA/MPS/CPU).
    """

    def __init__(
        self,
        hidden_dims: tuple[int, ...] = (256, 128, 64),
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        max_epochs: int = 500,
        patience: int = 50,
        val_fraction: float = 0.1,
        loss: str = "mse",
        huber_delta: float = 5.0,
        warmup_epochs: int = 10,
        grad_clip: float = 1.0,
        activation: str = "gelu",
        residual: bool = True,
        n_seeds: int = 1,
        random_state: int = 42,
    ) -> None:
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_fraction = val_fraction
        self.loss = loss
        self.huber_delta = huber_delta
        self.warmup_epochs = warmup_epochs
        self.grad_clip = grad_clip
        self.activation = activation
        self.residual = residual
        self.n_seeds = n_seeds
        self.random_state = random_state

        self.models_: list[HybridLinearMLP] = []
        self.n_features_in_: int | None = None
        self.best_epoch_: int = 0
        self.best_epochs_: list[int] = []

    def _train_single(self, X: np.ndarray, y: np.ndarray, seed: int) -> tuple[HybridLinearMLP, int]:
        """Train a single model with the given seed. Returns (model, best_epoch)."""
        torch.manual_seed(seed)
        np.random.seed(seed)

        n_total = len(X)
        n_val = max(1, int(n_total * self.val_fraction))
        n_tr = n_total - n_val

        X_tr, X_va = X[:n_tr], X[n_tr:]
        y_tr, y_va = y[:n_tr], y[n_tr:]

        n_features = X_tr.shape[1]

        model = HybridLinearMLP(
            input_dim=n_features,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
            activation=self.activation,
            residual=self.residual,
        ).to(MLP_DEVICE)

        ds_tr = TensorDataset(
            torch.from_numpy(X_tr).to(MLP_DEVICE),
            torch.from_numpy(y_tr).to(MLP_DEVICE),
        )
        drop = len(X_tr) % self.batch_size < 2
        loader = DataLoader(ds_tr, batch_size=self.batch_size, shuffle=True, drop_last=drop)

        X_va_t = torch.from_numpy(X_va).to(MLP_DEVICE)
        y_va_t = torch.from_numpy(y_va).to(MLP_DEVICE)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        # Cosine annealing with linear warmup
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.05, total_iters=self.warmup_epochs
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, self.max_epochs - self.warmup_epochs),
            eta_min=self.lr / 100,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[self.warmup_epochs]
        )

        if self.loss == "mse":
            criterion = nn.MSELoss()
        else:
            criterion = nn.HuberLoss(delta=self.huber_delta)

        # Always evaluate early stopping on MSE (aligned with RMSE metric)
        val_criterion = nn.MSELoss()

        best_val_loss = float("inf")
        best_state: dict | None = None
        no_improve = 0
        best_epoch = 0

        for epoch in range(self.max_epochs):
            model.train()
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.grad_clip)
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_loss = val_criterion(model(X_va_t), y_va_t).item()

            scheduler.step()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
                best_epoch = epoch + 1
            else:
                no_improve += 1

            if no_improve >= self.patience:
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        return model, best_epoch

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "HybridMLPRegressor":
        """Train the hybrid model with early stopping.

        If n_seeds > 1, trains multiple models and averages their predictions.
        The internal validation split is taken from the END of the array
        (chronological, no lookahead bias).
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self.n_features_in_ = X.shape[1]

        self.models_ = []
        self.best_epochs_ = []

        for i in range(self.n_seeds):
            seed = self.random_state + i
            model, best_ep = self._train_single(X, y, seed)
            self.models_.append(model)
            self.best_epochs_.append(best_ep)

        # Primary best_epoch_ is the median across seeds (for retrain logic)
        self.best_epoch_ = int(np.median(self.best_epochs_))
        return self

    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions as a 1-D numpy array (averaged across seeds)."""
        if not self.models_:
            raise RuntimeError("Call .fit() before .predict()")
        X = np.asarray(X, dtype=np.float32)
        X_t = torch.from_numpy(X).to(MLP_DEVICE)

        preds_list = []
        for model in self.models_:
            model.eval()
            with torch.no_grad():
                preds_list.append(model(X_t).cpu().numpy())

        return np.mean(preds_list, axis=0)


# ══════════════════════════════════════════════════════════════════════════════
# Standalone test block
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    import yaml
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    sys.path.insert(0, ".")
    from src.data_loading import load_data
    from src.feature_engineering import build_features
    from src.models.metrics import compute_rmse, compute_hbc
    from src.models.targets import prepare_stationary

    print("=" * 90)
    print("  HybridMLPRegressor — standalone test (real data, optimized)")
    print("=" * 90)
    print(f"  Device: {MLP_DEVICE}")

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    # ── Load + feature engineering (same as pipeline) ─────────────────
    x_train, y_train, x_test = load_data("data/raw")
    train_fe = build_features(pd.concat([x_train], axis=0), config)
    train_fe = train_fe.join(y_train[["fr_spot", "uk_spot"]])

    holdout_start = config["validation"]["holdout_start"]
    mask_val = train_fe["datetime_CET"] >= holdout_start
    df_train = train_fe[~mask_val].copy()
    df_val   = train_fe[mask_val].copy()
    df_train_uk = df_train.copy()

    print(f"  Train: {len(df_train)}, Val: {len(df_val)}")

    # ── feat_dnn: computed BEFORE the interaction feature is added ────
    _EXCLUDE = {"fr_spot", "uk_spot", "datetime_CET", "datetime_UTC", "date", "id"}
    _all_num = [c for c in df_train.columns
                if c not in _EXCLUDE
                and df_train[c].dtype in ["float64", "float32", "int64", "int32"]
                and df_train[c].notna().sum() > len(df_train) * 0.5]
    _corr = df_train[_all_num].corr().abs()
    _to_drop: set = set()
    for _i in range(len(_all_num)):
        if _all_num[_i] in _to_drop:
            continue
        for _j in range(_i + 1, len(_all_num)):
            if _all_num[_j] in _to_drop:
                continue
            if _corr.iloc[_i, _j] > 0.99:
                _to_drop.add(_all_num[_j])
    feat_dnn_final = [f for f in _all_num if f not in _to_drop and f in df_train.columns]
    print(f"  HybridMLP features: {len(feat_dnn_final)} (after 0.99 corr dedup)")

    # ── Interaction feature (same as pipeline, added after feat_dnn) ──
    for df in [train_fe, df_train, df_val]:
        if "fr_spot_la_roll_168h_mean" in df.columns and "uk_price_per_mw_7d" in df.columns:
            df["X_fr_spot_la_roll_168h_mean_x_uk_price_per_mw_7d"] = (
                df["fr_spot_la_roll_168h_mean"] * df["uk_price_per_mw_7d"]
            )

    # ── FR stationary target ───────────────────────────────────────────
    fr_stat = prepare_stationary("fr_spot_la", "fr_spot", train_fe, df_train, df_val)
    hours_va_fr = df_val["hour"].values

    # ── UK basis target ────────────────────────────────────────────────
    uk_spot_va  = df_val["uk_spot"].values
    uk_moc_va   = df_val["uk_merit_order_cost"].values
    uk_moc_tr   = df_train_uk["uk_merit_order_cost"].values
    y_basis_tr  = df_train_uk["uk_spot"].values - uk_moc_tr
    valid_basis_tr = np.isfinite(y_basis_tr)
    hours_va_uk = df_val["hour"].values

    # ── StandardScaler — fit on train only (same as pipeline) ─────────
    scaler_fr = StandardScaler()
    X_tr_fr = scaler_fr.fit_transform(np.nan_to_num(df_train[feat_dnn_final].values, 0))
    X_va_fr  = scaler_fr.transform(np.nan_to_num(df_val[feat_dnn_final].values, 0))

    scaler_uk = StandardScaler()
    X_tr_uk = scaler_uk.fit_transform(np.nan_to_num(df_train_uk[feat_dnn_final].values, 0))
    X_va_uk  = scaler_uk.transform(np.nan_to_num(df_val[feat_dnn_final].values, 0))

    # ── FR HybridMLP (optimized) ──────────────────────────────────────
    print("\n" + "=" * 90)
    print(f"  FR HybridMLP ({len(feat_dnn_final)} features)")
    print("=" * 90)
    hmlp_fr = HybridMLPRegressor(
        hidden_dims=(256, 128, 64), dropout=0.15, lr=2e-4, weight_decay=1e-4,
        batch_size=128, max_epochs=500, patience=50, val_fraction=0.1,
        loss="huber", huber_delta=10.0, warmup_epochs=20, grad_clip=1.0,
        activation="gelu", residual=False, n_seeds=3, random_state=42,
    )
    hmlp_fr.fit(
        X_tr_fr[fr_stat["valid_tr"]],
        fr_stat["y_dev_tr"][fr_stat["valid_tr"]].astype(np.float32),
    )
    preds_fr = fr_stat["rm_va"] + hmlp_fr.predict(X_va_fr)
    rmse_fr = compute_rmse(fr_stat["spot_va"], preds_fr)
    _, rmse_fr_hbc = compute_hbc(preds_fr, fr_stat["spot_va"], hours_va_fr)
    ep_str_fr = "/".join(str(e) for e in hmlp_fr.best_epochs_)
    print(f"  HybridMLP FR: RMSE={rmse_fr:.2f}, +HBC={rmse_fr_hbc:.2f}, "
          f"best_ep={hmlp_fr.best_epoch_} [{ep_str_fr}]")

    # ── UK HybridMLP (optimized) ──────────────────────────────────────
    print("\n" + "=" * 90)
    print(f"  UK HybridMLP ({len(feat_dnn_final)} features, basis target)")
    print("=" * 90)
    hmlp_uk = HybridMLPRegressor(
        hidden_dims=(512, 256, 128), dropout=0.25, lr=5e-4, weight_decay=1e-4,
        batch_size=128, max_epochs=500, patience=50, val_fraction=0.1,
        loss="mse", warmup_epochs=10, grad_clip=1.0,
        activation="gelu", residual=False, n_seeds=3, random_state=42,
    )
    hmlp_uk.fit(
        X_tr_uk[valid_basis_tr],
        y_basis_tr[valid_basis_tr].astype(np.float32),
    )
    preds_uk = uk_moc_va + hmlp_uk.predict(X_va_uk)
    rmse_uk = compute_rmse(uk_spot_va, preds_uk)
    _, rmse_uk_hbc = compute_hbc(preds_uk, uk_spot_va, hours_va_uk)
    ep_str_uk = "/".join(str(e) for e in hmlp_uk.best_epochs_)
    print(f"  HybridMLP UK: RMSE={rmse_uk:.2f}, +HBC={rmse_uk_hbc:.2f}, "
          f"best_ep={hmlp_uk.best_epoch_} [{ep_str_uk}]")

    # ── Final summary ──────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("  FINAL SUMMARY")
    print("=" * 90)
    print("\n  Validation scores:")
    print(f"    FR HybridMLP:      RMSE={rmse_fr:.2f}  +HBC={rmse_fr_hbc:.2f}  ep={hmlp_fr.best_epoch_} [{ep_str_fr}]")
    print(f"    UK HybridMLP:      RMSE={rmse_uk:.2f}  +HBC={rmse_uk_hbc:.2f}  ep={hmlp_uk.best_epoch_} [{ep_str_uk}]")
    print(f"\n  FINAL SUM (w/ HBC): {rmse_fr_hbc + rmse_uk_hbc:.2f}")
    print(f"    FR: {rmse_fr_hbc:.2f}")
    print(f"    UK: {rmse_uk_hbc:.2f}")
    print(f"\n  Baseline (old): FR=18.67 +HBC=18.27 / UK=13.48 +HBC=11.41 / SUM=29.68")
