"""CNN-LSTM hybrid model for electricity price forecasting.

Architecture:
  1. CNN stage: stacked Conv1d blocks extract local temporal features across
     the lookback window (short-range patterns: hourly fluctuations, ramps).
  2. LSTM stage: takes the CNN output sequence and captures long-range temporal
     dependencies (daily cycles, weekly patterns, regime changes).

The key insight is that CNN feature extraction reduces the effective input
dimensionality for the LSTM, allowing the recurrent stage to focus on
temporal structure rather than raw feature interactions.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin
from torch.utils.data import DataLoader, TensorDataset

# ── Device selection ──────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    CNNLSTM_DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    CNNLSTM_DEVICE = torch.device("cuda")
else:
    CNNLSTM_DEVICE = torch.device("cpu")


# ── Sequence utilities ────────────────────────────────────────────────────────

def _make_sequences(X: np.ndarray, lookback: int) -> np.ndarray:
    """(N, F) → (N, lookback, F) with zero-padding at the start."""
    pad = np.zeros((lookback - 1, X.shape[1]), dtype=X.dtype)
    padded = np.vstack([pad, X])
    return np.stack([padded[i:i + lookback] for i in range(len(X))])


def _make_sequences_notail(X_ext: np.ndarray, lookback: int) -> np.ndarray:
    """(N + lookback - 1, F) → (N, lookback, F) — tail already prepended, no padding."""
    N = len(X_ext) - lookback + 1
    return np.stack([X_ext[i:i + lookback] for i in range(N)])


# ══════════════════════════════════════════════════════════════════════════════
# 1. PyTorch Module
# ══════════════════════════════════════════════════════════════════════════════

class ElecCNNLSTM(nn.Module):
    """CNN feature extractor followed by LSTM temporal modelling.

    forward(x) takes (B, T, F) and returns (B,).

    CNN operates on (B, F, T) (channels-first), then the output is transposed
    back to (B, T, cnn_channels) before being fed to the LSTM.
    """

    def __init__(
        self,
        input_size: int,
        cnn_channels: int = 64,
        lstm_hidden: int = 128,
        cnn_layers: int = 2,
        lstm_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        # CNN stage — no global pooling; preserve time dimension for LSTM
        cnn_blocks: list[nn.Module] = []
        in_ch = input_size
        for _ in range(cnn_layers):
            cnn_blocks.extend([
                nn.Conv1d(in_ch, cnn_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(cnn_channels),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_ch = cnn_channels
        self.cnn = nn.Sequential(*cnn_blocks)

        # LSTM stage
        self.lstm = nn.LSTM(
            cnn_channels, lstm_hidden, lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(lstm_hidden)
        self.fc = nn.Linear(lstm_hidden, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.cnn.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        for name, p in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.zeros_(p)
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        # CNN expects (B, F, T)
        x = x.transpose(1, 2)                     # (B, F, T)
        x = self.cnn(x)                            # (B, cnn_channels, T)
        x = x.transpose(1, 2)                     # (B, T, cnn_channels)

        # LSTM temporal modelling
        out, _ = self.lstm(x)                     # (B, T, lstm_hidden)
        last = out[:, -1, :]                       # (B, lstm_hidden)
        return self.fc(self.norm(last)).squeeze(-1)  # (B,)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Scikit-Learn compatible wrapper
# ══════════════════════════════════════════════════════════════════════════════

class CNNLSTMRegressor(BaseEstimator, RegressorMixin):
    """Scikit-Learn wrapper for ElecCNNLSTM.

    Converts 2D (N, F) input to 3D sequences internally.
    Stores the training tail for zero-padding-free test predictions.
    """

    def __init__(
        self,
        lookback: int = 48,
        cnn_channels: int = 64,
        lstm_hidden: int = 128,
        cnn_layers: int = 2,
        lstm_layers: int = 1,
        dropout: float = 0.2,
        lr: float = 5e-4,
        weight_decay: float = 1e-4,
        batch_size: int = 128,
        max_epochs: int = 300,
        patience: int = 30,
        val_fraction: float = 0.1,
        loss: str = "huber",
        huber_delta: float = 5.0,
        warmup_epochs: int = 10,
        grad_clip: float = 1.0,
        n_seeds: int = 1,
        random_state: int = 42,
    ) -> None:
        self.lookback = lookback
        self.cnn_channels = cnn_channels
        self.lstm_hidden = lstm_hidden
        self.cnn_layers = cnn_layers
        self.lstm_layers = lstm_layers
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
        self.n_seeds = n_seeds
        self.random_state = random_state

        self.models_: list[ElecCNNLSTM] = []
        self.best_epoch_: int = 0
        self.best_epochs_: list[int] = []
        self._X_tail: np.ndarray | None = None

    def _train_single(self, X: np.ndarray, y: np.ndarray, seed: int) -> tuple[ElecCNNLSTM, int]:
        torch.manual_seed(seed)
        np.random.seed(seed)

        n_total = len(X)
        n_val = max(1, int(n_total * self.val_fraction))
        n_tr = n_total - n_val

        X_tr, X_va = X[:n_tr], X[n_tr:]
        y_tr, y_va = y[:n_tr], y[n_tr:]

        seqs_tr = _make_sequences(X_tr, self.lookback).astype(np.float32)
        va_ext = np.vstack([X_tr[-(self.lookback - 1):], X_va])
        seqs_va = _make_sequences_notail(va_ext, self.lookback).astype(np.float32)

        n_features = X_tr.shape[1]
        model = ElecCNNLSTM(
            n_features, self.cnn_channels, self.lstm_hidden,
            self.cnn_layers, self.lstm_layers, self.dropout,
        ).to(CNNLSTM_DEVICE)

        drop_last = len(seqs_tr) % self.batch_size < 2
        ds_tr = TensorDataset(
            torch.from_numpy(seqs_tr).to(CNNLSTM_DEVICE),
            torch.from_numpy(y_tr).to(CNNLSTM_DEVICE),
        )
        loader = DataLoader(ds_tr, batch_size=self.batch_size, shuffle=True, drop_last=drop_last)

        seqs_va_t = torch.from_numpy(seqs_va).to(CNNLSTM_DEVICE)
        y_va_t = torch.from_numpy(y_va).to(CNNLSTM_DEVICE)

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.05, total_iters=self.warmup_epochs
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, self.max_epochs - self.warmup_epochs), eta_min=self.lr / 100
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[self.warmup_epochs]
        )

        criterion = nn.HuberLoss(delta=self.huber_delta) if self.loss == "huber" else nn.MSELoss()
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
                val_loss = val_criterion(model(seqs_va_t), y_va_t).item()

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

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CNNLSTMRegressor":
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        self.models_ = []
        self.best_epochs_ = []

        for i in range(self.n_seeds):
            model, best_ep = self._train_single(X, y, self.random_state + i)
            self.models_.append(model)
            self.best_epochs_.append(best_ep)

        self.best_epoch_ = int(np.median(self.best_epochs_))
        self._X_tail = X[-(self.lookback - 1):].copy()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.models_:
            raise RuntimeError("Call .fit() before .predict()")
        X = np.asarray(X, dtype=np.float32)

        if self._X_tail is not None and len(self._X_tail) > 0:
            X_ext = np.vstack([self._X_tail, X])
            seqs = _make_sequences_notail(X_ext, self.lookback).astype(np.float32)
        else:
            seqs = _make_sequences(X, self.lookback).astype(np.float32)

        seqs_t = torch.from_numpy(seqs).to(CNNLSTM_DEVICE)
        preds_list = []
        for model in self.models_:
            model.eval()
            with torch.no_grad():
                preds_list.append(model(seqs_t).cpu().numpy())

        return np.mean(preds_list, axis=0)


# ══════════════════════════════════════════════════════════════════════════════
# Standalone test block
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"Device: {CNNLSTM_DEVICE}")
    N, F, L = 500, 50, 48
    rng = np.random.default_rng(0)
    X = rng.standard_normal((N, F)).astype(np.float32)
    y = rng.standard_normal(N).astype(np.float32)

    model = CNNLSTMRegressor(
        lookback=L, cnn_channels=16, lstm_hidden=32, cnn_layers=2,
        max_epochs=5, patience=5, n_seeds=2,
    )
    model.fit(X, y)
    preds = model.predict(X[:100])
    assert preds.shape == (100,), f"Expected (100,), got {preds.shape}"

    X_test = rng.standard_normal((20, F)).astype(np.float32)
    preds_test = model.predict(X_test)
    assert preds_test.shape == (20,), f"Expected (20,), got {preds_test.shape}"

    print(f"Sanity check passed. best_epoch={model.best_epoch_} seeds={model.best_epochs_}")
