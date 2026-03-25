"""CNN-LSTM hybrid model for electricity price forecasting.

Architecture:
  1. CNN stage: 1-2 Conv1d blocks extract local temporal features across
     the lookback window (short-range patterns: hourly ramps, spike onset).
  2. LSTM stage: captures sequential dependencies across the lookback
     (daily cycle, multi-hour regime shifts).

Spike-detection focus:
  - Lookback 24h (1 day) covers the onset patterns that precede spikes
    (nuclear trip → high load hour, gas squeeze → peak hours).
  - Small architecture (cnn_channels=32, lstm_hidden=64) keeps inference fast.
  - On-the-fly sequence generation in the DataLoader: O(N×F) memory instead
    of O(N×lookback×F) — eliminates the ~800 MB upfront allocation.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.base import BaseEstimator, RegressorMixin

# ── Device selection ──────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    CNNLSTM_DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    CNNLSTM_DEVICE = torch.device("cuda")
else:
    CNNLSTM_DEVICE = torch.device("cpu")


# ── On-the-fly sequence Dataset ───────────────────────────────────────────────

class _SeqDataset(Dataset):
    """Generates (lookback, F) windows on the fly — avoids precomputing all seqs.

    First (lookback-1) samples are zero-padded from the left.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, lookback: int) -> None:
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.lookback = lookback
        self.n_features = X.shape[1]

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = i - self.lookback + 1
        if start >= 0:
            seq = self.X[start : i + 1]
        else:
            pad = torch.zeros(-start, self.n_features)
            seq = torch.cat([pad, self.X[: i + 1]], dim=0)
        return seq, self.y[i]


# ── Predict helper: precompute small batch for inference ─────────────────────

def _make_sequences_notail(X_ext: np.ndarray, lookback: int) -> np.ndarray:
    """(N + lookback - 1, F) → (N, lookback, F). Tail prepended by caller."""
    N = len(X_ext) - lookback + 1
    return np.stack([X_ext[i : i + lookback] for i in range(N)])


# ══════════════════════════════════════════════════════════════════════════════
# 1. PyTorch Module
# ══════════════════════════════════════════════════════════════════════════════

class ElecCNNLSTM(nn.Module):
    """CNN feature extractor followed by LSTM temporal modelling.

    forward(x): (B, T, F) → (B,)

    CNN operates on (B, F, T) (channels-first), then transposed back to
    (B, T, cnn_channels) for the LSTM.
    """

    def __init__(
        self,
        input_size: int,
        cnn_channels: int = 32,
        lstm_hidden: int = 64,
        cnn_layers: int = 1,
        lstm_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

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
        x = x.transpose(1, 2)          # (B, F, T)
        x = self.cnn(x)                 # (B, cnn_channels, T)
        x = x.transpose(1, 2)          # (B, T, cnn_channels)
        out, _ = self.lstm(x)           # (B, T, lstm_hidden)
        last = out[:, -1, :]            # (B, lstm_hidden)
        return self.fc(self.norm(last)).squeeze(-1)  # (B,)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Scikit-Learn compatible wrapper
# ══════════════════════════════════════════════════════════════════════════════

class CNNLSTMRegressor(BaseEstimator, RegressorMixin):
    """Scikit-Learn wrapper for ElecCNNLSTM.

    Lighter defaults vs v10:
      - lookback 48 → 24  (spike onset visible in last 24h)
      - cnn_channels 64 → 32, lstm_hidden 128 → 64, cnn_layers 2 → 1
      - max_epochs 300 → 150, patience 30 → 20
      - Training uses on-the-fly DataLoader: ~50x less peak RAM

    Set use_spike_features=True (default False) to let the pipeline pass a
    reduced feature matrix focused on spike drivers (nuclear/gas/residual load).
    """

    def __init__(
        self,
        lookback: int = 24,
        cnn_channels: int = 32,
        lstm_hidden: int = 64,
        cnn_layers: int = 1,
        lstm_layers: int = 1,
        dropout: float = 0.2,
        lr: float = 5e-4,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        max_epochs: int = 150,
        patience: int = 20,
        val_fraction: float = 0.1,
        loss: str = "huber",
        huber_delta: float = 5.0,
        warmup_epochs: int = 5,
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

        # Training: on-the-fly Dataset — no large sequence preallocation
        ds_tr = _SeqDataset(X_tr, y_tr, self.lookback)
        drop_last = len(ds_tr) % self.batch_size < 2
        loader = DataLoader(ds_tr, batch_size=self.batch_size, shuffle=True, drop_last=drop_last)

        # Validation: precompute once (small — val_fraction*N rows)
        va_ext = np.vstack([X_tr[-(self.lookback - 1):], X_va])
        seqs_va = _make_sequences_notail(va_ext, self.lookback).astype(np.float32)
        seqs_va_t = torch.from_numpy(seqs_va).to(CNNLSTM_DEVICE)
        y_va_t = torch.from_numpy(y_va.astype(np.float32)).to(CNNLSTM_DEVICE)

        n_features = X_tr.shape[1]
        model = ElecCNNLSTM(
            n_features, self.cnn_channels, self.lstm_hidden,
            self.cnn_layers, self.lstm_layers, self.dropout,
        ).to(CNNLSTM_DEVICE)

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
                xb = xb.to(CNNLSTM_DEVICE)
                yb = yb.to(CNNLSTM_DEVICE)
                optimizer.zero_grad()
                loss_val = criterion(model(xb), yb)
                loss_val.backward()
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
            n = len(X)
            seqs = np.stack([
                np.vstack([np.zeros((max(0, self.lookback - 1 - i), X.shape[1]), dtype=np.float32),
                            X[max(0, i - self.lookback + 1): i + 1]])
                for i in range(n)
            ])

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
    N, F, L = 500, 50, 24
    rng = np.random.default_rng(0)
    X = rng.standard_normal((N, F)).astype(np.float32)
    y = rng.standard_normal(N).astype(np.float32)

    model = CNNLSTMRegressor(
        lookback=L, cnn_channels=16, lstm_hidden=32, cnn_layers=1,
        max_epochs=5, patience=5, n_seeds=2,
    )
    model.fit(X, y)
    preds = model.predict(X[:100])
    assert preds.shape == (100,), f"Expected (100,), got {preds.shape}"

    X_test = rng.standard_normal((20, F)).astype(np.float32)
    preds_test = model.predict(X_test)
    assert preds_test.shape == (20,), f"Expected (20,), got {preds_test.shape}"

    print(f"Sanity check passed. best_epoch={model.best_epoch_} seeds={model.best_epochs_}")
