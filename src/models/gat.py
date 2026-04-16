"""GATConv model for electricity price forecasting.

Architecture: NodeEncoder → GATConv(multi-head) → ELU → GATConv(single-head) → Linear
Operates on daily graphs: 24 nodes (hours), ring edges + h↔h+12 skip connections.
"""

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import GATConv

from .dnn import DNN_DEVICE


def _build_edge_index():
    """Build edge index for 24-node hourly ring graph + h↔h+12 skip."""
    edges = []
    for i in range(24):
        j_next = (i + 1) % 24
        j_skip = (i + 12) % 24
        edges.extend([[i, j_next], [j_next, i],
                       [i, j_skip], [j_skip, i]])
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


EDGE_INDEX_24 = _build_edge_index()


def _hourly_to_daily_graphs(X, y, hours):
    """Convert hourly arrays to list of daily PyG Data objects.

    Groups consecutive hours into complete 24h blocks (h=0..23).
    NaN targets are masked via data.mask.
    """
    graphs = []
    indices_map = []
    n = len(X)
    i = 0
    while i < n and hours[i] != 0:
        i += 1
    while i + 24 <= n:
        if np.array_equal(hours[i:i + 24], np.arange(24)):
            x_day = torch.FloatTensor(X[i:i + 24])
            if y is not None:
                y_day = torch.FloatTensor(np.nan_to_num(y[i:i + 24], nan=0.0))
                valid = torch.BoolTensor(np.isfinite(y[i:i + 24]))
            else:
                y_day = None
                valid = None
            data = Data(x=x_day, edge_index=EDGE_INDEX_24, y=y_day, mask=valid)
            graphs.append(data)
            indices_map.append(list(range(i, i + 24)))
            i += 24
        else:
            i += 1
    return graphs, indices_map


class ElecGAT(nn.Module):
    """Graph Attention Network for electricity price forecasting."""

    def __init__(self, n_features, hidden=64, heads=4, dropout=0.2):
        super().__init__()
        self.n_features = n_features
        self.hidden = hidden
        self.heads = heads
        self.dropout_rate = dropout
        self.node_encoder = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.BatchNorm1d(hidden),
            nn.ELU(),
        )
        self.gat1 = GATConv(hidden, hidden, heads=heads, dropout=dropout)
        self.elu = nn.ELU()
        self.gat2 = GATConv(hidden * heads, hidden, heads=1, dropout=dropout)
        self.output = nn.Linear(hidden, 1)

    def forward(self, x, edge_index, batch=None):
        h = self.node_encoder(x)
        h = self.gat1(h, edge_index)
        h = self.elu(h)
        h = self.gat2(h, edge_index)
        h = self.elu(h)
        return self.output(h).squeeze(-1)


def train_gat(X_tr, y_tr, hours_tr, X_va, y_va, hours_va,
              hidden=64, heads=4, dropout=0.2,
              lr=1e-3, wd=1e-4, bs=32,
              max_epochs=300, patience=30, criterion=None):
    """Train GATConv with early stopping on validation loss.

    Pass ALL hours (not just valid ones). NaN targets are masked in loss.
    Returns (model, n_epochs).
    """
    if criterion is None:
        criterion = nn.HuberLoss(delta=5.0)

    train_graphs, _ = _hourly_to_daily_graphs(X_tr, y_tr, hours_tr)
    val_graphs, _ = _hourly_to_daily_graphs(X_va, y_va, hours_va)

    if not train_graphs:
        raise ValueError(f"No complete days in training data ({len(X_tr)} hours)")
    if not val_graphs:
        raise ValueError(f"No complete days in validation data ({len(X_va)} hours)")

    n_features = X_tr.shape[1]
    model = ElecGAT(n_features, hidden=hidden, heads=heads, dropout=dropout).to(DNN_DEVICE)

    train_loader = PyGDataLoader(train_graphs, batch_size=bs, shuffle=True)
    val_loader = PyGDataLoader(val_graphs, batch_size=len(val_graphs), shuffle=False)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5, min_lr=1e-6)
    best_loss, best_state, no_imp = float("inf"), None, 0

    for ep in range(max_epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(DNN_DEVICE)
            opt.zero_grad()
            preds = model(batch.x, batch.edge_index, batch.batch)
            mask = batch.mask
            if mask is not None and mask.any():
                loss = criterion(preds[mask], batch.y[mask])
            elif mask is not None:
                continue  # skip batch with no valid targets
            else:
                loss = criterion(preds, batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            vl_sum, vl_count = 0.0, 0
            for batch in val_loader:
                batch = batch.to(DNN_DEVICE)
                preds = model(batch.x, batch.edge_index, batch.batch)
                mask = batch.mask
                if mask is not None and mask.any():
                    vl_sum += criterion(preds[mask], batch.y[mask]).item()
                    vl_count += 1
                elif mask is None:
                    vl_sum += criterion(preds, batch.y).item()
                    vl_count += 1
            vl = vl_sum / max(1, vl_count)

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


def retrain_gat(n_features, X_full, y_full, hours_full, n_epochs,
                hidden=64, heads=4, dropout=0.2,
                lr=1e-3, wd=1e-4, bs=32, criterion=None):
    """Retrain GATConv on full data for fixed number of epochs. Returns model."""
    if criterion is None:
        criterion = nn.HuberLoss(delta=5.0)

    graphs, _ = _hourly_to_daily_graphs(X_full, y_full, hours_full)
    model = ElecGAT(n_features, hidden=hidden, heads=heads, dropout=dropout).to(DNN_DEVICE)
    loader = PyGDataLoader(graphs, batch_size=bs, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    for _ in range(n_epochs):
        model.train()
        for batch in loader:
            batch = batch.to(DNN_DEVICE)
            opt.zero_grad()
            preds = model(batch.x, batch.edge_index, batch.batch)
            mask = batch.mask
            if mask is not None and mask.any():
                loss = criterion(preds[mask], batch.y[mask])
            elif mask is not None:
                continue  # skip batch with no valid targets
            else:
                loss = criterion(preds, batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()

    model.eval()
    return model


def predict_gat(model, X, hours):
    """Predict with GATConv model. Returns hourly predictions array.

    Handles incomplete days at boundaries via node_encoder fallback.
    """
    model.eval()
    graphs, indices_map = _hourly_to_daily_graphs(X, None, hours)
    preds = np.full(len(X), np.nan)

    if graphs:
        loader = PyGDataLoader(graphs, batch_size=64, shuffle=False)
        all_preds = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(DNN_DEVICE)
                bp = model(batch.x, batch.edge_index, batch.batch).cpu().numpy()
                all_preds.append(bp)
        batch_preds = np.concatenate(all_preds)

        offset = 0
        for graph_indices in indices_map:
            for j, orig_idx in enumerate(graph_indices):
                preds[orig_idx] = batch_preds[offset + j]
            offset += len(graph_indices)

    # Fallback: hours not covered by complete days → node encoder only
    nan_mask = np.isnan(preds)
    if nan_mask.any():
        with torch.no_grad():
            x_fb = torch.FloatTensor(X[nan_mask]).to(DNN_DEVICE)
            h = model.node_encoder(x_fb)
            preds[nan_mask] = model.output(h).squeeze(-1).cpu().numpy()

    return preds
