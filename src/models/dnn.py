"""DNN model for electricity price forecasting."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Device selection
if torch.backends.mps.is_available():
    DNN_DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DNN_DEVICE = torch.device("cuda")
else:
    DNN_DEVICE = torch.device("cpu")


class ElecDNN(nn.Module):
    """Dense NN for electricity price forecasting (epftoolbox-inspired)."""
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


def train_dnn(model, X_tr, y_tr, X_va, y_va, lr=1e-3, wd=1e-4, bs=256,
              max_epochs=500, patience=30, criterion=None):
    """Train DNN with early stopping on validation loss."""
    if criterion is None:
        criterion = nn.HuberLoss(delta=5.0)
    model = model.to(DNN_DEVICE)
    ds = TensorDataset(torch.FloatTensor(X_tr).to(DNN_DEVICE),
                       torch.FloatTensor(y_tr).to(DNN_DEVICE))
    loader = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=len(X_tr) % bs == 1)
    X_va_t = torch.FloatTensor(X_va).to(DNN_DEVICE)
    y_va_t = torch.FloatTensor(y_va).to(DNN_DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5, min_lr=1e-6)
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


def predict_dnn(model, X):
    """Predict with DNN model."""
    model.eval()
    with torch.no_grad():
        return model(torch.FloatTensor(X).to(DNN_DEVICE)).cpu().numpy()
