import logging
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import copy

# Ensure src is in standard path for execution
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from src.data_ingestion import load_and_merge_zone
from src.features import build_features
from src.preprocessing import chronological_train_val_test_split, scale_data
from src.evaluation.metrics import MAE, sMAPE, rMAE, save_metrics_to_csv
from src.constants import TARGET_COL

logger = logging.getLogger(__name__)

INT_PARAMS = [
    "num_leaves",
    "max_depth",
    "n_estimators",
    "min_child_weight",
    "min_samples_leaf",
    "min_samples_split",
    "batch_size",
    "depth",
]


def sanitize_int_params(params: dict) -> dict:
    for param in INT_PARAMS:
        if param in params and params[param] is not None:
            params[param] = int(params[param])
    return params


def reshape_to_daily(X: pd.DataFrame, y: pd.Series, augment: bool = False):
    """
    Groups hourly data into blocks of 24 resolving multivariate prediction nodes natively.
    Drops any partial sequences securely preventing dimensional indexing failures.
    Returns:
        X_daily (N_days, F*24)
        y_daily (N_days, 24)
    """
    X_daily_list = []
    y_daily_list = []

    if not augment:
        X_copy = X.copy()
        X_copy["Date"] = X_copy.index.date
        y_df = pd.DataFrame({"Target": y, "Date": y.index.date})

        for date, group_X in X_copy.groupby("Date"):
            group_y = y_df[y_df["Date"] == date]

            # We strictly maintain mathematically exact 24 hour batches natively avoiding dimension crashes
            if len(group_X) == 24 and len(group_y) == 24:
                x_flat = group_X.drop(columns=["Date"]).values.flatten()
                y_flat = group_y["Target"].values.flatten()

                X_daily_list.append(x_flat)
                y_daily_list.append(y_flat)
    else:
        # Augmentation: rolling 24-hour window mapped linearly at exactly 1-hour strides
        # producing purely inflated sequences without leakage since train is isolated!
        n_samples = len(X)
        X_vals = X.values
        y_vals = y.values
        for i in range(n_samples - 23):
            x_slice = X_vals[i : i + 24]
            y_slice = y_vals[i : i + 24]
            if len(x_slice) == 24 and len(y_slice) == 24:
                X_daily_list.append(x_slice.flatten())
                y_daily_list.append(y_slice.flatten())

    return np.array(X_daily_list, dtype=np.float32), np.array(
        y_daily_list, dtype=np.float32
    )


class EPFMultivariateDNN(nn.Module):
    """
    Architects the identical internal Sequence block built by J. Lago.
    (Linear -> BatchNorm1d -> ReLU -> Dropout) -> Linear(24).
    """

    def __init__(
        self, input_dim: int, hidden_dims=[256, 128], dropout_rate: float = 0.2
    ):
        super(EPFMultivariateDNN, self).__init__()

        layers = []
        last_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_rate))
            last_dim = h_dim

        # Natively outputs precisely a vector of length 24 simultaneously!
        layers.append(nn.Linear(last_dim, 24))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def train_pytorch_dnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: dict = None,
):
    """
    Orchestrates explicit EarlyStopping sequences training deep nodes exclusively
    driven via strictly mapped Absolute Loss (L1) bypassing quadratic variance.
    """
    if params is None:
        params = {}

    epochs = params.get("epochs", 150)
    batch_size = params.get("batch_size", 64)
    lr = params.get("lr", params.get("learning_rate", 0.001))
    patience = params.get("patience", 15)
    dropout_rate = params.get("dropout_rate", 0.2)
    weight_decay = params.get("weight_decay", 0.0)
    seed = params.get("seed", None)

    if seed is not None:
        torch.manual_seed(seed)

    input_dim = X_train.shape[1]
    model = EPFMultivariateDNN(input_dim=input_dim, dropout_rate=dropout_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Jesus Lago proved tracking MSE distorts EPF limits. We enforce MAE.
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    best_model_weights = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            # --- PREVENT EXPLODING GRADIENTS ---
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # -----------------------------------
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_x.size(0)

        val_loss /= len(val_loader.dataset)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Rollback buffer exactly capturing minimum loss parameters natively
            best_model_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.info(
                f"EarlyStopping trigger activated at Epoch {epoch + 1}. Restoring best Val Loss Weights [{best_val_loss:.4f}]."
            )
            break

    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)

    return model, device


def evaluate_dnn(
    model,
    device,
    X_test_daily: np.ndarray,
    y_test_daily: np.ndarray,
    y_scaler,
    y_test_raw: pd.Series,
    zone: str,
    split_name: str = "Test",
):
    """
    Maps 24D Test Tensors locally formatting arrays back to standard Flat sequences evaluating targets.
    """
    model.eval()
    X_test_tensor = torch.tensor(X_test_daily).to(device)

    with torch.no_grad():
        preds_daily = model(X_test_tensor).cpu().numpy()

    # Realign structurally converting explicit 24D predictions purely into unindexed continuous timelines
    y_pred_flat = preds_daily.flatten()

    # Scale predictions natively back into original numerical bounds!
    y_pred_unscaled = y_scaler.inverse_transform(y_pred_flat.reshape(-1, 1)).flatten()

    # Rebuild indices preventing DST offset bounds shifting the relative MAE calculation
    valid_indices = []
    y_df = pd.DataFrame(
        {"Target": y_test_raw.values, "Date": y_test_raw.index.date},
        index=y_test_raw.index,
    )
    for date, group in y_df.groupby("Date"):
        if len(group) == 24:
            valid_indices.extend(group.index)

    y_p_s = pd.Series(y_pred_unscaled, index=valid_indices)

    # Restrict test natively just in case
    y_t_s = y_test_raw.loc[valid_indices]

    mae_score = MAE(y_t_s, y_p_s)
    smape_score = sMAPE(y_t_s, y_p_s) * 100
    rmae_score = rMAE(y_t_s, y_p_s, m="W")

    logger.info(f"[Multivariate PyTorch DNN] MAE:   {mae_score:.3f} EUR/MWh")
    logger.info(f"[Multivariate PyTorch DNN] sMAPE: {smape_score:.3f} %")
    logger.info(f"[Multivariate PyTorch DNN] rMAE:  {rmae_score:.3f}")

    save_metrics_to_csv(
        zone=zone,
        model_name=f"PyTorch DNN ({split_name})",
        metrics_dict={"MAE": mae_score, "sMAPE": smape_score, "rMAE": rmae_score},
    )

    return y_p_s


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    raw_directory = config.get("data", {}).get("raw_dir", "data/raw/auhack_legacy/")

    val_pred_dir = Path("data/outputs/predictions/val")
    test_pred_dir = Path("data/outputs/predictions/test")
    val_pred_dir.mkdir(parents=True, exist_ok=True)
    test_pred_dir.mkdir(parents=True, exist_ok=True)

    base_dnn_params = config.get("model_settings", {}).get("dnn", {}).copy()
    val_split = base_dnn_params.get("validation_split", 0.2)
    use_data_augmentation = base_dnn_params.get("use_data_augmentation", False)

    try:
        with open("best_hyperparameters.yaml", "r") as f:
            best_hyperparams = yaml.safe_load(f) or {}
    except FileNotFoundError:
        best_hyperparams = {}
        logger.warning(
            "best_hyperparameters.yaml not found; falling back to default config.yaml parameters."
        )

    target_zones = config.get("data", {}).get("target_zones", ["DE"])
    flow_only_zones = config["data"].get("flow_only_zones", [])
    all_zones = target_zones + flow_only_zones
    raw_data_dict = {z: load_and_merge_zone(z, raw_directory) for z in all_zones}

    missing_targets = [z for z in target_zones if z not in raw_data_dict]
    if missing_targets:
        raise ValueError(
            f"Missing required target zones in preloaded raw_data_dict: {missing_targets}"
        )

    val_preds_dict_dnn = {}
    test_preds_dict_dnn = {}
    zone_mae_dnn = {}

    for target_zone in target_zones:
        logger.info("========================================")
        logger.info(
            f"Loading Multivariate DNN Evaluation natively for {target_zone}..."
        )

        df, active_features = build_features(
            raw_data_dict, target_zone, lag_actual_flows=True
        )

        logger.info(
            "Splitting & Applying StandardScaler mathematically locking zero node bias leakages..."
        )
        train_df, val_df, test_df = chronological_train_val_test_split(
            df, val_ratio=val_split, test_ratio=0.15
        )

        # 1. Standardizing features is CRITICAL strictly for gradients
        X_train_raw = train_df[active_features]
        y_train_raw = train_df[TARGET_COL]

        X_val_raw = val_df[active_features]
        y_val_raw = val_df[TARGET_COL]

        X_test_raw = test_df[active_features]
        y_test_raw = test_df[TARGET_COL]

        X_train_s, X_val_s, X_test_s, scaler = scale_data(
            X_train_raw, X_val_raw, X_test_raw
        )

        # 2. TARGET SCALING (Crucial for Neural Networks)
        # Temporarily cast Series to singular DataFrames to utilize standardized scale_data efficiently!
        y_train_df = y_train_raw.to_frame()
        y_val_df = y_val_raw.to_frame()
        y_test_df = y_test_raw.to_frame()

        y_train_s_df, y_val_s_df, y_test_s_df, y_scaler = scale_data(
            y_train_df, y_val_df, y_test_df
        )

        # Cast scaling vectors directly back into pandas Series objects identically matching Target format
        y_train_s = y_train_s_df[TARGET_COL]
        y_val_s = y_val_s_df[TARGET_COL]
        y_test_s = y_test_s_df[TARGET_COL]

        # 3. Reshape continuously into (N_days, F*24) inputs optimizing direct multivariate architectures
        logger.info(
            f"Flattening structurally routing explicit 1D DataFrames directly into 24-D Arrays (Augmentation={use_data_augmentation})..."
        )
        X_train_d, y_train_d = reshape_to_daily(
            X_train_s, y_train_s, augment=use_data_augmentation
        )
        X_val_d, y_val_d = reshape_to_daily(X_val_s, y_val_s, augment=False)
        X_test_d, y_test_d = reshape_to_daily(X_test_s, y_test_s, augment=False)

        logger.info(f"Reshaping Array Dimensions Completed:")
        logger.info(
            f" Train: {X_train_d.shape} | Val: {X_val_d.shape} | Test: {X_test_d.shape}"
        )
        logger.info("========================================")

        logger.info(f"Initiating Internal PyTorch Compiler Loop (Adam | L1Loss)...")
        dnn_params_zone = (
            best_hyperparams.get("PyTorch DNN", {}).get(target_zone, base_dnn_params)
            or base_dnn_params
        ).copy()
        dnn_params_zone = sanitize_int_params(dnn_params_zone)
        model, device = train_pytorch_dnn(
            X_train_d, y_train_d, X_val_d, y_val_d, params=dnn_params_zone
        )

        val_preds_dnn = evaluate_dnn(
            model,
            device,
            X_val_d,
            y_val_d,
            y_scaler,
            y_val_raw,
            zone=target_zone,
            split_name="Validation",
        )
        test_preds_dnn = evaluate_dnn(
            model,
            device,
            X_test_d,
            y_test_d,
            y_scaler,
            y_test_raw,
            zone=target_zone,
            split_name="Test",
        )

        val_preds_dict_dnn[target_zone] = val_preds_dnn
        test_preds_dict_dnn[target_zone] = test_preds_dnn
        y_test_aligned = y_test_raw.loc[test_preds_dnn.index]
        zone_mae_dnn[target_zone] = MAE(y_test_aligned, test_preds_dnn)

        logger.info("========================================")
        logger.info("--------- Evaluation Metrics -----------")
        _ = test_preds_dnn
        logger.info("========================================")

    # Save multi-zone predictions as DataFrames
    pd.DataFrame(val_preds_dict_dnn).to_csv(val_pred_dir / "dnn.csv")
    pd.DataFrame(test_preds_dict_dnn).to_csv(test_pred_dir / "dnn.csv")

    # Macro-averages
    logger.info("\n========== MACRO AVERAGES ACROSS ZONES ==========")
    if zone_mae_dnn:
        avg_mae_dnn = np.mean(list(zone_mae_dnn.values()))
        logger.info(f"[DNN] Avg MAE: {avg_mae_dnn:.3f} EUR/MWh")
    logger.info("================================================")
