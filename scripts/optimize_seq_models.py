"""Optuna hyperparameter optimization for 5 sequence models × 2 countries.

Saves trial history live to outputs/optuna_trials.json (resumable).
Updates outputs/best_parameters.yaml after every completed trial.

Runtime budget: ~2 hours on a single GPU (20 trials × 2 models × 2 countries).

Usage:
    cd "INCOMO 3" && python scripts/optimize_seq_models.py
    # To resume after interruption, just re-run — trials are persisted.
"""

from __future__ import annotations

import json
import sys
import time
import warnings

import numpy as np
import optuna
import pandas as pd
import yaml
from pathlib import Path
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, ".")
from src.data_loading import load_data
from src.feature_engineering import build_features
from src.models import compute_rmse, prepare_stationary
from src.models.cnn_lstm import CNNLSTMRegressor
from src.models.encoder_decoder import EncoderDecoderRegressor

warnings.filterwarnings("ignore", category=UserWarning)

# ── Paths ────────────────────────────────────────────────────────────────────
BEST_PARAMS_PATH = Path("outputs/best_parameters.yaml")
TRIALS_PATH = Path("outputs/optuna_trials.json")
DB_PATH = Path("outputs/optuna_study.db")

# ── Tuning budget ────────────────────────────────────────────────────────────
N_TRIALS = 5  # per (model, country) pair → 20 × 2 × 2 = 80 total
TUNE_MAX_EPOCHS = 50  # faster than full 300 during search
TUNE_PATIENCE = 15
TUNE_VAL_FRAC = 0.1
SEQ_LOOKBACK = 48

MODELS = ["CNNLSTM", "ENCDEC"]
COUNTRIES = ["uk", "fr"]


# ══════════════════════════════════════════════════════════════════════════════
# Data loading (same as pipeline v11 stages 0-3d)
# ══════════════════════════════════════════════════════════════════════════════


def load_and_prepare_data():
    """Load data and return scaled arrays + targets for both countries."""
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    x_train, y_train, x_test = load_data("data/raw")
    train_fe = build_features(pd.concat([x_train], axis=0), config)
    train_fe = train_fe.join(y_train[["fr_spot", "uk_spot"]])

    holdout_start = config["validation"]["holdout_start"]
    mask_val = train_fe["datetime_CET"] >= holdout_start
    df_train = train_fe[~mask_val].copy()
    df_val = train_fe[mask_val].copy()
    df_train_uk = df_train.copy()

    # Feature list: all numeric, deduped at corr > 0.99
    _EXCLUDE = {"fr_spot", "uk_spot", "datetime_CET", "datetime_UTC", "date", "id"}
    _all_num = [
        c
        for c in df_train.columns
        if c not in _EXCLUDE
        and df_train[c].dtype in ["float64", "float32", "int64", "int32"]
        and df_train[c].notna().sum() > len(df_train) * 0.5
    ]
    _corr = df_train[_all_num].corr().abs()
    _to_drop = set()
    for _i in range(len(_all_num)):
        if _all_num[_i] in _to_drop:
            continue
        for _j in range(_i + 1, len(_all_num)):
            if _all_num[_j] in _to_drop:
                continue
            if _corr.iloc[_i, _j] > 0.99:
                _to_drop.add(_all_num[_j])
    feat_dnn_final = [f for f in _all_num if f not in _to_drop]

    # FR data
    fr_stat = prepare_stationary("fr_spot_la", "fr_spot", train_fe, df_train, df_val)
    dnn_scaler_fr = StandardScaler()
    X_dnn_tr_fr = dnn_scaler_fr.fit_transform(
        np.nan_to_num(df_train[feat_dnn_final].values, 0)
    )
    X_dnn_va_fr = dnn_scaler_fr.transform(
        np.nan_to_num(df_val[feat_dnn_final].values, 0)
    )
    X_tr_fr = X_dnn_tr_fr[fr_stat["valid_tr"]]
    y_tr_fr = fr_stat["y_dev_tr"][fr_stat["valid_tr"]].astype(np.float32)
    rm_va_fr = fr_stat["rm_va"]
    spot_va_fr = fr_stat["spot_va"]

    # UK data
    uk_spot_tr = df_train_uk["uk_spot"].values
    uk_spot_va = df_val["uk_spot"].values
    uk_moc_tr = df_train_uk["uk_merit_order_cost"].values
    uk_moc_va = df_val["uk_merit_order_cost"].values
    y_basis_tr = uk_spot_tr - uk_moc_tr
    valid_basis_tr = np.isfinite(y_basis_tr)

    dnn_scaler_uk = StandardScaler()
    X_dnn_tr_uk = dnn_scaler_uk.fit_transform(
        np.nan_to_num(df_train_uk[feat_dnn_final].values, 0)
    )
    X_dnn_va_uk = dnn_scaler_uk.transform(
        np.nan_to_num(df_val[feat_dnn_final].values, 0)
    )
    X_tr_uk = X_dnn_tr_uk[valid_basis_tr]
    y_tr_uk = y_basis_tr[valid_basis_tr].astype(np.float32)

    return {
        "fr": {
            "X_tr": X_tr_fr,
            "y_tr": y_tr_fr,
            "X_va": X_dnn_va_fr,
            "rm_va": rm_va_fr,
            "spot_va": spot_va_fr,
        },
        "uk": {
            "X_tr": X_tr_uk,
            "y_tr": y_tr_uk,
            "X_va": X_dnn_va_uk,
            "rm_va": uk_moc_va,
            "spot_va": uk_spot_va,
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# Search spaces per model
# ══════════════════════════════════════════════════════════════════════════════


def _common_params(trial: optuna.Trial, country: str) -> dict:
    """Shared hyperparameters across all models."""
    huber_delta = trial.suggest_float("huber_delta", 2.0, 20.0, log=True)
    return {
        "lookback": SEQ_LOOKBACK,
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
        "max_epochs": TUNE_MAX_EPOCHS,
        "patience": TUNE_PATIENCE,
        "val_fraction": TUNE_VAL_FRAC,
        "warmup_epochs": trial.suggest_int("warmup_epochs", 3, 15),
        "grad_clip": 1.0,
        "n_seeds": 1,
        "random_state": 42,
        "loss": "huber",
        "huber_delta": huber_delta,
    }


def suggest_cnnlstm(trial: optuna.Trial, country: str) -> CNNLSTMRegressor:
    common = _common_params(trial, country)
    return CNNLSTMRegressor(
        cnn_channels=trial.suggest_categorical("cnn_channels", [32, 64, 128]),
        lstm_hidden=trial.suggest_categorical("lstm_hidden", [64, 128, 256]),
        cnn_layers=trial.suggest_int("cnn_layers", 1, 3),
        lstm_layers=trial.suggest_int("lstm_layers", 1, 2),
        dropout=trial.suggest_float("dropout", 0.05, 0.5),
        lr=trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        **common,
    )


def suggest_encdec(trial: optuna.Trial, country: str) -> EncoderDecoderRegressor:
    common = _common_params(trial, country)
    return EncoderDecoderRegressor(
        enc_hidden=trial.suggest_categorical("enc_hidden", [64, 128, 256]),
        dec_hidden=trial.suggest_categorical("dec_hidden", [32, 64, 128]),
        enc_layers=trial.suggest_int("enc_layers", 1, 3),
        dropout=trial.suggest_float("dropout", 0.05, 0.5),
        lr=trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        **common,
    )


MODEL_SUGGEST = {
    "CNNLSTM": suggest_cnnlstm,
    "ENCDEC": suggest_encdec,
}


# ══════════════════════════════════════════════════════════════════════════════
# Trial persistence
# ══════════════════════════════════════════════════════════════════════════════


def _load_trials() -> list[dict]:
    if TRIALS_PATH.exists():
        with open(TRIALS_PATH) as f:
            return json.load(f)
    return []


def _save_trial(trial_record: dict) -> None:
    """Append a trial record and write atomically."""
    trials = _load_trials()
    trials.append(trial_record)
    tmp = TRIALS_PATH.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(trials, f, indent=2)
    tmp.replace(TRIALS_PATH)


def _update_best_params(
    model_name: str, country: str, params: dict, rmse: float
) -> None:
    """Update best_parameters.yaml if this trial is the new best."""
    bp = yaml.safe_load(BEST_PARAMS_PATH.read_text())

    current = bp.get(country, {}).get(model_name, {})
    current_rmse = current.get("_best_rmse", float("inf"))

    if rmse < current_rmse:
        # Build clean param dict (exclude internal/shared keys)
        _SHARED_KEYS = {
            "lookback",
            "batch_size",
            "max_epochs",
            "patience",
            "val_fraction",
            "warmup_epochs",
            "grad_clip",
            "n_seeds",
            "random_state",
            "loss",
            "huber_delta",
        }
        clean = {k: v for k, v in params.items() if k not in _SHARED_KEYS}
        clean["_best_rmse"] = round(rmse, 4)

        # Also update shared keys that vary
        bp[country]["huber_delta"] = params.get(
            "huber_delta", bp[country].get("huber_delta")
        )

        bp[country][model_name] = clean

        tmp = BEST_PARAMS_PATH.with_suffix(".tmp")
        with open(tmp, "w") as f:
            yaml.dump(bp, f, default_flow_style=False, sort_keys=False)
        tmp.replace(BEST_PARAMS_PATH)
        print(f"    ✓ New best {country.upper()} {model_name}: RMSE={rmse:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# Objective factory
# ══════════════════════════════════════════════════════════════════════════════


def make_objective(model_name: str, country: str, data: dict):
    """Return an Optuna objective function for a (model, country) pair."""
    d = data[country]

    def objective(trial: optuna.Trial) -> float:
        t0 = time.time()
        try:
            model = MODEL_SUGGEST[model_name](trial, country)
            model.fit(d["X_tr"], d["y_tr"])

            preds = d["rm_va"] + model.predict(d["X_va"])
            rmse = compute_rmse(d["spot_va"], preds)

            elapsed = time.time() - t0
            print(
                f"  [{country.upper()} {model_name}] trial {trial.number}: RMSE={rmse:.4f} ({elapsed:.0f}s)"
            )

            # Save trial record
            _save_trial(
                {
                    "model": model_name,
                    "country": country,
                    "trial": trial.number,
                    "rmse": round(float(rmse), 4),
                    "best_epoch": model.best_epoch_,
                    "params": trial.params,
                    "elapsed_s": round(elapsed, 1),
                }
            )

            # Update best params if improved
            _update_best_params(model_name, country, trial.params, float(rmse))

            return rmse

        except Exception as e:
            print(
                f"  [{country.upper()} {model_name}] trial {trial.number} FAILED: {e}"
            )
            return float("inf")

    return objective


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════


def main():
    print("=" * 90)
    print("  Sequence Model Hyperparameter Optimization (Optuna)")
    print(f"  {N_TRIALS} trials × {len(MODELS)} models × {len(COUNTRIES)} countries")
    print(f"  Tuning epochs: {TUNE_MAX_EPOCHS}, patience: {TUNE_PATIENCE}")
    print("=" * 90)

    t_start = time.time()

    print("\nLoading data...")
    data = load_and_prepare_data()
    print(f"  Data loaded in {time.time() - t_start:.0f}s")
    print(f"  FR: X_tr={data['fr']['X_tr'].shape}, X_va={data['fr']['X_va'].shape}")
    print(f"  UK: X_tr={data['uk']['X_tr'].shape}, X_va={data['uk']['X_va'].shape}")

    storage = f"sqlite:///{DB_PATH.as_posix()}"

    for model_name in MODELS:
        for country in COUNTRIES:
            study_name = f"{model_name}_{country}"
            print(f"\n{'─' * 70}")
            print(f"  Optimizing {country.upper()} {model_name} ({N_TRIALS} trials)")
            print(f"{'─' * 70}")

            study = optuna.create_study(
                study_name=study_name,
                storage=storage,
                direction="minimize",
                load_if_exists=True,
            )

            n_existing = len(study.trials)
            n_remaining = max(0, N_TRIALS - n_existing)
            if n_remaining == 0:
                print(f"  Already completed {n_existing} trials, skipping.")
                continue

            if n_existing > 0:
                print(f"  Resuming: {n_existing} trials done, {n_remaining} remaining.")

            objective = make_objective(model_name, country, data)
            study.optimize(
                objective,
                n_trials=n_remaining,
                show_progress_bar=False,
            )

            best = study.best_trial
            print(f"  Best: RMSE={best.value:.4f}, params={best.params}")

    elapsed = time.time() - t_start
    print(f"\n{'=' * 90}")
    print(f"  Optimization complete in {elapsed / 3600:.1f}h ({elapsed:.0f}s)")
    print(f"  Results: {TRIALS_PATH}")
    print(f"  Best params: {BEST_PARAMS_PATH}")
    print(f"{'=' * 90}")


if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    main()
