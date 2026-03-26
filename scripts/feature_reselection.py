"""Unified feature re-selection for FR and UK on current targets.

Motivation
----------
FR (28 feats): originally selected via Boruta+SHAP on EMA(240h) target.
UK (150 feats): selected via ad-hoc research process, no strict pruning.
Both selections used different methods and different targets than what
the pipeline now uses (FR: STL(168), UK: basis=spot-MOC).

This script applies the same systematic process to both:
  1. Build eligible feature pool (all numeric, >50% non-null)
  2. Train CatBoost on full pool (Optuna params for each market)
  3. Rank by mean |SHAP| on training data
  4. Test union subsets: SHAP-top-N merged with current features
  5. Pick the N that minimises val HBC
  6. Save to data/outputs/feature_selection_v6_fr.json
                data/outputs/feature_selection_v6_uk.json

Usage:
  python scripts/feature_reselection.py              # both markets
  python scripts/feature_reselection.py --fr-only
  python scripts/feature_reselection.py --uk-only
"""

import sys, os, json, time, warnings, argparse
import numpy as np
import yaml

sys.path.insert(0, ".")
from src.data_loading import load_data
from src.feature_engineering import build_features
from src.models import compute_rmse, compute_hbc
from src.models.targets import prepare_stationary
from src.models.tree_models import train_tree, predict_tree

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--fr-only", action="store_true")
parser.add_argument("--uk-only", action="store_true")
args = parser.parse_args()

# ── Data ──────────────────────────────────────────────────────────────────
with open("config.yaml") as f:
    config = yaml.safe_load(f)

print("Loading data + features...")
t0 = time.time()
x_train, y_train, x_test = load_data("data/raw")
train_fe = build_features(x_train.copy(), config)
train_fe = train_fe.join(y_train[["fr_spot", "uk_spot"]])

holdout_start = config["validation"]["holdout_start"]
mask_val = train_fe["datetime_CET"] >= holdout_start
df_train = train_fe[~mask_val].copy()
df_val   = train_fe[mask_val].copy()

for df in [df_train, df_val]:
    if "fr_spot_la_roll_168h_mean" in df.columns and "uk_price_per_mw_7d" in df.columns:
        df["X_fr_spot_la_roll_168h_mean_x_uk_price_per_mw_7d"] = (
            df["fr_spot_la_roll_168h_mean"] * df["uk_price_per_mw_7d"]
        )
print(f"  Done in {time.time()-t0:.0f}s  train={len(df_train)} val={len(df_val)}")

# ── Load Optuna params ─────────────────────────────────────────────────────
RETUNE_PATH = "data/outputs/optuna_catboost_retune.json"
_retune = {}
if os.path.exists(RETUNE_PATH):
    with open(RETUNE_PATH) as f:
        _retune = json.load(f)
    print(f"  Loaded Optuna params from {RETUNE_PATH}")
else:
    print("  Optuna retune JSON not found — using hardcoded params from this session")

def _cb_params(market: str) -> dict:
    """Return CatBoost params for market ('fr' or 'uk')."""
    fixed = {"eval_metric": "RMSE", "iterations": 15000,
             "random_seed": 42, "verbose": 0,
             "allow_writing_files": False, "use_best_model": True}
    if market in _retune:
        p = _retune[market]["params"]
        return {**p, **fixed}
    # Fallback: hardcoded from this session's Optuna run
    if market == "fr":
        return {"loss_function": "Huber:delta=15", "learning_rate": 0.01478,
                "depth": 7, "l2_leaf_reg": 0.931, "subsample": 0.751,
                "colsample_bylevel": 0.104, "min_child_samples": 42,
                "random_strength": 2.836, **fixed}
    else:  # uk — baseline params (Optuna UK result not yet known)
        return {"loss_function": "MAE", "learning_rate": 0.03,
                "depth": 8, "l2_leaf_reg": 5.0, "subsample": 0.8,
                "colsample_bylevel": 0.8, "min_child_samples": 20,
                "random_strength": 1.0, **fixed}

# ── Eligible feature pool (shared) ────────────────────────────────────────
_EXCLUDE = {"fr_spot", "uk_spot", "datetime_CET", "datetime_UTC", "date", "id"}
pool_all = [
    c for c in df_train.columns
    if c not in _EXCLUDE
    and df_train[c].dtype in ["float64", "float32", "int64", "int32", "int8"]
    and df_train[c].notna().mean() > 0.5
]
print(f"  Eligible feature pool: {len(pool_all)} features\n")


# ── Core reselection function ──────────────────────────────────────────────
def run_reselection(
    market: str,
    X_tr_pool, y_tr, w_tr,
    X_va_pool, y_va,
    spot_va, rm_va_or_moc_va,
    hours_va,
    current_features: list[str],
    cb_params: dict,
    out_path: str,
    use_basis: bool = False,  # True for UK (add moc back), False for FR (add rm)
):
    print("=" * 65)
    print(f"  {market.upper()} RE-SELECTION")
    print("=" * 65)

    # Baseline with current features
    feat_avail = [f for f in current_features if f in df_train.columns]
    cb_base = train_tree("catboost", cb_params,
                         df_train.loc[df_train.index[valid_tr], feat_avail], y_tr,
                         df_val.loc[df_val.index[valid_va], feat_avail], y_va,
                         sample_weight=w_tr)

    if use_basis:
        pred_base = rm_va_or_moc_va + predict_tree(cb_base.model,
            df_val.loc[df_val.index[valid_va], feat_avail].values)
    else:
        pred_base = rm_va_or_moc_va + predict_tree(cb_base.model,
            df_val.loc[df_val.index[valid_va], feat_avail].values)

    _, hbc_base = compute_hbc(pred_base, spot_va, hours_va)
    rmse_base = compute_rmse(spot_va, pred_base)
    print(f"  Current ({len(feat_avail)} feats): RMSE={rmse_base:.2f}  HBC={hbc_base:.2f}  iter={cb_base.best_iteration}")

    # Train on full pool for SHAP
    print(f"  Training on full pool ({len(pool_all)} feats) for SHAP...")
    t1 = time.time()
    cb_full = train_tree("catboost", cb_params,
                         df_train.loc[df_train.index[valid_tr], pool_all], y_tr,
                         df_val.loc[df_val.index[valid_va], pool_all], y_va,
                         sample_weight=w_tr)
    print(f"  Full pool: iter={cb_full.best_iteration}  elapsed={time.time()-t1:.0f}s")

    from catboost import Pool as CatPool
    shap_vals = cb_full.model.get_feature_importance(
        CatPool(df_train.loc[df_train.index[valid_tr], pool_all],
                label=y_tr, weight=w_tr),
        type="ShapValues"
    )
    mean_abs_shap = np.abs(shap_vals[:, :-1]).mean(axis=0)
    ranking = sorted(zip(pool_all, mean_abs_shap), key=lambda x: x[1], reverse=True)

    print(f"\n  Top 20 by mean |SHAP|  (<< = new vs current selection):")
    for i, (feat, imp) in enumerate(ranking[:20], 1):
        marker = " <<" if feat not in set(current_features) else "  "
        print(f"    {i:2d}. {feat:<55} {imp:.4f}{marker}")

    # Sweep subsets
    print(f"\n  Subset sweep (union with current {len(feat_avail)} features):")
    best_result = {"hbc": hbc_base, "rmse": rmse_base, "feats": feat_avail,
                   "n": len(feat_avail), "top_n": 0, "mode": "baseline"}

    n_current = len(feat_avail)
    candidates = sorted(set(
        [20, 25, 30, 40, 50, 60, 80] +
        ([20, 25, 30] if market == "fr" else [60, 80, 100, 120])
    ))

    # Mode A: pure top-N (SHAP replacement — lets high-value new features in cleanly)
    print("\n  Mode A -- pure SHAP top-N (replacement):")
    for top_n in candidates:
        avail = [f for f, _ in ranking[:top_n] if f in df_train.columns]
        if len(avail) < 5:
            continue
        cb_n = train_tree("catboost", cb_params,
                          df_train.loc[df_train.index[valid_tr], avail], y_tr,
                          df_val.loc[df_val.index[valid_va], avail], y_va,
                          sample_weight=w_tr)
        pred_n = rm_va_or_moc_va + predict_tree(cb_n.model,
                     df_val.loc[df_val.index[valid_va], avail].values)
        _, hbc_n = compute_hbc(pred_n, spot_va, hours_va)
        rmse_n = compute_rmse(spot_va, pred_n)
        n_new = len([f for f in avail if f not in set(current_features)])
        marker = " <-- best" if hbc_n < best_result["hbc"] else ""
        print(f"    top-{top_n:<3} pure  n={len(avail):3d}  RMSE={rmse_n:.2f}  HBC={hbc_n:.2f}  "
              f"iter={cb_n.best_iteration}  +{n_new} new{marker}")
        if hbc_n < best_result["hbc"]:
            best_result = {"hbc": hbc_n, "rmse": rmse_n, "feats": avail,
                           "n": len(avail), "top_n": top_n, "iter": cb_n.best_iteration,
                           "mode": "pure"}

    # Mode B: union top-N with current features
    print(f"\n  Mode B — union SHAP top-N with current {n_current} features:")
    for top_n in candidates:
        top_feats = [f for f, _ in ranking[:top_n]]
        union = list(dict.fromkeys(top_feats + current_features))
        avail = [f for f in union if f in df_train.columns]
        cb_n = train_tree("catboost", cb_params,
                          df_train.loc[df_train.index[valid_tr], avail], y_tr,
                          df_val.loc[df_val.index[valid_va], avail], y_va,
                          sample_weight=w_tr)
        pred_n = rm_va_or_moc_va + predict_tree(cb_n.model,
                     df_val.loc[df_val.index[valid_va], avail].values)
        _, hbc_n = compute_hbc(pred_n, spot_va, hours_va)
        rmse_n = compute_rmse(spot_va, pred_n)
        n_new = len([f for f in avail if f not in set(current_features)])
        marker = " <-- best" if hbc_n < best_result["hbc"] else ""
        print(f"    top-{top_n:<3} union={len(avail):3d}  RMSE={rmse_n:.2f}  HBC={hbc_n:.2f}  "
              f"iter={cb_n.best_iteration}  +{n_new} new{marker}")
        if hbc_n < best_result["hbc"]:
            best_result = {"hbc": hbc_n, "rmse": rmse_n, "feats": avail,
                           "n": len(avail), "top_n": top_n, "iter": cb_n.best_iteration,
                           "mode": "union"}

    delta = best_result["hbc"] - hbc_base
    new_added = [f for f in best_result["feats"] if f not in set(current_features)]
    removed = [f for f in current_features if f not in set(best_result["feats"])]
    print(f"\n  Best: {len(best_result['feats'])} feats  HBC={best_result['hbc']:.2f}  "
          f"(baseline={hbc_base:.2f}  delta={delta:+.2f})")
    print(f"  Added ({len(new_added)}): {new_added}")
    if removed:
        print(f"  Removed from current ({len(removed)}): {removed}")

    # Save
    out = {
        "features": best_result["feats"],
        "n_features": len(best_result["feats"]),
        "hbc_val": round(best_result["hbc"], 4),
        "rmse_val": round(best_result["rmse"], 4),
        "baseline_n_features": len(feat_avail),
        "baseline_hbc": round(hbc_base, 4),
        "delta_hbc": round(delta, 4),
        "source": f"SHAP re-selection on {'STL(168)' if not use_basis else 'basis'} target, "
                  f"top-{best_result.get('top_n','?')} union with previous selection",
        "new_vs_previous": new_added,
        "shap_top30": [f for f, _ in ranking[:30]],
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved to {out_path}\n")
    return best_result


# ── FR ────────────────────────────────────────────────────────────────────
fr_stat = prepare_stationary("fr_spot_la", "fr_spot", train_fe, df_train, df_val)
hours_va = df_val["hour"].values
valid_tr = fr_stat["valid_tr"]
valid_va = fr_stat["valid_va"]

with open("data/outputs/feature_selection_v5_fr.json") as f:
    feat_fr_current = json.load(f)["features"]
feat_fr_current = feat_fr_current + ["X_fr_spot_la_roll_168h_mean_x_uk_price_per_mw_7d"]

if not args.uk_only:
    run_reselection(
        market="fr",
        X_tr_pool=None, y_tr=fr_stat["y_dev_tr"][valid_tr],
        w_tr=fr_stat["weights"][valid_tr],
        X_va_pool=None, y_va=fr_stat["y_dev_va"][valid_va],
        spot_va=fr_stat["spot_va"][valid_va],
        rm_va_or_moc_va=fr_stat["rm_va"][valid_va],
        hours_va=hours_va[valid_va],
        current_features=feat_fr_current,
        cb_params=_cb_params("fr"),
        out_path="data/outputs/feature_selection_v6_fr.json",
        use_basis=False,
    )

# ── UK ────────────────────────────────────────────────────────────────────
with open("data/outputs/uk_feature_research.json") as f:
    feat_uk_current = json.load(f)["confirmed_features"]

uk_spot_tr  = df_train["uk_spot"].values
uk_spot_va  = df_val["uk_spot"].values
uk_moc_tr   = df_train["uk_merit_order_cost"].values
uk_moc_va   = df_val["uk_merit_order_cost"].values
y_basis_tr  = uk_spot_tr - uk_moc_tr
y_basis_va  = uk_spot_va - uk_moc_va
valid_tr_uk = np.isfinite(y_basis_tr)
valid_va_uk = np.isfinite(y_basis_va)

# Override valid_tr/va for UK block
_valid_tr_orig, _valid_va_orig = valid_tr, valid_va
valid_tr, valid_va = valid_tr_uk, valid_va_uk

if not args.fr_only:
    run_reselection(
        market="uk",
        X_tr_pool=None, y_tr=y_basis_tr[valid_tr_uk],
        w_tr=None,
        X_va_pool=None, y_va=y_basis_va[valid_va_uk],
        spot_va=uk_spot_va[valid_va_uk],
        rm_va_or_moc_va=uk_moc_va[valid_va_uk],
        hours_va=hours_va[valid_va_uk],
        current_features=feat_uk_current,
        cb_params=_cb_params("uk"),
        out_path="data/outputs/feature_selection_v6_uk.json",
        use_basis=True,
    )

valid_tr, valid_va = _valid_tr_orig, _valid_va_orig
