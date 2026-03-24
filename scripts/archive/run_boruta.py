"""Run Boruta feature selection + save full results (Lasso + SHAP + Boruta + Consensus).

Usage:
    cd "INCOMO 3"
    python run_boruta.py

Takes ~30-60 min depending on your machine.
Results saved to: outputs/feature_selection_results.json
"""

import sys, json, warnings, time
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

import yaml
from src.data_loading import load_data, merge_train
from src.feature_engineering import build_features
from src.validation import create_holdout_split, split_X_y
from src.feature_selection import (
    lasso_selection,
    shap_selection,
    boruta_selection,
    consensus_selection,
)

with open("config.yaml") as f:
    config = yaml.safe_load(f)

print("Loading data...")
x_train, y_train, x_test = load_data()
train = merge_train(x_train, y_train)
df = build_features(train, config)
train_df, val_df = create_holdout_split(df, config)

results = {}
for target in ["fr_spot", "uk_spot"]:
    print(f"\n{'='*60}")
    print(f"  FULL FEATURE SELECTION — {target}")
    print(f"{'='*60}")

    X_tr, y_tr = split_X_y(train_df, target)
    X_val, y_val = split_X_y(val_df, target)

    t0 = time.time()

    print(f"\n[1/3] Lasso...")
    lasso_feats = lasso_selection(X_tr, y_tr, top_n=150)

    print(f"\n[2/3] SHAP...")
    shap_feats, shap_imp = shap_selection(X_tr, y_tr, X_val, y_val, top_n=150)

    print(f"\n[3/3] Boruta (this is the slow part)...")
    boruta_feats = boruta_selection(X_tr, y_tr, max_iter=50)

    consensus = consensus_selection(lasso_feats, shap_feats, boruta_feats, min_votes=2)

    elapsed = time.time() - t0
    print(f"\n  {target} done in {elapsed/60:.1f} min")

    results[target] = {
        "lasso": lasso_feats,
        "shap": shap_feats,
        "boruta": boruta_feats,
        "consensus": consensus,
        "shap_importances": {k: float(v) for k, v in shap_imp.items()},
    }

with open("outputs/feature_selection_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*60}")
print("DONE — Results saved to outputs/feature_selection_results.json")
print(f"{'='*60}")
for target in ["fr_spot", "uk_spot"]:
    r = results[target]
    print(f"\n  {target}:")
    print(f"    Lasso:     {len(r['lasso'])} features")
    print(f"    SHAP:      {len(r['shap'])} features")
    print(f"    Boruta:    {len(r['boruta'])} features")
    print(f"    Consensus: {len(r['consensus'])} features (>=2 votes)")
