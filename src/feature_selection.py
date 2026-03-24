"""Feature selection module for INCOMO 3.

Three complementary methods to identify the most relevant features:
1. Lasso (L1 regularization) — fast linear filter, kills noise
2. SHAP (TreeExplainer) — non-linear importance from a quick LightGBM
3. Boruta (shadow features) — statistical test for feature relevance

Usage:
    from src.feature_selection import run_feature_selection

    selected = run_feature_selection(X_train, y_train, X_val, y_val, target="fr_spot", config=config)
    # selected = {"lasso": [...], "shap": [...], "boruta": [...], "consensus": [...]}
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Lasso — L1 linear filter
# ---------------------------------------------------------------------------

def lasso_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    top_n: int = 150,
    alpha: float | None = None,
) -> list[str]:
    """Select features via Lasso (L1 regularized linear regression).

    If alpha is None, uses LassoCV with 5-fold time-series split.
    Returns feature names with non-zero coefficients, sorted by |coeff|.
    """
    from sklearn.linear_model import LassoCV, Lasso
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import TimeSeriesSplit

    # Keep only numeric columns
    X_train = X_train.select_dtypes(include="number")

    # Standardize (Lasso is scale-sensitive)
    scaler = StandardScaler()
    # Handle NaN: fill with median for Lasso (it can't handle NaN)
    X_filled = X_train.fillna(X_train.median())
    X_scaled = scaler.fit_transform(X_filled)

    if alpha is None:
        tscv = TimeSeriesSplit(n_splits=3)
        model = LassoCV(
            alphas=np.logspace(-4, 1, 50),
            cv=tscv,
            max_iter=5000,
            n_jobs=-1,
        )
        model.fit(X_scaled, y_train)
        print(f"  LassoCV alpha: {model.alpha_:.4f}")
    else:
        model = Lasso(alpha=alpha, max_iter=5000)
        model.fit(X_scaled, y_train)

    coefs = pd.Series(np.abs(model.coef_), index=X_train.columns)
    nonzero = coefs[coefs > 0].sort_values(ascending=False)

    selected = nonzero.head(top_n).index.tolist()
    print(f"  Lasso: {len(nonzero)} non-zero / {len(X_train.columns)} total → keeping top {len(selected)}")

    return selected


# ---------------------------------------------------------------------------
# 2. SHAP — TreeExplainer on quick LightGBM
# ---------------------------------------------------------------------------

def shap_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    top_n: int = 150,
) -> tuple[list[str], pd.Series]:
    """Select features via SHAP importance from a quick LightGBM model.

    Returns:
        (selected_features, shap_importances) — sorted by mean |SHAP|.
    """
    import lightgbm as lgb
    import shap

    # Quick LightGBM (not optimized, just for SHAP)
    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        feature_fraction=0.7,
        bagging_fraction=0.8,
        bagging_freq=5,
        min_child_samples=20,
        verbose=-1,
        n_jobs=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    # SHAP values on validation set (faster than train)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_val)

    # Mean absolute SHAP per feature
    mean_abs_shap = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=X_train.columns,
    ).sort_values(ascending=False)

    selected = mean_abs_shap.head(top_n).index.tolist()
    print(f"  SHAP: top-{len(selected)} features selected (max |SHAP| = {mean_abs_shap.iloc[0]:.2f})")

    return selected, mean_abs_shap


# ---------------------------------------------------------------------------
# 3. Boruta — Shadow feature statistical test
# ---------------------------------------------------------------------------

def boruta_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    max_iter: int = 100,
) -> list[str]:
    """Select features via Boruta (shadow feature importance test).

    Uses RandomForest as the base estimator. Returns features confirmed
    as important by the Boruta statistical test.
    """
    from boruta import BorutaPy
    from sklearn.ensemble import RandomForestRegressor

    # Keep only numeric columns + handle NaN
    X_train = X_train.select_dtypes(include="number")
    X_filled = X_train.fillna(X_train.median())

    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=7,
        n_jobs=-1,
        random_state=42,
    )

    boruta = BorutaPy(
        estimator=rf,
        n_estimators="auto",
        max_iter=max_iter,
        random_state=42,
        verbose=0,
    )
    boruta.fit(X_filled.values, y_train.values)

    confirmed = X_train.columns[boruta.support_].tolist()
    tentative = X_train.columns[boruta.support_weak_].tolist()

    # Include both confirmed and tentative
    selected = confirmed + tentative
    print(f"  Boruta: {len(confirmed)} confirmed + {len(tentative)} tentative = {len(selected)} selected")

    return selected


# ---------------------------------------------------------------------------
# 4. Consensus — Intersection / Union with voting
# ---------------------------------------------------------------------------

def consensus_selection(
    lasso_features: list[str],
    shap_features: list[str],
    boruta_features: list[str],
    min_votes: int = 2,
) -> list[str]:
    """Select features that appear in at least ``min_votes`` methods.

    Args:
        min_votes: Minimum number of methods agreeing (1=union, 2=majority, 3=intersection).

    Returns:
        Sorted list of consensus features.
    """
    from collections import Counter

    all_features = lasso_features + shap_features + boruta_features
    vote_counts = Counter(all_features)

    consensus = sorted([f for f, count in vote_counts.items() if count >= min_votes])

    print(f"\n  Consensus (>={min_votes} votes): {len(consensus)} features")
    print(f"    All 3 methods: {sum(1 for c in vote_counts.values() if c == 3)}")
    print(f"    2 methods:     {sum(1 for c in vote_counts.values() if c == 2)}")
    print(f"    1 method only: {sum(1 for c in vote_counts.values() if c == 1)}")

    return consensus


# ---------------------------------------------------------------------------
# 5. Main entry point
# ---------------------------------------------------------------------------

def run_feature_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    target: str = "fr_spot",
    config: dict | None = None,
    top_n: int = 150,
    boruta_max_iter: int = 100,
    min_votes: int = 2,
) -> dict:
    """Run all three feature selection methods and return consensus.

    Args:
        X_train, y_train: Training data.
        X_val, y_val: Validation data (for SHAP).
        target: Target name for logging.
        config: Optional config dict (unused for now, future extensibility).
        top_n: Max features to keep from Lasso and SHAP.
        boruta_max_iter: Max Boruta iterations.
        min_votes: Min votes for consensus (2 = majority).

    Returns:
        Dictionary with keys: lasso, shap, boruta, consensus, shap_importances.
    """
    print(f"\n{'='*60}")
    print(f"  Feature Selection — {target}")
    print(f"  {X_train.shape[1]} features, {len(X_train):,} train rows")
    print(f"{'='*60}")

    print(f"\n[1/3] Lasso (L1)...")
    lasso_feats = lasso_selection(X_train, y_train, top_n=top_n)

    print(f"\n[2/3] SHAP (LightGBM TreeExplainer)...")
    shap_feats, shap_importances = shap_selection(X_train, y_train, X_val, y_val, top_n=top_n)

    print(f"\n[3/3] Boruta (shadow features)...")
    boruta_feats = boruta_selection(X_train, y_train, max_iter=boruta_max_iter)

    consensus = consensus_selection(lasso_feats, shap_feats, boruta_feats, min_votes=min_votes)

    results = {
        "lasso": lasso_feats,
        "shap": shap_feats,
        "boruta": boruta_feats,
        "consensus": consensus,
        "shap_importances": shap_importances,
    }

    print(f"\n  Summary for {target}:")
    print(f"    Lasso:     {len(lasso_feats)} features")
    print(f"    SHAP:      {len(shap_feats)} features")
    print(f"    Boruta:    {len(boruta_feats)} features")
    print(f"    Consensus: {len(consensus)} features (>={min_votes} votes)")
    print(f"{'='*60}\n")

    return results
