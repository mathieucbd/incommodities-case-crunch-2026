"""Quick test: Elastic Net with ALL features vs selected features.

Hypothesis: L1 regularization does its own feature selection,
so giving it all 400+ features could be better than our 28/150 curated list.
Also tests finer HP grid for alpha/l1_ratio.
"""

import sys, yaml, warnings, time, json
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, ".")
from src.data_loading import load_data, merge_train
from src.feature_engineering import build_features

warnings.filterwarnings("ignore")

with open("config.yaml") as f:
    config = yaml.safe_load(f)

print("=" * 90)
print("  ELASTIC NET — FEATURE COUNT & HP SWEEP")
print("=" * 90)

t0 = time.time()
x_train, y_train, x_test = load_data("data/raw")
train = merge_train(x_train, y_train)
df = build_features(train, config)

holdout_start = config["validation"]["holdout_start"]
mask_val = (df["datetime_CET"] >= holdout_start).values

if "fr_spot_la_roll_168h_mean" in df.columns and "uk_price_per_mw_7d" in df.columns:
    df["X_fr_spot_la_roll_168h_mean_x_uk_price_per_mw_7d"] = (
        df["fr_spot_la_roll_168h_mean"] * df["uk_price_per_mw_7d"]
    )

df_tr = df[~mask_val].copy()
df_va = df[mask_val].copy()

# ── Feature lists ────────────────────────────────────────────────────
with open("outputs/feature_selection_v5_fr.json") as f:
    fs_v5 = json.load(f)
FR_SELECTED = fs_v5["features"] + ["X_fr_spot_la_roll_168h_mean_x_uk_price_per_mw_7d"]
FR_SELECTED = [f for f in FR_SELECTED if f in df_tr.columns]

with open("outputs/uk_feature_research.json") as f:
    uk_research = json.load(f)
UK_SELECTED = [f for f in uk_research["confirmed_features"] if f in df_tr.columns]

# All numeric features (exclude targets, dates, IDs)
EXCLUDE = {"fr_spot", "uk_spot", "datetime_CET", "datetime_UTC", "date", "id"}
ALL_FEATURES = [c for c in df_tr.columns
                if c not in EXCLUDE
                and df_tr[c].dtype in ["float64", "float32", "int64", "int32"]
                and df_tr[c].notna().sum() > len(df_tr) * 0.5]  # >50% non-null

# Deduplicate by correlation >0.99
print(f"  All numeric features before dedup: {len(ALL_FEATURES)}")
corr_matrix = df_tr[ALL_FEATURES].corr().abs()
to_drop = set()
for i in range(len(ALL_FEATURES)):
    if ALL_FEATURES[i] in to_drop:
        continue
    for j in range(i + 1, len(ALL_FEATURES)):
        if ALL_FEATURES[j] in to_drop:
            continue
        if corr_matrix.iloc[i, j] > 0.99:
            to_drop.add(ALL_FEATURES[j])
ALL_DEDUP = [f for f in ALL_FEATURES if f not in to_drop]
print(f"  After 0.99 correlation dedup: {len(ALL_DEDUP)}")

# ── Targets ──────────────────────────────────────────────────────────
fr_spot_la_full = df["fr_spot_la"].values
ema_fr = pd.Series(fr_spot_la_full).ewm(span=240).mean().values
fr_anchor_va = ema_fr[mask_val]
fr_spot_va = df_va["fr_spot"].values
fr_y_tr = df_tr["fr_spot"].values - ema_fr[~mask_val]
fr_y_va = fr_spot_va - fr_anchor_va

uk_moc_va = df_va["uk_merit_order_cost"].values
uk_spot_va = df_va["uk_spot"].values
uk_y_tr = df_tr["uk_spot"].values - df_tr["uk_merit_order_cost"].values
uk_y_va = uk_spot_va - uk_moc_va

hours_va = df_va["hour"].values

fr_valid_tr = np.isfinite(fr_y_tr) & np.isfinite(ema_fr[~mask_val])
fr_valid_va = np.isfinite(fr_y_va) & np.isfinite(fr_anchor_va)
uk_valid_tr = np.isfinite(uk_y_tr)
uk_valid_va = np.isfinite(uk_y_va)


def compute_rmse(actual, preds):
    return np.sqrt(np.mean((actual - preds) ** 2))


def apply_hbc(preds, actual, hours):
    errors = actual - preds
    hbc = {h: float(errors[hours == h].mean()) for h in range(24) if (hours == h).sum() > 0}
    corrected = preds + np.array([hbc.get(h, 0) for h in hours])
    return np.sqrt(np.mean((actual - corrected) ** 2))


# ── HP grid ──────────────────────────────────────────────────────────
ALPHAS = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
L1_RATIOS = [0.5, 0.7, 0.8, 0.9, 0.95, 1.0]


def test_en(market, features, y_tr, y_va, v_tr, v_va, anchor_va, spot_va, feat_label):
    """Test EN with given features across HP grid."""
    feat = [f for f in features if f in df_tr.columns]
    hrs = hours_va[v_va]
    actual = spot_va[v_va]

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(np.nan_to_num(df_tr.loc[df_tr.index[v_tr], feat].values, 0))
    X_va = scaler.transform(np.nan_to_num(df_va.loc[df_va.index[v_va], feat].values, 0))

    best = {"rmse_hbc": 999, "name": "", "n_nonzero": 0}
    results = []

    for alpha in ALPHAS:
        for l1 in L1_RATIOS:
            en = ElasticNet(alpha=alpha, l1_ratio=l1, max_iter=10000)
            en.fit(X_tr, y_tr[v_tr])
            preds = anchor_va[v_va] + en.predict(X_va)
            rmse = compute_rmse(actual, preds)
            rmse_hbc = apply_hbc(preds, actual, hrs)
            n_nz = np.sum(en.coef_ != 0)
            results.append({
                "alpha": alpha, "l1": l1, "rmse": rmse, "rmse_hbc": rmse_hbc,
                "n_nonzero": int(n_nz), "n_features": len(feat),
            })
            if rmse_hbc < best["rmse_hbc"]:
                best = {"rmse_hbc": rmse_hbc, "rmse": rmse,
                        "alpha": alpha, "l1": l1,
                        "n_nonzero": int(n_nz), "name": feat_label}

    # Also test Ridge
    for alpha in [1.0, 10.0, 100.0]:
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_tr, y_tr[v_tr])
        preds = anchor_va[v_va] + ridge.predict(X_va)
        rmse = compute_rmse(actual, preds)
        rmse_hbc = apply_hbc(preds, actual, hrs)
        results.append({
            "alpha": alpha, "l1": "ridge", "rmse": rmse, "rmse_hbc": rmse_hbc,
            "n_nonzero": len(feat), "n_features": len(feat),
        })
        if rmse_hbc < best["rmse_hbc"]:
            best = {"rmse_hbc": rmse_hbc, "rmse": rmse,
                    "alpha": alpha, "l1": "ridge",
                    "n_nonzero": len(feat), "name": feat_label}

    return best, results


# ══════════════════════════════════════════════════════════════════════
for market, features_sets, y_tr, y_va, v_tr, v_va, anchor_va, spot_va in [
    ("FR", [("selected_28", FR_SELECTED), ("all_dedup", ALL_DEDUP)],
     fr_y_tr, fr_y_va, fr_valid_tr, fr_valid_va, fr_anchor_va, fr_spot_va),
    ("UK", [("selected_150", UK_SELECTED), ("all_dedup", ALL_DEDUP)],
     uk_y_tr, uk_y_va, uk_valid_tr, uk_valid_va, uk_moc_va, uk_spot_va),
]:
    print(f"\n{'='*90}")
    print(f"  {market} — FEATURE & HP SWEEP")
    print(f"{'='*90}")

    all_bests = []
    for feat_label, feat_list in features_sets:
        best, results = test_en(market, feat_list, y_tr, y_va, v_tr, v_va,
                                anchor_va, spot_va, feat_label)
        all_bests.append(best)

        # Show top 5
        top5 = sorted(results, key=lambda x: x["rmse_hbc"])[:5]
        print(f"\n  {feat_label} ({len([f for f in feat_list if f in df_tr.columns])} features):")
        print(f"    Best: alpha={best['alpha']}, l1={best['l1']}, "
              f"RMSE={best['rmse']:.2f}, +HBC={best['rmse_hbc']:.2f}, "
              f"n_nonzero={best['n_nonzero']}")
        for r in top5:
            print(f"      a={r['alpha']:5.1f} l1={str(r['l1']):5s}  "
                  f"RMSE={r['rmse']:.2f}  +HBC={r['rmse_hbc']:.2f}  "
                  f"nz={r['n_nonzero']}/{r['n_features']}")

    # Compare
    print(f"\n  {market} comparison:")
    for b in all_bests:
        print(f"    {b['name']:15s}: RMSE={b['rmse']:.2f}  +HBC={b['rmse_hbc']:.2f}  "
              f"(alpha={b['alpha']}, l1={b['l1']}, nz={b['n_nonzero']})")

print(f"\n  Total time: {time.time() - t0:.0f}s")
