"""FR error diagnostic v2 — on best model (arcsinh + Cat32 + weights 2.0).

Identifies WHERE and WHY the RMSE is still 24.45.
"""

import sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.data_loading import load_data, merge_train
from src.feature_engineering import build_features
from catboost import CatBoostRegressor, Pool
import yaml

with open("config.yaml") as f:
    config = yaml.safe_load(f)

x_train, y_train, x_test = load_data("data/raw")
train = merge_train(x_train, y_train)
train_fe = build_features(train, config)

holdout_start = config["validation"]["holdout_start"]
mask_val = train_fe["datetime_CET"] >= holdout_start
df_train = train_fe[~mask_val].copy()
df_val = train_fe[mask_val].copy()

with open("outputs/shap_ranking_v3_clean.json") as f:
    clean_ranking = json.load(f)

# ── Train best model (config N) ─────────────────────────────────────────
CAT32_FR = [
    "fr_opportunity_cost", "fr_dynamic_marginal", "fr_import_price",
    "fr_scarcity_barrier", "fr_load_price_signal_7d",
    "fr_load_price_signal_load", "fr_hydro_opp_cost",
    "fr_basis_v2", "fr_basis_v2_lag_48h", "fr_basis_v2_roll_24h_mean",
    "fr_price_per_mw_7d",
]

base_feat = [f for f in clean_ranking["fr_spot"][:20] if f in df_train.columns]
extras = [f for f in CAT32_FR if f in df_train.columns and f not in base_feat]
features = base_feat + extras

X_tr = df_train[features]
X_va = df_val[features]
y_tr = np.arcsinh(df_train["fr_spot"])
y_va = np.arcsinh(df_val["fr_spot"])

# Sample weights
dt = pd.to_datetime(df_train["datetime_CET"])
max_dt = dt.max()
days_ago = (max_dt - dt).dt.total_seconds() / 86400
weights = np.exp(-2.0 * days_ago / 365)

CB_PARAMS = {
    "loss_function": "RMSE", "eval_metric": "RMSE",
    "iterations": 5000, "learning_rate": 0.03, "depth": 8,
    "l2_leaf_reg": 5, "subsample": 0.8, "random_seed": 42,
    "verbose": 0, "allow_writing_files": False, "use_best_model": True,
}

model = CatBoostRegressor(**CB_PARAMS)
model.fit(Pool(X_tr, y_tr, weight=weights.values), eval_set=Pool(X_va, y_va),
          early_stopping_rounds=100, verbose=0)

preds = np.sinh(model.predict(X_va))
actual = df_val["fr_spot"].values

# ── Build residuals ──────────────────────────────────────────────────────
res = df_val[["datetime_CET", "fr_spot", "fr_spot_la", "hour", "month",
              "day_of_week", "fr_load_f", "fr_wind_f", "fr_solar_f",
              "fr_nuclear_avcap_f", "fr_residual_load", "fr_thermal_need",
              "fr_scarcity_ratio", "fr_merit_order_cost",
              "fr_opportunity_cost", "fr_gas_on_margin",
              "fr_thermal_gap", "fr_renewable_pen"]].copy()
res["pred"] = preds
res["error"] = actual - preds
res["sq_error"] = res["error"] ** 2
res["abs_error"] = res["error"].abs()
res["pct_error"] = res["error"] / actual.clip(min=1) * 100

total_sq = res["sq_error"].sum()
rmse = np.sqrt(res["sq_error"].mean())
print(f"RMSE: {rmse:.3f}")
print(f"Bias: {res['error'].mean():+.2f}")
print(f"MAE: {res['abs_error'].mean():.2f}")
print(f"Samples: {len(res)}")

# ── 1. RMSE par mois ────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  1. RMSE PAR MOIS")
print("=" * 70)
ym = pd.to_datetime(res["datetime_CET"]).dt.to_period("M")
monthly = res.groupby(ym).agg(
    n=("sq_error", "count"),
    rmse=("sq_error", lambda x: np.sqrt(x.mean())),
    bias=("error", "mean"),
    mean_price=("fr_spot", "mean"),
    std_price=("fr_spot", "std"),
    sum_sq=("sq_error", "sum"),
).round(2)
monthly["pct_rmse"] = (monthly["sum_sq"] / total_sq * 100).round(1)
print(monthly[["n", "rmse", "bias", "mean_price", "std_price", "pct_rmse"]].to_string())

# ── 2. RMSE par heure ───────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  2. RMSE PAR HEURE")
print("=" * 70)
hourly = res.groupby("hour").agg(
    rmse=("sq_error", lambda x: np.sqrt(x.mean())),
    bias=("error", "mean"),
    mean_price=("fr_spot", "mean"),
    pct_sq=("sq_error", "sum"),
).round(2)
hourly["pct_rmse"] = (hourly["pct_sq"] / total_sq * 100).round(1)
print(hourly[["rmse", "bias", "mean_price", "pct_rmse"]].to_string())

# Top 5 worst hours
print("\nTop-5 pires heures:")
print(hourly.nlargest(5, "rmse")[["rmse", "bias", "mean_price"]].to_string())

# ── 3. Regime decomposition ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("  3. DECOMPOSITION PAR REGIME")
print("=" * 70)

# By price level
bins = [-500, -50, 0, 20, 40, 60, 100, 200, 5000]
labels = ["<-50", "-50..0", "0..20", "20..40", "40..60", "60..100", "100..200", ">200"]
res["price_bin"] = pd.cut(res["fr_spot"], bins=bins, labels=labels)

regime = res.groupby("price_bin", observed=True).agg(
    n=("sq_error", "count"),
    rmse=("sq_error", lambda x: np.sqrt(x.mean())),
    bias=("error", "mean"),
    mean_pred=("pred", "mean"),
    sum_sq=("sq_error", "sum"),
).round(2)
regime["pct_hours"] = (regime["n"] / len(res) * 100).round(1)
regime["pct_rmse"] = (regime["sum_sq"] / total_sq * 100).round(1)
print(regime[["n", "pct_hours", "rmse", "bias", "mean_pred", "pct_rmse"]].to_string())

# ── 4. Gas on margin vs not ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("  4. GAS ON MARGIN vs NOT")
print("=" * 70)
for flag_val, label in [(1, "Gas on margin"), (0, "Nuclear/RE on margin")]:
    sub = res[res["fr_gas_on_margin"] == flag_val]
    sub_rmse = np.sqrt(sub["sq_error"].mean())
    sub_bias = sub["error"].mean()
    sub_pct = sub["sq_error"].sum() / total_sq * 100
    print(f"  {label:25s}: {len(sub):5d} h ({len(sub)/len(res)*100:5.1f}%)  "
          f"RMSE={sub_rmse:.2f}  Bias={sub_bias:+.1f}  %RMSE={sub_pct:.1f}%")

# ── 5. Top-30 worst errors ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("  5. TOP-30 PIRES ERREURS")
print("=" * 70)
worst30 = res.nlargest(30, "sq_error")[
    ["datetime_CET", "fr_spot", "pred", "error", "hour",
     "fr_load_f", "fr_nuclear_avcap_f", "fr_wind_f",
     "fr_scarcity_ratio", "fr_gas_on_margin"]
].copy()
print(worst30.to_string(index=False))

# Concentration
for n in [10, 20, 30, 50, 100]:
    pct = res.nlargest(n, "sq_error")["sq_error"].sum() / total_sq * 100
    print(f"  Top-{n:3d} heures: {pct:.1f}% du RMSE")

# ── 6. Direction du biais ────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  6. DIRECTION DU BIAIS — sur-prediction vs sous-prediction")
print("=" * 70)
over = res[res["error"] < 0]  # actual < pred → sur-prediction
under = res[res["error"] > 0]  # actual > pred → sous-prediction
print(f"  Sur-prediction (pred > actual):  {len(over)} h ({len(over)/len(res)*100:.1f}%)")
print(f"    RMSE = {np.sqrt(over['sq_error'].mean()):.2f}, Biais moyen = {over['error'].mean():+.1f}")
print(f"    % du RMSE = {over['sq_error'].sum()/total_sq*100:.1f}%")
print(f"  Sous-prediction (pred < actual): {len(under)} h ({len(under)/len(res)*100:.1f}%)")
print(f"    RMSE = {np.sqrt(under['sq_error'].mean()):.2f}, Biais moyen = {under['error'].mean():+.1f}")
print(f"    % du RMSE = {under['sq_error'].sum()/total_sq*100:.1f}%")

# ── 7. Correlation erreur vs fondamentaux ────────────────────────────────
print("\n" + "=" * 70)
print("  7. CORRELATION |erreur| vs FONDAMENTAUX")
print("=" * 70)
fund_cols = ["fr_spot", "fr_spot_la", "fr_load_f", "fr_nuclear_avcap_f",
             "fr_wind_f", "fr_solar_f", "fr_residual_load",
             "fr_thermal_need", "fr_scarcity_ratio",
             "fr_merit_order_cost", "fr_opportunity_cost",
             "fr_thermal_gap", "fr_renewable_pen", "hour"]
corr = res[fund_cols + ["abs_error"]].corr()["abs_error"].drop("abs_error").sort_values(ascending=False)
for feat, c in corr.items():
    print(f"  {c:+.3f}  {feat}")

# ── 8. Erreur par jour de semaine ────────────────────────────────────────
print("\n" + "=" * 70)
print("  8. RMSE PAR JOUR DE SEMAINE")
print("=" * 70)
dow_names = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]
dow = res.groupby("day_of_week").agg(
    rmse=("sq_error", lambda x: np.sqrt(x.mean())),
    bias=("error", "mean"),
    mean_price=("fr_spot", "mean"),
).round(2)
dow.index = dow_names
print(dow.to_string())

# ── 9. Weekend vs weekday ───────────────────────────────────────────────
print("\n" + "=" * 70)
print("  9. WEEKEND vs WEEKDAY")
print("=" * 70)
res["is_weekend"] = res["day_of_week"].isin([5, 6])
for label, mask in [("Weekday", ~res["is_weekend"]), ("Weekend", res["is_weekend"])]:
    sub = res[mask]
    sub_rmse = np.sqrt(sub["sq_error"].mean())
    print(f"  {label:10s}: {len(sub):5d} h  RMSE={sub_rmse:.2f}  Bias={sub['error'].mean():+.1f}  "
          f"MeanPrice={sub['fr_spot'].mean():.1f}  %RMSE={sub['sq_error'].sum()/total_sq*100:.1f}%")

# ── 10. Analyse des prix NEGATIFS ────────────────────────────────────────
print("\n" + "=" * 70)
print("  10. PRIX NEGATIFS")
print("=" * 70)
neg = res[res["fr_spot"] < 0]
if len(neg) > 0:
    neg_rmse = np.sqrt(neg["sq_error"].mean())
    print(f"  {len(neg)} heures ({len(neg)/len(res)*100:.1f}%)")
    print(f"  RMSE = {neg_rmse:.2f}")
    print(f"  Bias = {neg['error'].mean():+.1f} (pred moyen = {neg['pred'].mean():.1f})")
    print(f"  % du RMSE = {neg['sq_error'].sum()/total_sq*100:.1f}%")
    print(f"  Prix moyen = {neg['fr_spot'].mean():.1f}, min = {neg['fr_spot'].min():.1f}")
    print(f"  Le modele predit-il du negatif ? {(neg['pred'] < 0).mean()*100:.0f}% des cas")
else:
    print("  Aucun prix negatif")

# ── 11. SYNTHESE ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  SYNTHESE — Sources du RMSE FR = {:.2f}".format(rmse))
print("=" * 70)

# Decompose RMSE^2 = bias^2 + variance
bias_val = res["error"].mean()
var_val = res["error"].var()
print(f"  RMSE^2 = {rmse**2:.1f}")
print(f"  = Bias^2 ({bias_val**2:.1f}) + Variance ({var_val:.1f})")
print(f"  Bias contribue {bias_val**2/rmse**2*100:.1f}% du RMSE^2")
print(f"  Variance contribue {var_val/rmse**2*100:.1f}% du RMSE^2")

# If we could perfectly correct the bias
rmse_no_bias = np.sqrt(var_val)
print(f"\n  Si on corrige parfaitement le biais: RMSE = {rmse_no_bias:.2f}")
print(f"  Si on corrige le biais PAR HEURE:")
hourly_bias = res.groupby("hour")["error"].mean()
corrected = res["error"] - res["hour"].map(hourly_bias)
rmse_hourly_corrected = np.sqrt((corrected ** 2).mean())
print(f"    RMSE = {rmse_hourly_corrected:.2f}")

print(f"\n  Si on corrige le biais PAR HEURE x MOIS:")
hm_bias = res.groupby(["hour", ym])["error"].mean()
corrected_hm = res.copy()
corrected_hm["ym"] = ym.values
corrected_hm["hm_key"] = list(zip(corrected_hm["hour"], corrected_hm["ym"]))
for key, bias in hm_bias.items():
    mask = corrected_hm["hm_key"] == key
    corrected_hm.loc[mask, "error_corrected"] = corrected_hm.loc[mask, "error"] - bias
rmse_hm = np.sqrt((corrected_hm["error_corrected"] ** 2).mean())
print(f"    RMSE = {rmse_hm:.2f}")
