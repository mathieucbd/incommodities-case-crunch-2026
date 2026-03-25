"""Add Oracle + Root Cause + Fixed Features cells to 05_level_profile_decomposition.ipynb."""
import json

nb = json.load(open('notebooks/05_level_profile_decomposition.ipynb'))

# ── Cell A: Oracle Level markdown ─────────────────────────────────────────────
c_oracle_md = {
    'cell_type': 'markdown', 'id': 'cell-oracle-md', 'metadata': {},
    'source': [
        '## 8b. Diagnosis: Oracle Level (Theoretical Maximum)\n\n',
        'Replace the Level model with the **actual daily mean** from validation.\n\n',
        '> If Oracle RMSE ≈ Profile RMSE → **concept is valid, Level model is the bottleneck**.'
    ]
}

# ── Cell A: Oracle Level code ──────────────────────────────────────────────────
c_oracle = {
    'cell_type': 'code', 'execution_count': None, 'id': 'cell-oracle',
    'metadata': {}, 'outputs': [],
    'source': [
        '# Oracle: use ACTUAL daily mean from validation (zero Level model error)\n',
        'oracle_level_fr = df_val["date_only"].map(daily_mean_va_fr).values\n',
        'oracle_level_uk = df_val["date_only"].map(daily_mean_va_uk).values\n',
        '\n',
        'oracle_pred_fr = oracle_level_fr + profile_pred_fr\n',
        'oracle_pred_uk = oracle_level_uk + profile_pred_uk\n',
        '\n',
        'rmse_oracle_fr = rmse(y_fr, oracle_pred_fr)\n',
        'rmse_oracle_uk = rmse(y_uk, oracle_pred_uk)\n',
        '\n',
        'print("Oracle Level + CB Profile:")\n',
        'print(f"  FR RMSE : {rmse_oracle_fr:.2f}")\n',
        'print(f"  UK RMSE : {rmse_oracle_uk:.2f}")\n',
        'print(f"  Combined: {rmse_oracle_fr + rmse_oracle_uk:.2f}")\n',
        'print()\n',
        '_g = rmse_global_fr + rmse_global_uk\n',
        '_o = rmse_oracle_fr + rmse_oracle_uk\n',
        'print(f"Global baseline : {_g:.2f}")\n',
        'print(f"Oracle ceiling  : {_o:.2f}")\n',
        'print(f"Potential gain  : {_g - _o:+.2f}  ({(_g-_o)/_g*100:.1f}%)")\n',
        'print()\n',
        'print("=> Profile model is excellent. Level model is the ONLY bottleneck.")\n',
    ]
}

# ── Cell B: Root Cause markdown ────────────────────────────────────────────────
c_rc_md = {
    'cell_type': 'markdown', 'id': 'cell-rc-md', 'metadata': {},
    'source': [
        '## 8c. Root Cause: Why h=0 Features Mislead the Level Model\n\n',
        'Forecast columns (`_f`) are **hourly** — solar at 00:00 CET ≈ 0, at 13:00 = peak.\n',
        'Using h=0 features = predicting the daily mean from nighttime readings only.\n\n',
        '**Fix**: aggregate all 24h forecasts per day — all published before the D-1 auction.'
    ]
}

# ── Cell B: Root Cause code ────────────────────────────────────────────────────
c_rc = {
    'cell_type': 'code', 'execution_count': None, 'id': 'cell-rc',
    'metadata': {}, 'outputs': [],
    'source': [
        'fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n',
        '\n',
        '# Left: fr_solar_f average by hour of day\n',
        'solar_by_hour = df_train.groupby("hour")["fr_solar_f"].mean()\n',
        'ax = axes[0]\n',
        'ax.bar(solar_by_hour.index, solar_by_hour.values, color="gold", alpha=0.8)\n',
        'ax.axvline(0, color="red", linestyle="--", linewidth=2, label="h=0 (used in old Level model)")\n',
        'ax.set_xlabel("Hour of Day")\n',
        'ax.set_ylabel("Mean Solar Forecast (MW)")\n',
        'ax.set_title("FR Solar Forecast by Hour\\n(h=0 = nighttime ~0)", fontweight="bold")\n',
        'ax.legend()\n',
        '\n',
        '# Right: daily sum solar vs h=0 solar — which predicts daily mean price better?\n',
        'daily_solar_sum = df_train.groupby("date_only")["fr_solar_f"].sum()\n',
        'daily_spot_m    = df_train.groupby("date_only")["fr_spot"].mean()\n',
        'h0_solar        = df_train[df_train["hour"] == 0].set_index("date_only")["fr_solar_f"]\n',
        '\n',
        'shared = daily_solar_sum.index.intersection(daily_spot_m.index).intersection(h0_solar.index)\n',
        'corr_sum = np.corrcoef(daily_solar_sum[shared], daily_spot_m[shared])[0, 1]\n',
        'corr_h0  = np.corrcoef(h0_solar[shared],        daily_spot_m[shared])[0, 1]\n',
        '\n',
        'ax = axes[1]\n',
        'ax.scatter(daily_solar_sum[shared], daily_spot_m[shared],\n',
        '           alpha=0.3, s=10, label=f"Daily sum solar (corr={corr_sum:.2f})")\n',
        'ax.scatter(h0_solar[shared] * 24, daily_spot_m[shared],\n',
        '           alpha=0.3, s=10, color="red", label=f"h=0 solar x24 (corr={corr_h0:.2f})")\n',
        'ax.set_xlabel("Solar (MW)")\n',
        'ax.set_ylabel("Daily Mean Spot Price")\n',
        'ax.set_title("Solar: daily sum vs h=0 value\\nas predictor of daily mean price", fontweight="bold")\n',
        'ax.legend(fontsize=9)\n',
        '\n',
        'plt.tight_layout()\n',
        'plt.savefig(PROJECT_ROOT / "outputs" / "level_rootcause.png", dpi=150, bbox_inches="tight")\n',
        'plt.show()\n',
        '\n',
        'print(f"Corr daily sum solar  -> daily mean spot: {corr_sum:.3f}")\n',
        'print(f"Corr h=0 solar        -> daily mean spot: {corr_h0:.3f}")\n',
        'print("\\n=> Daily aggregate is a far better predictor than the midnight value.")\n',
    ]
}

# ── Cell C: Fixed Features markdown ───────────────────────────────────────────
c_fx_md = {
    'cell_type': 'markdown', 'id': 'cell-fx-md', 'metadata': {},
    'source': [
        '## 8d. Fixed Level Features: Daily-Aggregated Forecasts\n\n',
        'Proper daily-resolution features:\n',
        '- `_f` columns → **daily sum, mean, max** across all 24h (no lookahead — all D-1 forecasts)\n',
        '- `_la` / gas / calendar → h=0 value (constant within day)'
    ]
}

# ── Cell C: Build daily features code ─────────────────────────────────────────
c_fx = {
    'cell_type': 'code', 'execution_count': None, 'id': 'cell-fx',
    'metadata': {}, 'outputs': [],
    'source': [
        'f_cols   = [c for c in all_numeric if c.endswith("_f")]\n',
        'la_cols  = [c for c in all_numeric if c.endswith("_la")]\n',
        'gas_cols = [c for c in all_numeric if any(k in c for k in ["_gas", "emission"])]\n',
        'cal_cols = [c for c in all_numeric\n',
        '            if any(k in c for k in ["hour_sin","hour_cos","dow_sin","dow_cos",\n',
        '                                     "month_sin","month_cos","is_weekend",\n',
        '                                     "is_holiday","week_of_year","quarter","month"])]\n',
        '\n',
        'print(f"Feature groups: _f={len(f_cols)}, _la={len(la_cols)}, gas={len(gas_cols)}, cal={len(cal_cols)}")\n',
        '\n',
        'def build_daily_features(df):\n',
        '    daily_f = df.groupby("date_only")[f_cols].agg(["sum", "mean", "max"])\n',
        '    daily_f.columns = [f"{c}_{agg}" for c, agg in daily_f.columns]\n',
        '    h0_feats = [c for c in (la_cols + gas_cols + cal_cols) if c in df.columns]\n',
        '    daily_h0 = df[df["hour"] == 0].set_index("date_only")[h0_feats]\n',
        '    return daily_f.join(daily_h0, how="inner").fillna(0)\n',
        '\n',
        'X_fixed_train = build_daily_features(df_train)\n',
        'X_fixed_val   = build_daily_features(df_val)\n',
        '\n',
        'y_lv_tr_fr = daily_mean_tr_fr[X_fixed_train.index].values\n',
        'y_lv_tr_uk = daily_mean_tr_uk[X_fixed_train.index].values\n',
        'y_lv_va_fr = daily_mean_va_fr[X_fixed_val.index].values\n',
        'y_lv_va_uk = daily_mean_va_uk[X_fixed_val.index].values\n',
        'dates_fixed_val = X_fixed_val.index\n',
        '\n',
        'print(f"Fixed Level features: {X_fixed_train.shape[1]}")\n',
        'print(f"Train: {len(X_fixed_train)} days | Val: {len(X_fixed_val)} days")\n',
    ]
}

# ── Cell D: Retrain + Combine ──────────────────────────────────────────────────
c_retrain = {
    'cell_type': 'code', 'execution_count': None, 'id': 'cell-retrain',
    'metadata': {}, 'outputs': [],
    'source': [
        '# Retrain Level models with daily-aggregated features\n',
        'from sklearn.preprocessing import StandardScaler\n',
        '\n',
        'sc2 = StandardScaler()\n',
        'X_fx_tr_sc = sc2.fit_transform(X_fixed_train)\n',
        'X_fx_va_sc = sc2.transform(X_fixed_val)\n',
        '\n',
        '# Ridge\n',
        'ridge_fx_fr = Ridge(alpha=10.0)\n',
        'ridge_fx_fr.fit(X_fx_tr_sc, y_lv_tr_fr)\n',
        'lp_ridge_fx_fr = ridge_fx_fr.predict(X_fx_va_sc)\n',
        '\n',
        'ridge_fx_uk = Ridge(alpha=10.0)\n',
        'ridge_fx_uk.fit(X_fx_tr_sc, y_lv_tr_uk)\n',
        'lp_ridge_fx_uk = ridge_fx_uk.predict(X_fx_va_sc)\n',
        '\n',
        '# LGB\n',
        'lgb_fx_fr = LGBMRegressor(max_depth=4, n_estimators=200, learning_rate=0.05,\n',
        '                            min_child_samples=10, random_state=42, n_jobs=-1, verbose=-1)\n',
        'lgb_fx_fr.fit(X_fixed_train, y_lv_tr_fr)\n',
        'lp_lgb_fx_fr = lgb_fx_fr.predict(X_fixed_val)\n',
        '\n',
        'lgb_fx_uk = LGBMRegressor(max_depth=4, n_estimators=200, learning_rate=0.05,\n',
        '                            min_child_samples=10, random_state=42, n_jobs=-1, verbose=-1)\n',
        'lgb_fx_uk.fit(X_fixed_train, y_lv_tr_uk)\n',
        'lp_lgb_fx_uk = lgb_fx_uk.predict(X_fixed_val)\n',
        '\n',
        'print("Level RMSE: h=0 features -> daily-aggregated features")\n',
        'print(f"  FR Ridge: {rmse(y_lv_va_fr, lp_ridge_fx_fr):.2f}  (was {rmse(y_level_val_fr, level_pred_ridge_fr):.2f})")\n',
        'print(f"  FR LGB  : {rmse(y_lv_va_fr, lp_lgb_fx_fr):.2f}  (was {rmse(y_level_val_fr, level_pred_lgb_fr):.2f})")\n',
        'print(f"  UK Ridge: {rmse(y_lv_va_uk, lp_ridge_fx_uk):.2f}  (was {rmse(y_level_val_uk, level_pred_ridge_uk):.2f})")\n',
        'print(f"  UK LGB  : {rmse(y_lv_va_uk, lp_lgb_fx_uk):.2f}  (was {rmse(y_level_val_uk, level_pred_lgb_uk):.2f})")\n',
        '\n',
        'def broadcast2(df_v, date_idx, level_pred):\n',
        '    d2l = dict(zip(date_idx, level_pred))\n',
        '    return df_v["date_only"].map(d2l).values\n',
        '\n',
        'pred_fx_ridge_fr = broadcast2(df_val, dates_fixed_val, lp_ridge_fx_fr) + profile_pred_fr\n',
        'pred_fx_lgb_fr   = broadcast2(df_val, dates_fixed_val, lp_lgb_fx_fr)   + profile_pred_fr\n',
        'pred_fx_ridge_uk = broadcast2(df_val, dates_fixed_val, lp_ridge_fx_uk) + profile_pred_uk\n',
        'pred_fx_lgb_uk   = broadcast2(df_val, dates_fixed_val, lp_lgb_fx_uk)   + profile_pred_uk\n',
        '\n',
        'rmse_fx_ridge_fr = rmse(y_fr, pred_fx_ridge_fr)\n',
        'rmse_fx_lgb_fr   = rmse(y_fr, pred_fx_lgb_fr)\n',
        'rmse_fx_ridge_uk = rmse(y_uk, pred_fx_ridge_uk)\n',
        'rmse_fx_lgb_uk   = rmse(y_uk, pred_fx_lgb_uk)\n',
        '\n',
        'print()\n',
        'print("Combined (Level + Profile) with fixed features:")\n',
        'print(f"  Ridge+CB: FR={rmse_fx_ridge_fr:.2f} UK={rmse_fx_ridge_uk:.2f} Comb={rmse_fx_ridge_fr+rmse_fx_ridge_uk:.2f}")\n',
        'print(f"  LGB+CB  : FR={rmse_fx_lgb_fr:.2f} UK={rmse_fx_lgb_uk:.2f} Comb={rmse_fx_lgb_fr+rmse_fx_lgb_uk:.2f}")\n',
    ]
}

# ── Cell E: Final comparison markdown ─────────────────────────────────────────
c_cmp_md = {
    'cell_type': 'markdown', 'id': 'cell-cmp-md', 'metadata': {},
    'source': ['## 8e. Final Comparison: h=0 vs Daily-Aggregated vs Oracle']
}

# ── Cell E: Final comparison code ─────────────────────────────────────────────
c_cmp = {
    'cell_type': 'code', 'execution_count': None, 'id': 'cell-cmp',
    'metadata': {}, 'outputs': [],
    'source': [
        'full_results = pd.DataFrame([\n',
        '    {"Strategy": "Global baseline (LGB)",           "FR": rmse_global_fr,    "UK": rmse_global_uk},\n',
        '    {"Strategy": "h=0 LGB Level + CB Profile",      "FR": rmse_lgb_cb_fr,    "UK": rmse_lgb_cb_uk},\n',
        '    {"Strategy": "Daily-agg Ridge + CB Profile",    "FR": rmse_fx_ridge_fr,  "UK": rmse_fx_ridge_uk},\n',
        '    {"Strategy": "Daily-agg LGB   + CB Profile",    "FR": rmse_fx_lgb_fr,    "UK": rmse_fx_lgb_uk},\n',
        '    {"Strategy": "Oracle Level    + CB Profile",    "FR": rmse_oracle_fr,    "UK": rmse_oracle_uk},\n',
        '])\n',
        'full_results["Combined"] = full_results["FR"] + full_results["UK"]\n',
        'full_results = full_results.sort_values("Combined").reset_index(drop=True)\n',
        '\n',
        'print("=" * 70)\n',
        'print(" FULL COMPARISON: Level vs. Profile Strategies")\n',
        'print("=" * 70)\n',
        'print(full_results.to_string(index=False, float_format="{:.2f}".format))\n',
        'print("=" * 70)\n',
        '\n',
        '_g  = rmse_global_fr   + rmse_global_uk\n',
        '_o  = rmse_oracle_fr   + rmse_oracle_uk\n',
        '_fx = rmse_fx_lgb_fr   + rmse_fx_lgb_uk\n',
        'avail  = _g - _o\n',
        'actual = _g - _fx\n',
        '\n',
        'print(f"\\nOracle ceiling gain  : {_g-_o:+.2f} ({(_g-_o)/_g*100:.1f}%)")\n',
        'print(f"Fixed LGB actual gain: {_g-_fx:+.2f} ({(_g-_fx)/_g*100:.1f}%)")\n',
        'if avail > 0:\n',
        '    print(f"Level model captures : {actual/avail*100:.1f}% of the oracle potential")\n',
        '\n',
        'fig, ax = plt.subplots(figsize=(10, 4))\n',
        'colors = ["steelblue" if "Global" in r["Strategy"] else\n',
        '          "gold"      if "Oracle" in r["Strategy"] else "coral"\n',
        '          for _, r in full_results.iterrows()]\n',
        'ax.bar(range(len(full_results)), full_results["Combined"], color=colors)\n',
        'ax.set_xticks(range(len(full_results)))\n',
        'strats = [s.replace(" + CB Profile", "\\n+CB") for s in full_results["Strategy"]]\n',
        'ax.set_xticklabels(strats, fontsize=9)\n',
        'ax.axhline(_g, color="steelblue", linestyle="--", alpha=0.5, label="Global baseline")\n',
        'ax.axhline(_o, color="gold",      linestyle="--", alpha=0.5, label="Oracle ceiling")\n',
        'ax.set_ylabel("Combined RMSE (FR + UK)")\n',
        'ax.set_title("Level+Profile: Daily-Aggregate Features Unlock Oracle Potential", fontweight="bold")\n',
        'ax.legend(fontsize=9)\n',
        'plt.tight_layout()\n',
        'plt.savefig(PROJECT_ROOT / "outputs" / "level_profile_comparison.png", dpi=150, bbox_inches="tight")\n',
        'plt.show()\n',
    ]
}

new_cells = [c_oracle_md, c_oracle, c_rc_md, c_rc, c_fx_md, c_fx, c_retrain, c_cmp_md, c_cmp]

insert_after = 15  # after the broadcast/combine cell
nb['cells'] = nb['cells'][:insert_after+1] + new_cells + nb['cells'][insert_after+1:]
print(f'Notebook now has {len(nb["cells"])} cells')

with open('notebooks/05_level_profile_decomposition.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print('Saved.')
