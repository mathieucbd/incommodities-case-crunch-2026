"""
Comprehensive market-specific pattern analysis for electricity price forecasting.
Analyzes FR net position, UK import dependency, merit order proxies,
price convergence, gas spreads, wind thresholds, and interconnector saturation.
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# LOAD DATA
# ==============================================================================
x = pd.read_csv('/Users/paul/Desktop/Incommodities - Cruch/INCOMO 3/data/raw/x_train.csv')
y = pd.read_csv('/Users/paul/Desktop/Incommodities - Cruch/INCOMO 3/data/raw/y_train.csv')
df = x.merge(y, on='id')

print("=" * 80)
print("DATA OVERVIEW")
print("=" * 80)
print(f"Shape: {df.shape}")
print(f"Date range: {df['datetime_CET'].min()} to {df['datetime_CET'].max()}")
print(f"\nTarget stats:")
print(f"  fr_spot: mean={df['fr_spot'].mean():.2f}, std={df['fr_spot'].std():.2f}, "
      f"min={df['fr_spot'].min():.2f}, max={df['fr_spot'].max():.2f}")
print(f"  uk_spot: mean={df['uk_spot'].mean():.2f}, std={df['uk_spot'].std():.2f}, "
      f"min={df['uk_spot'].min():.2f}, max={df['uk_spot'].max():.2f}")

# Identify column groups
flow_cols = [c for c in df.columns if c.startswith('flow_')]
atc_cols = [c for c in df.columns if c.startswith('atc_')]
ntc_cols = [c for c in df.columns if c.startswith('ntc_')]
cost_cols = [c for c in df.columns if c.startswith('cost_')]
spot_la_cols = [c for c in df.columns if c.endswith('_spot_la')]

print(f"\nFlow columns: {flow_cols}")
print(f"ATC columns: {atc_cols}")
print(f"NTC columns: {ntc_cols}")
print(f"Lagged spot columns: {spot_la_cols}")

# Forward-fill daily columns (gas, emissions)
daily_cols = ['eu_emission', 'uk_emission', 'de_gas', 'es_gas', 'fr_gas', 'nl_gas', 'uk_gas']
for col in daily_cols:
    if col in df.columns:
        df[col] = df[col].ffill()

print(f"\nMissing values in key columns after ffill:")
key_cols = daily_cols + ['fr_spot', 'uk_spot'] + flow_cols
for col in key_cols:
    if col in df.columns:
        pct = df[col].isna().mean() * 100
        if pct > 0:
            print(f"  {col}: {pct:.1f}% missing")

# ==============================================================================
# 1. FR NET POSITION ANALYSIS
# ==============================================================================
print("\n" + "=" * 80)
print("1. FR NET POSITION ANALYSIS")
print("=" * 80)

# Flow columns related to FR
# flow_fr-uk-1_la, flow_fr-uk-2_la, flow_fr-uk-3_la = FR -> UK (positive = FR exporting to UK)
# flow_uk-fr-1_la, flow_uk-fr-2_la, flow_uk-fr-3_la = UK -> FR (positive = UK exporting to FR)
# Note: flows are lagged actuals

# FR -> UK flows (FR exporting)
fr_to_uk_cols = [c for c in flow_cols if c.startswith('flow_fr-')]
uk_to_fr_cols = [c for c in flow_cols if c.startswith('flow_uk-fr')]

print(f"\nFR->UK flow columns: {fr_to_uk_cols}")
print(f"UK->FR flow columns: {uk_to_fr_cols}")

# Compute net FR exports to UK (positive = FR exporting)
df['fr_uk_net_export'] = df[fr_to_uk_cols].sum(axis=1) - df[uk_to_fr_cols].sum(axis=1)

# We don't have direct FR-DE or FR-ES flows, but we can use what we have
# The available flows are all UK-connected interconnectors
# Let's compute FR net position from all available FR flow data
print(f"\nFR-UK net export stats:")
print(f"  Mean: {df['fr_uk_net_export'].mean():.1f} MW")
print(f"  Std:  {df['fr_uk_net_export'].std():.1f} MW")
print(f"  Min:  {df['fr_uk_net_export'].min():.1f} MW")
print(f"  Max:  {df['fr_uk_net_export'].max():.1f} MW")

# Check if FR is typically net exporter or importer to UK
fr_exporting = df['fr_uk_net_export'] > 0
print(f"\n  FR exporting to UK: {fr_exporting.mean()*100:.1f}% of the time")
print(f"  FR importing from UK: {(~fr_exporting).mean()*100:.1f}% of the time")

# Correlation with prices
mask = df['fr_uk_net_export'].notna() & df['fr_spot'].notna()
if mask.sum() > 100:
    corr_fr = df.loc[mask, 'fr_uk_net_export'].corr(df.loc[mask, 'fr_spot'])
    corr_uk = df.loc[mask, 'fr_uk_net_export'].corr(df.loc[mask, 'uk_spot'])
    print(f"\n  Correlation(FR_net_export, fr_spot): {corr_fr:.4f}")
    print(f"  Correlation(FR_net_export, uk_spot): {corr_uk:.4f}")

    # When FR exports, what are prices?
    export_mask = mask & (df['fr_uk_net_export'] > 0)
    import_mask = mask & (df['fr_uk_net_export'] < 0)

    print(f"\n  When FR exports to UK (n={export_mask.sum()}):")
    print(f"    fr_spot mean: {df.loc[export_mask, 'fr_spot'].mean():.2f} EUR/MWh")
    print(f"    uk_spot mean: {df.loc[export_mask, 'uk_spot'].mean():.2f} EUR/MWh")
    print(f"    spread (UK-FR): {(df.loc[export_mask, 'uk_spot'] - df.loc[export_mask, 'fr_spot']).mean():.2f} EUR/MWh")

    print(f"\n  When FR imports from UK (n={import_mask.sum()}):")
    print(f"    fr_spot mean: {df.loc[import_mask, 'fr_spot'].mean():.2f} EUR/MWh")
    print(f"    uk_spot mean: {df.loc[import_mask, 'uk_spot'].mean():.2f} EUR/MWh")
    print(f"    spread (UK-FR): {(df.loc[import_mask, 'uk_spot'] - df.loc[import_mask, 'fr_spot']).mean():.2f} EUR/MWh")

# Quintile analysis
print(f"\n  FR net export quintile analysis:")
valid = df[mask].copy()
valid['net_export_q'] = pd.qcut(valid['fr_uk_net_export'], 5, labels=['Q1(import)', 'Q2', 'Q3', 'Q4', 'Q5(export)'])
for q in ['Q1(import)', 'Q2', 'Q3', 'Q4', 'Q5(export)']:
    subset = valid[valid['net_export_q'] == q]
    print(f"    {q}: fr_spot={subset['fr_spot'].mean():.1f}, uk_spot={subset['uk_spot'].mean():.1f}, "
          f"spread={( subset['uk_spot'] - subset['fr_spot']).mean():.1f}, n={len(subset)}")


# ==============================================================================
# 2. UK IMPORT DEPENDENCY ANALYSIS
# ==============================================================================
print("\n" + "=" * 80)
print("2. UK IMPORT DEPENDENCY ANALYSIS")
print("=" * 80)

# All flows TO UK (imports)
to_uk_cols = [c for c in flow_cols if not c.startswith('flow_uk-')]
from_uk_cols = [c for c in flow_cols if c.startswith('flow_uk-')]

print(f"Flows TO UK: {to_uk_cols}")
print(f"Flows FROM UK: {from_uk_cols}")

df['uk_total_imports'] = df[to_uk_cols].sum(axis=1)
df['uk_total_exports'] = df[from_uk_cols].sum(axis=1)
df['uk_net_imports'] = df['uk_total_imports'] - df['uk_total_exports']

# Import ratio
df['uk_import_ratio'] = df['uk_net_imports'] / df['uk_load_f']

print(f"\nUK net imports stats:")
print(f"  Mean: {df['uk_net_imports'].mean():.1f} MW")
print(f"  Std:  {df['uk_net_imports'].std():.1f} MW")
print(f"  UK typically net importer: {(df['uk_net_imports'] > 0).mean()*100:.1f}%")

mask = df['uk_import_ratio'].notna() & df['uk_spot'].notna() & np.isfinite(df['uk_import_ratio'])
if mask.sum() > 100:
    corr = df.loc[mask, 'uk_import_ratio'].corr(df.loc[mask, 'uk_spot'])
    print(f"\n  Correlation(uk_import_ratio, uk_spot): {corr:.4f}")

    # Quintile analysis
    valid = df[mask].copy()
    valid['import_q'] = pd.qcut(valid['uk_import_ratio'], 5, labels=['Q1(low)', 'Q2', 'Q3', 'Q4', 'Q5(high)'], duplicates='drop')
    print(f"\n  UK import ratio quintile analysis:")
    for q in valid['import_q'].cat.categories:
        subset = valid[valid['import_q'] == q]
        print(f"    {q}: uk_spot={subset['uk_spot'].mean():.1f}, "
              f"import_ratio={subset['uk_import_ratio'].mean():.3f}, "
              f"uk_spot_std={subset['uk_spot'].std():.1f}, n={len(subset)}")

    # Price sensitivity: volatility at different import levels
    print(f"\n  Price sensitivity (std/mean) by import quintile:")
    for q in valid['import_q'].cat.categories:
        subset = valid[valid['import_q'] == q]
        cv = subset['uk_spot'].std() / subset['uk_spot'].mean() if subset['uk_spot'].mean() != 0 else np.nan
        print(f"    {q}: CV={cv:.3f}")


# ==============================================================================
# 3. CONTINENTAL MERIT ORDER PROXY
# ==============================================================================
print("\n" + "=" * 80)
print("3. CONTINENTAL MERIT ORDER PROXY")
print("=" * 80)

# Gas efficiency ~ 50% for CCGT, emission factor for gas ~ 0.37 tCO2/MWh
# Spark spread = gas_price / efficiency + emission_price * emission_factor
# Using efficiency = 0.5 (50%)

mask = df['de_gas'].notna() & df['fr_gas'].notna() & df['eu_emission'].notna() & df['fr_spot'].notna()
print(f"\nValid rows for merit order analysis: {mask.sum()}")

if mask.sum() > 100:
    # DE-based spark spread
    df['de_spark'] = df['de_gas'] / 0.5 + df['eu_emission'] * 0.37
    # FR-based spark spread
    df['fr_spark'] = df['fr_gas'] / 0.5 + df['eu_emission'] * 0.37

    mask2 = mask & df['de_spark'].notna() & df['fr_spark'].notna()

    corr_de_fr = df.loc[mask2, 'de_spark'].corr(df.loc[mask2, 'fr_spot'])
    corr_fr_fr = df.loc[mask2, 'fr_spark'].corr(df.loc[mask2, 'fr_spot'])
    corr_de_de = df.loc[mask2, 'de_spark'].corr(df.loc[mask2, 'de_spot_la'])
    corr_fr_de = df.loc[mask2, 'fr_spark'].corr(df.loc[mask2, 'de_spot_la'])

    print(f"\n  Correlation with fr_spot:")
    print(f"    DE spark spread vs fr_spot: {corr_de_fr:.4f}")
    print(f"    FR spark spread vs fr_spot: {corr_fr_fr:.4f}")
    print(f"    >>> {'DE' if abs(corr_de_fr) > abs(corr_fr_fr) else 'FR'} gas is better predictor of FR spot")

    print(f"\n  Correlation with de_spot_la:")
    print(f"    DE spark spread vs de_spot_la: {corr_de_de:.4f}")
    print(f"    FR spark spread vs de_spot_la: {corr_fr_de:.4f}")

    # Also check for UK
    mask3 = mask2 & df['uk_spot'].notna() & df['uk_gas'].notna()
    if mask3.sum() > 100:
        df['uk_spark'] = df['uk_gas'] / 0.5 + df['uk_emission'] * 0.37
        corr_uk_uk = df.loc[mask3, 'uk_spark'].corr(df.loc[mask3, 'uk_spot'])
        corr_de_uk = df.loc[mask3, 'de_spark'].corr(df.loc[mask3, 'uk_spot'])
        corr_fr_uk = df.loc[mask3, 'fr_spark'].corr(df.loc[mask3, 'uk_spot'])

        print(f"\n  Correlation with uk_spot:")
        print(f"    UK spark spread vs uk_spot: {corr_uk_uk:.4f}")
        print(f"    DE spark spread vs uk_spot: {corr_de_uk:.4f}")
        print(f"    FR spark spread vs uk_spot: {corr_fr_uk:.4f}")

    # Cross-correlation of gas prices
    gas_cols_avail = ['de_gas', 'fr_gas', 'nl_gas', 'uk_gas', 'es_gas']
    print(f"\n  Gas price cross-correlations:")
    gas_mask = df[gas_cols_avail].notna().all(axis=1)
    if gas_mask.sum() > 100:
        gas_corr = df.loc[gas_mask, gas_cols_avail].corr()
        print(gas_corr.to_string())

    # Spark spread stats
    print(f"\n  Spark spread stats:")
    for name, col in [('DE', 'de_spark'), ('FR', 'fr_spark'), ('UK', 'uk_spark')]:
        if col in df.columns:
            valid = df[col].dropna()
            print(f"    {name}: mean={valid.mean():.1f}, std={valid.std():.1f}, "
                  f"min={valid.min():.1f}, max={valid.max():.1f}")


# ==============================================================================
# 4. PRICE CONVERGENCE ANALYSIS
# ==============================================================================
print("\n" + "=" * 80)
print("4. PRICE CONVERGENCE ANALYSIS")
print("=" * 80)

# FR vs DE convergence
mask_frde = df['fr_spot'].notna() & df['de_spot_la'].notna()
if mask_frde.sum() > 100:
    df['fr_de_spread'] = df['fr_spot'] - df['de_spot_la']
    converged = (df.loc[mask_frde, 'fr_de_spread'].abs() < 1)

    print(f"\n  FR-DE Price Convergence (within 1 EUR):")
    print(f"    Convergence rate: {converged.mean()*100:.1f}%")
    print(f"    Mean absolute spread: {df.loc[mask_frde, 'fr_de_spread'].abs().mean():.2f} EUR/MWh")
    print(f"    Median absolute spread: {df.loc[mask_frde, 'fr_de_spread'].abs().median():.2f} EUR/MWh")
    print(f"    Spread std: {df.loc[mask_frde, 'fr_de_spread'].std():.2f} EUR/MWh")

    # What drives divergence?
    diverged_mask = mask_frde & (df['fr_de_spread'].abs() >= 5)
    converged_mask = mask_frde & (df['fr_de_spread'].abs() < 1)

    drivers = ['fr_wind_f', 'de_wind_f', 'fr_load_f', 'de_load_f', 'fr_solar_f', 'de_solar_f',
               'fr_nuclear_avcap_f']

    print(f"\n  Drivers of FR-DE divergence (|spread| >= 5 EUR, n={diverged_mask.sum()}):")
    print(f"  vs convergence (|spread| < 1 EUR, n={converged_mask.sum()}):")
    for driver in drivers:
        if driver in df.columns:
            div_mean = df.loc[diverged_mask, driver].mean()
            conv_mean = df.loc[converged_mask, driver].mean()
            diff_pct = (div_mean - conv_mean) / conv_mean * 100 if conv_mean != 0 else np.nan
            print(f"    {driver}: diverged={div_mean:.1f}, converged={conv_mean:.1f}, diff={diff_pct:+.1f}%")

# FR vs UK convergence
mask_fruk = df['fr_spot'].notna() & df['uk_spot'].notna()
if mask_fruk.sum() > 100:
    df['fr_uk_spread'] = df['fr_spot'] - df['uk_spot']
    converged_uk = (df.loc[mask_fruk, 'fr_uk_spread'].abs() < 1)

    print(f"\n  FR-UK Price Convergence (within 1 EUR):")
    print(f"    Convergence rate: {converged_uk.mean()*100:.1f}%")
    print(f"    Mean absolute spread: {df.loc[mask_fruk, 'fr_uk_spread'].abs().mean():.2f} EUR/MWh")
    print(f"    FR typically {'above' if df.loc[mask_fruk, 'fr_uk_spread'].mean() > 0 else 'below'} UK")
    print(f"    Mean spread (FR-UK): {df.loc[mask_fruk, 'fr_uk_spread'].mean():.2f} EUR/MWh")

    # Divergence drivers for FR-UK
    diverged_uk_mask = mask_fruk & (df['fr_uk_spread'].abs() >= 10)
    converged_uk_mask = mask_fruk & (df['fr_uk_spread'].abs() < 1)

    uk_drivers = ['uk_wind_f', 'fr_wind_f', 'uk_load_f', 'fr_load_f',
                  'uk_gas_avcap_f', 'fr_nuclear_avcap_f']
    atc_fr_uk = [c for c in atc_cols if 'fr-uk' in c or 'uk-fr' in c]
    uk_drivers += atc_fr_uk

    print(f"\n  Drivers of FR-UK divergence (|spread| >= 10 EUR, n={diverged_uk_mask.sum()}):")
    print(f"  vs convergence (|spread| < 1 EUR, n={converged_uk_mask.sum()}):")
    for driver in uk_drivers:
        if driver in df.columns:
            div_mean = df.loc[diverged_uk_mask, driver].mean()
            conv_mean = df.loc[converged_uk_mask, driver].mean()
            diff_pct = (div_mean - conv_mean) / conv_mean * 100 if conv_mean != 0 else np.nan
            print(f"    {driver}: diverged={div_mean:.1f}, converged={conv_mean:.1f}, diff={diff_pct:+.1f}%")


# ==============================================================================
# 5. UK GAS vs EU GAS SPREAD
# ==============================================================================
print("\n" + "=" * 80)
print("5. UK GAS vs EU GAS SPREAD (NBP vs TTF proxy)")
print("=" * 80)

mask_gas = df['uk_gas'].notna() & df['nl_gas'].notna()
if mask_gas.sum() > 100:
    df['gas_spread'] = df['uk_gas'] - df['nl_gas']

    print(f"\n  UK-NL gas spread stats (n={mask_gas.sum()}):")
    print(f"    Mean: {df.loc[mask_gas, 'gas_spread'].mean():.2f} EUR/MWh")
    print(f"    Std:  {df.loc[mask_gas, 'gas_spread'].std():.2f} EUR/MWh")
    print(f"    Min:  {df.loc[mask_gas, 'gas_spread'].min():.2f} EUR/MWh")
    print(f"    Max:  {df.loc[mask_gas, 'gas_spread'].max():.2f} EUR/MWh")
    print(f"    UK gas > NL gas: {(df.loc[mask_gas, 'gas_spread'] > 0).mean()*100:.1f}%")

    mask_spot = mask_gas & df['uk_spot'].notna() & df['fr_spot'].notna()
    corr_uk = df.loc[mask_spot, 'gas_spread'].corr(df.loc[mask_spot, 'uk_spot'])
    corr_fr = df.loc[mask_spot, 'gas_spread'].corr(df.loc[mask_spot, 'fr_spot'])

    print(f"\n  Correlation with spot prices:")
    print(f"    gas_spread vs uk_spot: {corr_uk:.4f}")
    print(f"    gas_spread vs fr_spot: {corr_fr:.4f}")

    # Does gas spread predict power price differential?
    df['power_spread_uk_fr'] = df['uk_spot'] - df['fr_spot']
    corr_power = df.loc[mask_spot, 'gas_spread'].corr(df.loc[mask_spot, 'power_spread_uk_fr'])
    print(f"    gas_spread vs power_spread(UK-FR): {corr_power:.4f}")

    # Quintile analysis of gas spread
    valid = df[mask_spot].copy()
    valid['gas_q'] = pd.qcut(valid['gas_spread'], 5, labels=['Q1(UK cheap)', 'Q2', 'Q3', 'Q4', 'Q5(UK expensive)'], duplicates='drop')
    print(f"\n  Gas spread quintile analysis:")
    for q in valid['gas_q'].cat.categories:
        subset = valid[valid['gas_q'] == q]
        print(f"    {q}: uk_spot={subset['uk_spot'].mean():.1f}, fr_spot={subset['fr_spot'].mean():.1f}, "
              f"power_spread={subset['power_spread_uk_fr'].mean():.1f}, n={len(subset)}")


# ==============================================================================
# 6. WIND PENETRATION THRESHOLDS — NON-LINEAR EFFECTS
# ==============================================================================
print("\n" + "=" * 80)
print("6. WIND PENETRATION THRESHOLDS — NON-LINEAR EFFECTS")
print("=" * 80)

# UK wind penetration
mask_uk_wind = df['uk_wind_f'].notna() & df['uk_load_f'].notna() & df['uk_spot'].notna()
df['uk_wind_pen'] = df['uk_wind_f'] / df['uk_load_f']

if mask_uk_wind.sum() > 100:
    print(f"\n  UK Wind Penetration Analysis (n={mask_uk_wind.sum()}):")
    print(f"    Mean penetration: {df.loc[mask_uk_wind, 'uk_wind_pen'].mean()*100:.1f}%")
    print(f"    Max penetration: {df.loc[mask_uk_wind, 'uk_wind_pen'].max()*100:.1f}%")

    # Decile analysis for non-linearity
    valid = df[mask_uk_wind].copy()
    valid['wind_dec'] = pd.qcut(valid['uk_wind_pen'], 10, duplicates='drop')
    print(f"\n  UK wind penetration decile analysis:")
    for dec in sorted(valid['wind_dec'].unique()):
        subset = valid[valid['wind_dec'] == dec]
        print(f"    {str(dec):30s}: uk_spot_mean={subset['uk_spot'].mean():7.1f}, "
              f"uk_spot_median={subset['uk_spot'].median():7.1f}, "
              f"uk_spot_std={subset['uk_spot'].std():7.1f}, n={len(subset)}")

    # Specific thresholds
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    print(f"\n  UK price at wind penetration thresholds:")
    for i in range(len(thresholds) - 1):
        low, high = thresholds[i], thresholds[i+1]
        band = valid[(valid['uk_wind_pen'] >= low) & (valid['uk_wind_pen'] < high)]
        if len(band) > 10:
            print(f"    {low*100:.0f}%-{high*100:.0f}%: mean={band['uk_spot'].mean():.1f}, "
                  f"median={band['uk_spot'].median():.1f}, n={len(band)}")

    # Cliff detection: rate of change in mean price per penetration increment
    print(f"\n  Marginal price impact per 5% wind penetration increase:")
    steps = np.arange(0, 0.7, 0.05)
    prev_price = None
    for i in range(len(steps) - 1):
        low, high = steps[i], steps[i+1]
        band = valid[(valid['uk_wind_pen'] >= low) & (valid['uk_wind_pen'] < high)]
        if len(band) > 10:
            mean_price = band['uk_spot'].mean()
            delta = mean_price - prev_price if prev_price is not None else 0
            prev_price = mean_price
            cliff_marker = " <-- CLIFF" if delta < -15 else (" <-- BIG DROP" if delta < -10 else "")
            print(f"    {low*100:5.0f}%-{high*100:5.0f}%: mean_price={mean_price:7.1f}, "
                  f"delta={delta:+7.1f}, n={len(band)}{cliff_marker}")

# DE wind effect on FR prices
mask_de_wind = df['de_wind_f'].notna() & df['de_load_f'].notna() & df['fr_spot'].notna()
df['de_wind_pen'] = df['de_wind_f'] / df['de_load_f']

if mask_de_wind.sum() > 100:
    print(f"\n  DE Wind Penetration -> FR Price Analysis (n={mask_de_wind.sum()}):")
    valid = df[mask_de_wind].copy()

    print(f"    Mean DE wind penetration: {valid['de_wind_pen'].mean()*100:.1f}%")

    # Decile analysis
    valid['de_wind_dec'] = pd.qcut(valid['de_wind_pen'], 10, duplicates='drop')
    print(f"\n  DE wind penetration decile -> FR spot price:")
    for dec in sorted(valid['de_wind_dec'].unique()):
        subset = valid[valid['de_wind_dec'] == dec]
        print(f"    {str(dec):30s}: fr_spot_mean={subset['fr_spot'].mean():7.1f}, "
              f"fr_spot_std={subset['fr_spot'].std():7.1f}, n={len(subset)}")

    # Cliff detection for DE wind -> FR spot
    print(f"\n  DE wind -> FR price marginal impact per 5% increment:")
    steps = np.arange(0, 1.0, 0.05)
    prev_price = None
    for i in range(len(steps) - 1):
        low, high = steps[i], steps[i+1]
        band = valid[(valid['de_wind_pen'] >= low) & (valid['de_wind_pen'] < high)]
        if len(band) > 10:
            mean_price = band['fr_spot'].mean()
            delta = mean_price - prev_price if prev_price is not None else 0
            prev_price = mean_price
            cliff_marker = " <-- CLIFF" if delta < -15 else (" <-- BIG DROP" if delta < -10 else "")
            print(f"    {low*100:5.0f}%-{high*100:5.0f}%: fr_spot_mean={mean_price:7.1f}, "
                  f"delta={delta:+7.1f}, n={len(band)}{cliff_marker}")


# ==============================================================================
# 7. INTERCONNECTOR SATURATION AND PRICE DECOUPLING
# ==============================================================================
print("\n" + "=" * 80)
print("7. INTERCONNECTOR SATURATION AND PRICE DECOUPLING")
print("=" * 80)

# ATC columns for FR-UK
atc_fr_uk_cols = [c for c in atc_cols if 'fr-uk' in c]
atc_uk_fr_cols = [c for c in atc_cols if 'uk-fr' in c]
ntc_fr_uk_cols = [c for c in ntc_cols if 'fr-uk' in c]
ntc_uk_fr_cols = [c for c in ntc_cols if 'uk-fr' in c]

print(f"\n  ATC FR->UK columns: {atc_fr_uk_cols}")
print(f"  ATC UK->FR columns: {atc_uk_fr_cols}")
print(f"  NTC FR->UK columns: {ntc_fr_uk_cols}")
print(f"  NTC UK->FR columns: {ntc_uk_fr_cols}")

# Total ATC FR->UK
df['atc_fr_uk_total'] = df[atc_fr_uk_cols].sum(axis=1)
df['ntc_fr_uk_total'] = df[ntc_fr_uk_cols].sum(axis=1)
df['atc_uk_fr_total'] = df[atc_uk_fr_cols].sum(axis=1)
df['ntc_uk_fr_total'] = df[ntc_uk_fr_cols].sum(axis=1)

# ATC ratio (available / nominal)
df['atc_ratio_fr_uk'] = df['atc_fr_uk_total'] / df['ntc_fr_uk_total']
df['atc_ratio_uk_fr'] = df['atc_uk_fr_total'] / df['ntc_uk_fr_total']

# Combined ATC ratio (both directions)
df['atc_ratio_combined'] = (df['atc_fr_uk_total'] + df['atc_uk_fr_total']) / (df['ntc_fr_uk_total'] + df['ntc_uk_fr_total'])

# Price spread
df['uk_fr_price_spread'] = (df['uk_spot'] - df['fr_spot']).abs()

mask_atc = (df['atc_ratio_combined'].notna() & df['uk_fr_price_spread'].notna() &
            np.isfinite(df['atc_ratio_combined']) & (df['atc_ratio_combined'] >= 0))

if mask_atc.sum() > 100:
    corr = df.loc[mask_atc, 'atc_ratio_combined'].corr(df.loc[mask_atc, 'uk_fr_price_spread'])
    print(f"\n  Correlation(ATC_ratio, |UK-FR spread|): {corr:.4f}")

    # Low vs high ATC
    low_atc = mask_atc & (df['atc_ratio_combined'] < 0.2)
    med_atc = mask_atc & (df['atc_ratio_combined'] >= 0.4) & (df['atc_ratio_combined'] <= 0.6)
    high_atc = mask_atc & (df['atc_ratio_combined'] > 0.8)

    print(f"\n  Price spread by ATC congestion level:")
    print(f"    Low ATC ratio (<0.2, congested, n={low_atc.sum()}):")
    if low_atc.sum() > 0:
        print(f"      Mean |UK-FR spread|: {df.loc[low_atc, 'uk_fr_price_spread'].mean():.2f} EUR/MWh")
        print(f"      Median |UK-FR spread|: {df.loc[low_atc, 'uk_fr_price_spread'].median():.2f} EUR/MWh")
        print(f"      Mean FR spot: {df.loc[low_atc, 'fr_spot'].mean():.2f}")
        print(f"      Mean UK spot: {df.loc[low_atc, 'uk_spot'].mean():.2f}")

    print(f"    Medium ATC ratio (0.4-0.6, n={med_atc.sum()}):")
    if med_atc.sum() > 0:
        print(f"      Mean |UK-FR spread|: {df.loc[med_atc, 'uk_fr_price_spread'].mean():.2f} EUR/MWh")
        print(f"      Median |UK-FR spread|: {df.loc[med_atc, 'uk_fr_price_spread'].median():.2f} EUR/MWh")

    print(f"    High ATC ratio (>0.8, uncongested, n={high_atc.sum()}):")
    if high_atc.sum() > 0:
        print(f"      Mean |UK-FR spread|: {df.loc[high_atc, 'uk_fr_price_spread'].mean():.2f} EUR/MWh")
        print(f"      Median |UK-FR spread|: {df.loc[high_atc, 'uk_fr_price_spread'].median():.2f} EUR/MWh")
        print(f"      Mean FR spot: {df.loc[high_atc, 'fr_spot'].mean():.2f}")
        print(f"      Mean UK spot: {df.loc[high_atc, 'uk_spot'].mean():.2f}")

    # Quintile analysis
    valid = df[mask_atc].copy()
    valid['atc_q'] = pd.qcut(valid['atc_ratio_combined'], 5,
                              labels=['Q1(congested)', 'Q2', 'Q3', 'Q4', 'Q5(uncongested)'],
                              duplicates='drop')
    print(f"\n  ATC ratio quintile analysis:")
    for q in valid['atc_q'].cat.categories:
        subset = valid[valid['atc_q'] == q]
        print(f"    {q}: |spread|_mean={subset['uk_fr_price_spread'].mean():.1f}, "
              f"|spread|_median={subset['uk_fr_price_spread'].median():.1f}, "
              f"fr_spot={subset['fr_spot'].mean():.1f}, uk_spot={subset['uk_spot'].mean():.1f}, "
              f"n={len(subset)}")

    # Direction analysis: when congested, who has higher prices?
    low_atc_mask = mask_atc & (df['atc_ratio_combined'] < 0.2)
    if low_atc_mask.sum() > 0:
        uk_higher = (df.loc[low_atc_mask, 'uk_spot'] > df.loc[low_atc_mask, 'fr_spot']).mean() * 100
        print(f"\n  When congested (ATC<0.2): UK price > FR price {uk_higher:.1f}% of the time")


# ==============================================================================
# SUMMARY: FEATURE ENGINEERING RECOMMENDATIONS
# ==============================================================================
print("\n" + "=" * 80)
print("SUMMARY: FEATURE ENGINEERING RECOMMENDATIONS")
print("=" * 80)
print("""
Based on the analysis above, consider these features for the forecasting model:

1. FR Net Position (FR-UK net export): Strong indicator of FR supply/demand balance.
   Feature: fr_uk_net_export = sum(flow_fr-uk) - sum(flow_uk-fr) [lagged]

2. UK Import Ratio: uk_net_imports / uk_load_f
   Feature: uk_import_ratio (captures import dependency)

3. Continental Spark Spread: Use BOTH de_gas and fr_gas based spark spreads.
   Features: de_spark = de_gas/0.5 + eu_emission*0.37
             fr_spark = fr_gas/0.5 + eu_emission*0.37
             uk_spark = uk_gas/0.5 + uk_emission*0.37

4. Price Convergence Indicators:
   Features: fr_de_spread_la (lagged), fr_uk_spread_la (lagged)
   Binary: fr_de_converged (abs spread < 1)

5. Gas Spread (NBP-TTF): uk_gas - nl_gas
   Feature: gas_spread (predicts power price differential)

6. Wind Penetration: Non-linear effects captured via:
   Features: uk_wind_pen = uk_wind_f / uk_load_f
             de_wind_pen = de_wind_f / de_load_f
   Consider: piecewise linear or polynomial terms

7. ATC Ratio (interconnector congestion):
   Feature: atc_ratio_combined = total_ATC / total_NTC
   Binary: interconnector_congested (ratio < 0.2)
""")

print("\nAnalysis complete.")
