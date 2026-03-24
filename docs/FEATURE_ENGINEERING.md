# InCommodities Case Crunch 2026: Feature Engineering Pipeline

~290 engineered features across 30 categories, built on top of the 111 raw columns.
All transformations are **stateless** (train = test behaviour) and **leakage-free** (only lagged targets used, rolling windows are trailing-only).

---

## Category Overview

| Cat | Function | # Features | Description |
|-----|----------|------------|-------------|
| 1 | `engineer_calendar_effects` | 21 | Hour/DoW/month raw + cyclical sin/cos + binary period flags |
| 2 | `engineer_holiday_features` | 10 | FR/UK/DE/BE/NL holidays, bridge days, distance to next holiday |
| 3 | `engineer_fundamental_market` | 23 | Residual loads, spark spreads, baseload gaps, supply-demand ratios |
| 4 | `engineer_renewable_penetration` | 12 | Wind/solar penetration FR/UK/DE + continental + UK threshold flags |
| 5 | `engineer_renewable_dynamics` | 15 | Ramps (1h/3h/6h), 24h changes, 24h rolling volatility |
| 6 | `engineer_nuclear_features` | 11 | Changes (24h/48h/168h), threshold flags, % of load, rolling deviation |
| 7 | `engineer_hydro_features` | 9 | Alpine hydro, reservoir share, nuclear+hydro combined, baseload surplus |
| 8 | `engineer_interconnectors` | 23 | ATC/NTC totals, utilization rates, net flows, cost spreads, congestion |
| 9 | `engineer_price_features` | 19 | Multi-horizon lags, cross-zone spreads, continental average, 24h/168h changes |
| 10 | `engineer_interaction_features` | 13 | Thermal_need×gas, wind×hour, solar×load, congestion×spark_diff |
| 11 | `engineer_regional_features` | 11 | Continental aggregates, Nordic wind, Iberian, Benelux, Alpine |
| 12 | `engineer_river_temperature` | 12 | Binary threshold flags, excess temp, river temp × nuclear_low interaction |
| 13 | `engineer_nonlinear_transforms` | 11 | Squared loads, cubed thermal need, log-spark, sqrt gas, clipped wind |
| 14 | `engineer_rolling_statistics` | 24 | 24h/168h mean/std/min/max for spot; gas/emission rolling means; deviations |
| 15 | `engineer_momentum_features` | 8 | Price acceleration (2nd derivative), gas momentum, nuclear trend, EWM |
| 16 | `engineer_advanced_supply_demand` | 8 | Residual v2 (minus run-of-river), security margin, scarcity critical/extreme |
| 17 | `engineer_residual_ramps` | 8 | 1h and 3h ramps for load and residual load (duck-curve pressure) |
| 18 | `engineer_multi_efficiency_spark` | 4 | OCGT (40% eff) and CCGT (55% eff) marginal costs for FR and UK |
| 19 | `engineer_advanced_interconnection` | 5 | Flow/ATC ratio, unused capacity per cable, total unused to UK |
| 20 | `engineer_regime_zscores` | 8 | 14-day z-scores for residual, load, wind; lag reliability ratio |
| 21 | `engineer_stochastic_signals` | 10 | Jump count/magnitude (24h/48h), vol ratio (24h/168h), mean reversion |
| 22 | `engineer_asinh_transforms` | 4 | arcsinh for spot_la and spark spreads (handles negatives without discontinuity) |
| 23 | `engineer_nuclear_shortfall` | 2 | Expanding-max gap: `max_seen_so_far − current_availability` |
| 24 | `engineer_atc_ratios` | 6 | Per-cable ATC/NTC ratio for all 6 FR↔UK cables |
| 25 | `engineer_market_specific_features` | 10 | SDAC/N2EX: gas spread, continental floor, import ratio, intraday anchors, dark doldrums |
| 26 | `engineer_price_formation_signals` | 7 | Nuke×gas interaction, implied RE surplus, UK cheapest import, net capacity cost |
| 27 | `engineer_fr_continent_features` | 12 | Continental merit order, zero-MC pen, euro scarcity, wind-nuke gap, ES fundamentals |
| 28 | `engineer_uk_island_features` | 6 | Gas utilization, gas headroom, capacity margin, gas cost per MW, self-sufficiency |
| 29 | `engineer_regime_features` | 11 | Iberian exception flag, gas-on-margin binary, oversupply/negative-price risk, nuclear avail ratio |
| 30 | `engineer_advanced_price_proxies` | 14 | Dynamic marginal, opportunity cost, scarcity barrier, load-price signal, basis v2 |

---

## Key Physical Assumptions

### Thermal Floor / Spark Spread
```
spark_spread = gas_price / efficiency + emission_price × emission_factor
```
- **Baseline (Cat 3):** 50% efficiency, 0.37 tCO2/MWh → `fr_spark_spread`, `uk_spark_spread`
- **CCGT (Cat 18):** 55% efficiency → `fr_spark_ccgt`, `uk_spark_ccgt`
- **OCGT (Cat 18):** 40% efficiency → `fr_spark_ocgt`, `uk_spark_ocgt`
- **Merit order cost (Cat 25):** Interpolates CCGT↔OCGT based on scarcity ratio

### Nuclear Shortfall (Cat 23)
Uses `expanding().max()` (the highest availability seen so far) rather than a hardcoded constant. This eliminates look-ahead bias while still capturing the crisis/recovery arc.

### Scarcity Features (Cat 16)
```
scarcity_ratio = residual_load / (nuclear_cap + gas_cap)
security_margin = (nuclear_cap + gas_cap) - residual_load
```
Critical threshold (>0.85) and extreme threshold (>0.95) binary flags capture convex merit-order effects.

### Scarcity Barrier (Cat 30)
```
scarcity_barrier = spark_spread × (1 / (1 − scarcity)^1.5)
```
Models the exponential price explosion as the system approaches capacity limits.

---

## Leakage Prevention

1. **Rolling windows:** All use `center=False` (trailing-only).
2. **Shift convention:** `fr_spot_la` is already T-24 lagged. Additional shifts add further days: `.shift(24)` → T-48, `.shift(144)` → T-168.
3. **Expanding max** in nuclear shortfall only uses data seen before each timestamp.
4. **Regime dummies** (e.g., `iberian_exception`) are hardcoded date ranges, not derived from price data.

---

## Config Parameters (`config.yaml → feature_engineering`)

| Parameter | Default | Usage |
|-----------|---------|-------|
| `gas_efficiency` | 0.50 | Baseline spark spread denominator |
| `emission_factor` | 0.37 | tCO2/MWh for spark spread |
| `ocgt_efficiency` | 0.40 | Cat 18 OCGT spark spread |
| `ccgt_efficiency` | 0.55 | Cat 18 CCGT spark spread |
| `nuclear_marginal_cost` | 12.0 | EUR/MWh for dynamic marginal cost (Cat 30) |
| `transport_cost_approx` | 2.0 | EUR/MWh cross-border transport (Cat 30) |
| `river_hot_threshold_fr` | 25.0 | °C threshold for FR nuclear derating |
| `river_hot_threshold_de` | 23.0 | °C threshold for DE thermal derating |
| `nuclear_low_threshold` | 35000 | MW below = FR nuclear "low" flag |
| `nuclear_very_low_threshold` | 25000 | MW below = FR nuclear "very low" flag |
| `scarcity_critical_threshold` | 0.85 | Cat 16 scarcity critical flag |
| `scarcity_extreme_threshold` | 0.95 | Cat 16 scarcity extreme flag |
| `jump_threshold` | 50.0 | EUR/MWh change defining a price jump (Cat 21) |
| `scarcity_barrier_power` | 1.5 | Exponent for convex barrier (Cat 30) |

---

## Empirically Validated Features (Paul's Research)

The following were validated by partial correlation analysis:

| Feature | Partial r (FR) | Partial r (UK) | Controls |
|---------|---------------|---------------|----------|
| `continent_thermal_need` | 0.534 | — | spark_spread |
| `continent_zero_mc_pen` | −0.550 | — | spark_spread |
| `euro_scarcity_ratio` | 0.618 | — | spark_spread |
| `continent_weighted_price` | 0.609 | — | spark_spread |
| `uk_gas_cost_per_mw` | — | 0.663 | spot_la |
| `uk_gas_utilization` | — | 0.513 | spark_spread |
| `uk_capacity_margin` | — | −0.517 | spark_spread |
| `uk_self_sufficiency` | — | −0.509 | spark_spread |
| `fr_nuke_shortfall_x_gas` | 0.768 | 0.763 | shortfall alone |
| `uk_cheapest_import` | — | 0.433 (partial) | fr_spot_la |
