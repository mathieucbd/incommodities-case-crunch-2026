# INCOMO 3 — Catalogue Complet des Features (~340 engineered + 111 raw = ~450 colonnes)

**Fichier source** : `src/feature_engineering.py`
**Pipeline** : `build_features(df, config) -> DataFrame` — stateless, identique train/test.

---

## Cat 1 — Pre-processing (0 nouvelles colonnes, 7 modifiees)

Forward-fill des colonnes daily (NaN sauf a 00:00 CET) :
`de_gas`, `es_gas`, `fr_gas`, `nl_gas`, `uk_gas`, `eu_emission`, `uk_emission`

---

## Cat 2 — Calendrier / Temps (21 features)

| Feature | Formule / Description |
|---------|----------------------|
| `hour` | Heure CET (0-23) |
| `day_of_week` | Jour de la semaine (0=lundi) |
| `month` | Mois (1-12) |
| `day_of_year` | Jour de l'annee (1-366) |
| `week_of_year` | Semaine ISO (1-53) |
| `quarter` | Trimestre (1-4) |
| `hour_sin` | sin(2π · hour/24) |
| `hour_cos` | cos(2π · hour/24) |
| `dow_sin` | sin(2π · dow/7) |
| `dow_cos` | cos(2π · dow/7) |
| `month_sin` | sin(2π · month/12) |
| `month_cos` | cos(2π · month/12) |
| `doy_sin` | sin(2π · doy/365.25) |
| `doy_cos` | cos(2π · doy/365.25) |
| `is_weekend` | Samedi ou dimanche |
| `is_business_hour` | Lun-Ven, 8h-19h |
| `is_morning_ramp` | 6h-9h |
| `is_evening_peak` | 17h-20h |
| `is_night` | 23h-5h |
| `is_solar_hours` | 10h-16h |
| `hour_x_dow` | hour × 7 + dow (interaction) |

---

## Cat 3 — Jours feries (10 features)

| Feature | Description |
|---------|-------------|
| `is_holiday_fr` | Jour ferie France |
| `is_holiday_uk` | Jour ferie UK |
| `is_holiday_de` | Jour ferie Allemagne |
| `is_holiday_be` | Jour ferie Belgique |
| `is_holiday_nl` | Jour ferie Pays-Bas |
| `is_bridge_day_fr` | Pont (lundi apres ferie dimanche / vendredi avant ferie samedi) |
| `is_holiday_or_weekend_fr` | Ferie OU weekend FR |
| `is_holiday_or_weekend_uk` | Ferie OU weekend UK |
| `days_to_next_holiday_fr` | Jours avant prochain ferie FR (cap 30) |
| `days_since_last_holiday_fr` | Jours depuis dernier ferie FR (cap 30) |

---

## Cat 4 — Fondamentaux marche (23 features)

| Feature | Formule |
|---------|---------|
| `fr_residual_load` | load_f − solar_f − wind_f |
| `uk_residual_load` | load_f − solar_f − wind_f |
| `de_residual_load` | load_f − solar_f − wind_f |
| `fr_thermal_need` | residual_load − nuclear_avcap_f |
| `uk_thermal_need` | residual_load − nuclear_avcap_f |
| `fr_spark_spread` | gas/0.50 + emission×0.37 |
| `uk_spark_spread` | gas/0.50 + emission×0.37 |
| `de_spark_spread` | gas/0.50 + emission×0.37 |
| `nl_spark_spread` | gas/0.50 + emission×0.37 |
| `fr_hydro_total` | hydro_res_f + hydro_ror_f |
| `fr_baseload_gap` | load − nuclear − hydro_total |
| `uk_baseload_gap` | load − nuclear − biomass |
| `fr_thermal_need_pos` | max(thermal_need, 0) |
| `uk_thermal_need_pos` | max(thermal_need, 0) |
| `fr_baseload_gap_pos` | max(baseload_gap, 0) |
| `fr_spot_minus_spark` | spot_la − spark_spread |
| `uk_spot_minus_spark` | spot_la − spark_spread |
| `fr_gas_margin` | gas_avcap_f − thermal_need_pos |
| `uk_gas_margin` | gas_avcap_f − thermal_need_pos |
| `fr_total_dispatchable` | nuclear + gas + hydro |
| `uk_total_dispatchable` | nuclear + gas + biomass |
| `fr_supply_demand_ratio` | total_dispatchable / load |
| `uk_supply_demand_ratio` | (total_dispatchable + wind) / load |

---

## Cat 5 — Penetration renouvelables (12 features)

| Feature | Formule |
|---------|---------|
| `fr_wind_pen` | wind_f / load_f |
| `uk_wind_pen` | wind_f / load_f |
| `fr_solar_pen` | solar_f / load_f |
| `uk_solar_pen` | solar_f / load_f |
| `fr_renewable_pen` | (wind + solar) / load |
| `uk_renewable_pen` | (wind + solar) / load |
| `de_wind_pen` | wind_f / load_f |
| `de_solar_pen` | solar_f / load_f |
| `de_renewable_pen` | (wind + solar) / load |
| `continental_wind_pen` | continental_wind / continental_load |
| `uk_wind_high` | wind_pen > 0.50 |
| `uk_wind_very_high` | wind_pen > 0.65 |

---

## Cat 6 — Dynamique renouvelables (15 features)

| Feature | Formule |
|---------|---------|
| `uk_wind_ramp_1h` | wind_f.diff(1) |
| `uk_wind_ramp_3h` | wind_f.diff(3) |
| `uk_wind_ramp_6h` | wind_f.diff(6) |
| `fr_wind_ramp_3h` | wind_f.diff(3) |
| `fr_solar_ramp_3h` | solar_f.diff(3) |
| `de_wind_ramp_3h` | wind_f.diff(3) |
| `fr_wind_change_24h` | wind_f − wind_f.shift(24) |
| `uk_wind_change_24h` | wind_f − wind_f.shift(24) |
| `fr_solar_change_24h` | solar_f − solar_f.shift(24) |
| `fr_load_change_24h` | load_f − load_f.shift(24) |
| `uk_load_change_24h` | load_f − load_f.shift(24) |
| `fr_residual_change_24h` | residual − residual.shift(24) |
| `uk_residual_change_24h` | residual − residual.shift(24) |
| `fr_wind_volatility_24h` | wind_f.rolling(24).std() |
| `uk_wind_volatility_24h` | wind_f.rolling(24).std() |

---

## Cat 7 — Nucleaire (11 features)

| Feature | Formule |
|---------|---------|
| `fr_nuclear_change_24h` | nuclear.diff(24) |
| `fr_nuclear_change_48h` | nuclear.diff(48) |
| `fr_nuclear_change_168h` | nuclear.diff(168) |
| `uk_nuclear_change_24h` | nuclear.diff(24) |
| `fr_nuclear_low` | nuclear < 35,000 MW |
| `fr_nuclear_very_low` | nuclear < 25,000 MW |
| `fr_nuclear_pct_of_load` | nuclear / load |
| `uk_nuclear_pct_of_load` | nuclear / load |
| `fr_nuclear_rolling_7d_mean` | nuclear.rolling(168).mean() |
| `fr_nuclear_deviation_from_7d` | nuclear − rolling_7d_mean |
| `fr_nuclear_ramp_magnitude` | |nuclear_change_24h| |

---

## Cat 8 — Hydro (9 features)

| Feature | Formule |
|---------|---------|
| `ch_hydro_total` | CH hydro_res + hydro_ror |
| `at_hydro_total` | AT hydro_res + hydro_ror |
| `alpine_hydro_total` | CH + AT hydro |
| `fr_hydro_change_24h` | hydro_total.diff(24) |
| `alpine_hydro_change_168h` | alpine.diff(168) |
| `fr_hydro_res_share` | hydro_res / hydro_total |
| `fr_nuclear_plus_hydro` | nuclear + hydro_total |
| `fr_baseload_surplus` | nuclear + hydro − load |

---

## Cat 9 — Interconnecteurs (23 features)

| Feature | Formule |
|---------|---------|
| `fr_uk_atc_total` | somme ATC FR→UK (3 cables) |
| `uk_fr_atc_total` | somme ATC UK→FR (3 cables) |
| `fr_uk_ntc_total` | somme NTC FR→UK (3 cables) |
| `uk_fr_ntc_total` | somme NTC UK→FR (3 cables) |
| `all_to_uk_atc` | somme ATC tous pays → UK |
| `all_from_uk_atc` | somme ATC UK → tous pays |
| `fr_uk_utilization` | 1 − ATC/NTC (taux congestion) |
| `uk_fr_utilization` | 1 − ATC/NTC |
| `be_uk_utilization` | 1 − ATC/NTC |
| `nl_uk_utilization` | 1 − ATC/NTC |
| `max_utilization_to_uk` | max(fr_uk, be_uk, nl_uk util) |
| `fr_uk_net_flow_la` | flow FR→UK − flow UK→FR (lagge) |
| `be_uk_net_flow_la` | flow BE→UK − flow UK→BE |
| `nl_uk_net_flow_la` | flow NL→UK − flow UK→NL |
| `dk1_uk_net_flow_la` | flow DK1→UK − flow UK→DK1 |
| `total_net_import_uk_la` | somme de tous les net flows vers UK |
| `fr_uk_avg_cost_la` | cout moyen FR→UK (lagge) |
| `uk_fr_avg_cost_la` | cout moyen UK→FR (lagge) |
| `fr_uk_cost_spread_la` | cout FR→UK − cout UK→FR |
| `fr_uk_congested` | utilization > 0.90 |
| `uk_fr_congested` | utilization > 0.90 |
| `any_direction_congested` | FR→UK OU UK→FR congestionne |
| `fr_uk_atc_change_24h` | ATC_total.diff(24) |

---

## Cat 10 — Prix / Lags (19 features)

| Feature | Formule |
|---------|---------|
| `fr_spot_lag_48h` | spot_la.shift(24) |
| `fr_spot_lag_168h` | spot_la.shift(144) |
| `uk_spot_lag_48h` | spot_la.shift(24) |
| `uk_spot_lag_168h` | spot_la.shift(144) |
| `de_spot_lag_48h` | spot_la.shift(24) |
| `be_spot_lag_48h` | spot_la.shift(24) |
| `nl_spot_lag_48h` | spot_la.shift(24) |
| `spread_fr_uk_la` | fr_spot − uk_spot (lagge) |
| `spread_fr_de_la` | fr_spot − de_spot |
| `spread_uk_nl_la` | uk_spot − nl_spot |
| `spread_uk_be_la` | uk_spot − be_spot |
| `spread_de_fr_abs_la` | |fr_spot − de_spot| |
| `continental_avg_spot_la` | mean(DE, BE, NL, FR spot_la) |
| `fr_vs_continental_la` | fr_spot − avg continental |
| `uk_vs_continental_la` | uk_spot − avg continental |
| `fr_spot_change_24h_la` | spot_la − spot_la.shift(24) |
| `uk_spot_change_24h_la` | spot_la − spot_la.shift(24) |
| `fr_spot_change_168h_la` | spot_la − spot_la.shift(144) |
| `uk_spot_change_168h_la` | spot_la − spot_la.shift(144) |

---

## Cat 11 — Interactions (13 features)

| Feature | Formule |
|---------|---------|
| `fr_thermal_need_x_gas` | thermal_need_pos × gas |
| `uk_thermal_need_x_gas` | thermal_need_pos × gas |
| `fr_baseload_gap_x_gas` | baseload_gap_pos × gas |
| `uk_wind_gap_x_gas` | (load − wind).clip(0) × gas |
| `fr_residual_x_spark` | residual_load × spark_spread |
| `uk_residual_x_spark` | residual_load × spark_spread |
| `fr_wind_x_hour` | wind × hour_sin |
| `uk_wind_x_hour` | wind × hour_sin |
| `fr_solar_x_load` | solar × load |
| `fr_nuclear_x_gas` | nuclear × gas |
| `uk_wind_x_gas` | wind × gas |
| `fr_congestion_x_spark_diff` | utilization × (spark_FR − spark_UK) |
| `fr_thermal_need_x_nuclear_change` | thermal_need × |nuclear_change_24h| |

---

## Cat 12 — Regional / Voisins (11 features)

| Feature | Formule |
|---------|---------|
| `continental_load_total` | FR + DE + BE + NL load |
| `continental_wind_total` | FR + DE + BE + NL wind |
| `continental_solar_total` | FR + DE + BE + NL solar |
| `continental_residual_load` | load_total − wind − solar |
| `nordic_wind_total` | DK1 + DK2 wind |
| `iberian_load` | ES load |
| `iberian_wind` | ES wind |
| `iberian_solar` | ES solar |
| `de_load_change_24h` | DE load.diff(24) |
| `be_nl_combined_load` | BE + NL load |
| `at_ch_combined_load` | AT + ITN load |

---

## Cat 13 — Temperature riviere (12 features)

| Feature | Formule |
|---------|---------|
| `fr_rhone_hot` | rhone_temp > 25°C |
| `fr_rhine_hot` | rhine_temp > 25°C |
| `de_danube_hot_donauworth` | danube_temp > 23°C |
| `de_danube_hot_ingolstadt` | danube_temp > 23°C |
| `fr_any_river_hot` | rhone OU rhine > seuil |
| `de_any_river_hot` | danube_don OU danube_ing > seuil |
| `fr_rhone_temp_excess` | max(rhone − 25, 0) |
| `fr_rhine_temp_excess` | max(rhine − 25, 0) |
| `fr_max_river_temp` | max(rhone, rhine) |
| `de_max_river_temp` | max(danube_don, danube_ing) |
| `fr_river_temp_change_24h` | rhone.diff(24) |
| `fr_hot_river_x_nuclear_low` | any_river_hot × nuclear_low |

---

## Cat 14 — Transforms non-lineaires (11 features)

| Feature | Formule |
|---------|---------|
| `fr_residual_load_squared` | residual_load² |
| `uk_residual_load_squared` | residual_load² |
| `fr_thermal_need_cubed_pos` | thermal_need_pos^1.5 |
| `uk_wind_pen_squared` | wind_pen² |
| `fr_spark_spread_log` | log1p(spark_spread.clip(0)) |
| `uk_spark_spread_log` | log1p(spark_spread.clip(0)) |
| `fr_spot_la_log` | sign(spot) × log1p(|spot|) |
| `uk_spot_la_log` | sign(spot) × log1p(|spot|) |
| `fr_gas_sqrt` | sqrt(gas) |
| `fr_wind_f_clipped` | wind.clip(upper=22000) |
| `uk_wind_f_clipped` | wind.clip(upper=20000) |

---

## Cat 15 — Rolling statistics (24 features)

| Feature | Formule |
|---------|---------|
| `fr_spot_la_roll_24h_mean` | spot_la.rolling(24).mean() |
| `fr_spot_la_roll_24h_std` | spot_la.rolling(24).std() |
| `uk_spot_la_roll_24h_mean` | spot_la.rolling(24).mean() |
| `uk_spot_la_roll_24h_std` | spot_la.rolling(24).std() |
| `fr_spot_la_roll_168h_mean` | spot_la.rolling(168).mean() |
| `fr_spot_la_roll_168h_std` | spot_la.rolling(168).std() |
| `uk_spot_la_roll_168h_mean` | spot_la.rolling(168).mean() |
| `uk_spot_la_roll_168h_std` | spot_la.rolling(168).std() |
| `fr_spot_la_roll_24h_min` | spot_la.rolling(24).min() |
| `fr_spot_la_roll_24h_max` | spot_la.rolling(24).max() |
| `uk_spot_la_roll_24h_min` | spot_la.rolling(24).min() |
| `uk_spot_la_roll_24h_max` | spot_la.rolling(24).max() |
| `fr_spot_la_roll_24h_range` | max − min (24h) |
| `uk_spot_la_roll_24h_range` | max − min (24h) |
| `uk_wind_roll_24h_mean` | wind.rolling(24).mean() |
| `fr_load_roll_168h_mean` | load.rolling(168).mean() |
| `uk_load_roll_168h_mean` | load.rolling(168).mean() |
| `fr_gas_roll_168h_mean` | gas.rolling(168).mean() |
| `uk_gas_roll_168h_mean` | gas.rolling(168).mean() |
| `eu_emission_roll_168h_mean` | emission.rolling(168).mean() |
| `fr_spot_la_deviation_24h` | spot − rolling_24h_mean |
| `uk_spot_la_deviation_24h` | spot − rolling_24h_mean |
| `fr_spot_la_deviation_168h` | spot − rolling_168h_mean |
| `uk_spot_la_deviation_168h` | spot − rolling_168h_mean |

---

## Cat 16 — Momentum / Trend (8 features)

| Feature | Formule |
|---------|---------|
| `fr_price_acceleration` | change_24h − change_24h.shift(24) (derivee seconde) |
| `uk_price_acceleration` | idem |
| `fr_gas_momentum_24h` | gas − gas.shift(24) |
| `uk_gas_momentum_24h` | gas − gas.shift(24) |
| `fr_nuclear_trend_3d` | nuclear.rolling(72).mean() − rolling(168).mean() |
| `fr_spot_la_ewm_24h` | spot_la.ewm(span=24).mean() |
| `uk_spot_la_ewm_24h` | spot_la.ewm(span=24).mean() |
| `spread_fr_uk_momentum` | spread − spread.shift(24) |

---

## Cat 17 — Offre/Demande avancee (9 features) [NOUVEAU]

| Feature | Formule | Priorite |
|---------|---------|----------|
| `fr_residual_load_v2` | load − wind − solar − hydro_ror | Haute — must-run soustrait |
| `fr_security_margin` | (nuclear + gas_cap) − residual_load | Haute — distance au blackout (MW) |
| `uk_security_margin` | (nuclear + gas_cap) − residual_load | Haute |
| `fr_scarcity_ratio` | residual / (nuclear + gas_cap) | **CRITIQUE** — convexite merit order |
| `uk_scarcity_ratio` | residual / (nuclear + gas_cap) | **CRITIQUE** |
| `fr_scarcity_critical` | scarcity_ratio > 0.85 | CRITIQUE — seuil stress |
| `uk_scarcity_critical` | scarcity_ratio > 0.85 | CRITIQUE |
| `fr_scarcity_extreme` | scarcity_ratio > 0.95 | CRITIQUE — seuil panique |
| `uk_scarcity_extreme` | scarcity_ratio > 0.95 | CRITIQUE |

---

## Cat 18 — Load & Residual ramps 1h/3h (8 features) [NOUVEAU]

| Feature | Formule | Priorite |
|---------|---------|----------|
| `fr_load_ramp_1h` | load.diff(1) | Haute — cout demarrage |
| `fr_load_ramp_3h` | load.diff(3) | Haute |
| `uk_load_ramp_1h` | load.diff(1) | Haute |
| `uk_load_ramp_3h` | load.diff(3) | Haute |
| `fr_residual_ramp_1h` | residual_load.diff(1) | Haute — pression thermique |
| `fr_residual_ramp_3h` | residual_load.diff(3) | Haute — duck curve |
| `uk_residual_ramp_1h` | residual_load.diff(1) | Haute |
| `uk_residual_ramp_3h` | residual_load.diff(3) | Haute |

---

## Cat 19 — Spark spreads multi-efficacite (4 features) [NOUVEAU]

| Feature | Formule | Priorite |
|---------|---------|----------|
| `fr_spark_ocgt` | gas/0.40 + emission×0.37 | Moyenne — vieille OCGT |
| `uk_spark_ocgt` | gas/0.40 + emission×0.37 | Moyenne |
| `fr_spark_ccgt` | gas/0.55 + emission×0.37 | Moyenne — CCGT moderne |
| `uk_spark_ccgt` | gas/0.55 + emission×0.37 | Moyenne |

---

## Cat 20 — Interconnexion avancee (5 features) [NOUVEAU]

| Feature | Formule | Priorite |
|---------|---------|----------|
| `fr_uk_flow_over_atc` | sum(flow_fr-uk_la) / ATC_total | Haute — pression reelle cable |
| `fr_uk_unused_capacity` | ATC_total − sum(flow_la).clip(0) | Haute — marge MW |
| `be_uk_unused_capacity` | ATC_be-uk − flow_be-uk_la | Moyenne |
| `nl_uk_unused_capacity` | ATC_nl-uk − flow_nl-uk_la | Moyenne |
| `total_unused_capacity_to_uk` | somme des 3 unused | Haute — isolement UK |

---

## Cat 21 — Z-Scores & anomalies (8 features) [NOUVEAU]

| Feature | Formule | Priorite |
|---------|---------|----------|
| `fr_residual_zscore_14d` | (residual − mean_336h) / std_336h | **Haute** — detecteur panique |
| `uk_residual_zscore_14d` | idem | **Haute** |
| `fr_load_zscore_14d` | (load − mean_336h) / std_336h | Moyenne |
| `uk_load_zscore_14d` | idem | Moyenne |
| `fr_wind_zscore_14d` | (wind − mean_336h) / std_336h | Moyenne |
| `uk_wind_zscore_14d` | idem | Moyenne |
| `fr_lag_reliability_ratio` | spot_la / spot_lag_168h | Moyenne — fiabilite du lag |
| `uk_lag_reliability_ratio` | spot_la / spot_lag_168h | Moyenne |

---

## Cat 22 — Signaux stochastiques / SDE (12 features) [NOUVEAU]

| Feature | Formule | Priorite |
|---------|---------|----------|
| `fr_jump_count_24h` | rolling_sum(\|ΔP\| > 50, 24h) | **Haute** — regime de sauts |
| `uk_jump_count_24h` | idem | **Haute** |
| `fr_jump_count_48h` | rolling_sum(\|ΔP\| > 50, 48h) | Moyenne |
| `uk_jump_count_48h` | idem | Moyenne |
| `fr_jump_magnitude_24h` | mean(\|ΔP\| quand > seuil) sur 24h | Moyenne |
| `uk_jump_magnitude_24h` | idem | Moyenne |
| `fr_vol_ratio` | std_24h / std_168h | **Haute** — stress vs calme |
| `uk_vol_ratio` | idem | **Haute** |
| `fr_mean_reversion_strength` | deviation_168h / std_168h | **Haute** — force de rappel O-U |
| `uk_mean_reversion_strength` | idem | **Haute** |

---

## Cat 23 — Transforms ameliorees (4 features) [NOUVEAU]

| Feature | Formule | Priorite |
|---------|---------|----------|
| `fr_asinh_spot_la` | arcsinh(spot_la) | Remplace sign×log1p |
| `uk_asinh_spot_la` | arcsinh(spot_la) | Remplace sign×log1p |
| `fr_asinh_spark` | arcsinh(spark_spread) | Remplace log1p(clip) |
| `uk_asinh_spark` | arcsinh(spark_spread) | Remplace log1p(clip) |

---

## Cat 24 — Nuclear shortfall (2 features) [NOUVEAU — recherche]

| Feature | Formule |
|---------|---------|
| `fr_nuclear_shortfall` | expanding_max(nuclear_avcap) − nuclear_avcap |
| `uk_nuclear_shortfall` | idem UK |

---

## Cat 25 — ATC/NTC ratios par cable (6 features) [NOUVEAU — recherche]

| Feature | Formule |
|---------|---------|
| `atc_fr-uk-{1,2,3}_f_ratio` | ATC / NTC par cable FR→UK |
| `atc_uk-fr-{1,2,3}_f_ratio` | ATC / NTC par cable UK→FR |

---

## Cat 26 — Market-specific (SDAC/N2EX) (18 features) [NOUVEAU — recherche]

| Feature | Formule | Source |
|---------|---------|--------|
| `gas_spread_uk_eu` | uk_gas − nl_gas | Empirique r=-0.74 FR |
| `continent_thermal_floor` | avg(DE, FR, NL spark_spread) | Empirique r=0.902 FR |
| `uk_import_ratio` | net_import_uk / uk_load | N2EX theory |
| `fr_export_ratio` | fr_uk_net_flow / fr_load | SDAC theory |
| `de_wind_high` | de_wind_pen > 0.30 | Empirique cliff -27 EUR |
| `de_wind_very_high` | de_wind_pen > 0.50 | Empirique |
| `uk_wind_share_flexible` | uk_wind / (load − nuclear − biomass) | N2EX merit order |
| `fr_de_decoupled` | \|fr_spot_la − de_spot_la\| > 10 | SDAC theory |
| `fr_merit_order_cost` | interpolation CCGT↔OCGT par scarcity | Merit order theory |
| `uk_merit_order_cost` | idem UK | Merit order theory |
| `fr/uk_spot_la_h3` | spot_la quand hour=3, ffill | Ziel & Weron 2018 |
| `fr/uk_spot_la_h19` | spot_la quand hour=19, ffill | Ziel & Weron 2018 |
| `fr/uk_intraday_amplitude` | h19 − h3 | EPF literature |
| `dark_doldrums_fr` | winter × evening × (1−wind) × (1−solar) | Physical mechanism |
| `dark_doldrums_uk` | winter × evening × (1−wind) | Physical mechanism |

---

## Cat 27 — Advanced price formation (8 features) [NOUVEAU — recherche]

| Feature | Formule | Partial r |
|---------|---------|-----------|
| `fr_nuke_shortfall_x_gas` | shortfall × nl_gas | 0.768 vs shortfall |
| `uk_nuke_shortfall_x_gas` | shortfall × uk_gas | 0.763 vs shortfall |
| `fr_implied_re_surplus` | spark − spot_la | — |
| `uk_implied_re_surplus` | spark − spot_la | 0.547 vs spot_la |
| `uk_cheapest_import` | min(FR+cost, BE+cost, NL+cost, DK1+cost) | 0.433 vs fr_spot_la |
| `uk_import_price_range` | max − min import prices | 0.293 raw |
| `net_capacity_cost_fr_uk` | cost_fr→uk − cost_uk→fr | -0.278 vs spark |
| `uk_fossil_or_import_need` | load − (nuke + bio + wind + solar) | 0.393 raw |

---

## Cat 28 — FR continent territory (15 features) [NOUVEAU — recherche]

| Feature | Formule | Partial r vs spark |
|---------|---------|-------------------|
| `continent_nuclear_total` | FR + DE + BE nuclear | -0.141 |
| `continent_thermal_need` | continental_residual − nuclear | **0.534** |
| `continent_zero_mc_pen` | (wind+solar+nuke+hydro) / load | **-0.550** |
| `continent_re_pen` | (wind+solar) / load continental | **-0.483** |
| `continent_weighted_price` | load-weighted avg (6 marches) | **0.609** |
| `carbon_to_gas_ratio` | eu_emission / nl_gas | -0.226 vs spot_la |
| `spread_fr_es` | fr_spot_la − es_spot_la | 0.311 |
| `euro_scarcity_ratio` | euro_deficit / euro_gas_cap | **0.618** |
| `euro_adequacy_deficit` | euro_load − euro_zero_mc (MW) | **0.553** |
| `wind_tier1_pen` | (DE+BE wind) / continental_load | **-0.453** |
| `continent_wind_nuke_ratio` | continental_wind / continental_nuke | **-0.437** |
| `es_thermal_floor` | es_gas/eff + emission×em | 0.499 vs spot_la |
| `es_residual_load` | es_load − es_wind − es_solar − es_hydro | 0.324 |
| `de_river_high` | de_river_temp_avg > 20°C | **0.334** |
| `wind_nuke_deviation_gap` | (wind/norm) − (nuke/norm) | **-0.307** |

---

## Cat 29 — UK island territory (6 features) [NOUVEAU — recherche]

| Feature | Formule | Partial r vs spark |
|---------|---------|-------------------|
| `uk_gas_utilization` | gas_gen_proxy / gas_cap | **0.513** |
| `uk_gas_headroom` | gas_cap − gas_gen_proxy | **-0.507** |
| `uk_capacity_margin` | total_domestic − load | **-0.517** |
| `uk_gas_cost_per_mw` | gas × gas_gen / load | 0.663 vs spot_la |
| `uk_self_sufficiency` | total_domestic / load | **-0.509** |
| `uk_load_pct_weekly_peak` | load / rolling_168h_max | 0.427 |

---

## Cat 30 — Regime & structural breaks (14 features) [NOUVEAU — recherche]

| Feature | Formule | Notes |
|---------|---------|-------|
| `iberian_exception` | 1 si 15/06/2022 — 31/12/2023 | Cap gaz ES/PT |
| `fr_thermal_gap` | load − (nuke + wind + solar + hydro) | MW a couvrir par gaz/imports |
| `fr_gas_on_margin` | thermal_gap > 0 | 34.6% train, 8.8% test |
| `fr_gas_price_if_marginal` | nl_gas × gas_on_margin | Prix gaz quand il est marginal |
| `uk_thermal_gap` | load − zero_mc_gen | MW gaz necessaires UK |
| `uk_gas_on_margin` | thermal_gap > 0 | 93.4% train |
| `uk_gas_price_if_marginal` | uk_gas × gas_on_margin | Prix gaz quand marginal |
| `fr_oversupply_mw` | max(zero_mc − load, 0) | MW en surplus (prix negatifs) |
| `fr_negative_price_risk` | oversupply > 0 | 63.7% des heures |
| `uk_oversupply_mw` | max(zero_mc − load, 0) | UK surplus |
| `uk_negative_price_risk` | oversupply > 0 | 6.6% des heures |
| `fr_nuclear_avail_ratio` | nuclear / expanding_max | Arc crise→recovery |
| `uk_nuclear_avail_ratio` | nuclear / expanding_max | |
| `fr_gas_spot_rolling_corr` | rolling_168h_corr(gas, spot_la) | Regime gaz-driven vs pas |

---

## Cat 32 — Advanced Price Proxies (22 features) [NOUVEAU — physique marche]

Fonction : `_add_advanced_price_proxies(df, cfg)` dans `feature_engineering.py`

| Feature | Formule | Raison |
|---------|---------|--------|
| `fr_dynamic_marginal` | `w*merit_order + (1-w)*nuclear_mc`, w=f(scarcity) | Prix marginal adaptatif (gaz vs nucleaire) |
| `uk_dynamic_marginal` | idem UK | |
| `fr_import_price` | `min(de, be, ch, es spot_la) + transport_cost` | Prix import le moins cher |
| `fr_opportunity_cost` | `min(dynamic_marginal, import_price)` | **CRITIQUE** — vrai cout d'opportunite |
| `uk_import_floor` | `min(fr, be, nl, dk1 spot_la) + transport_cost` | Prix import UK |
| `uk_opportunity_cost` | `min(dynamic_marginal, import_floor)` | |
| `fr_scarcity_barrier` | `spark * (1/(1-scarcity)^p)` | Pont exponentiel vers stress |
| `uk_scarcity_barrier` | idem UK | |
| `fr_load_price_signal_7d` | `price_per_mw_7d * residual_load` | Signal load-prix recent |
| `fr_load_price_signal_load` | `price_per_mw_7d * load_f` | |
| `uk_load_price_signal_7d` | idem UK | |
| `fr_hydro_opp_cost` | `rolling_max(spark, 168h)` | Valeur de l'eau |
| `uk_hydro_opp_cost` | idem UK | |
| `fr_basis_v2` | `spot_la - opportunity_cost` | Base v2 (vs meilleur proxy) |
| `uk_basis_v2` | `spot_la - opportunity_cost` | |
| `fr_basis_v2_lag_48h` | `basis_v2.shift(24)` | Persistence base |
| `uk_basis_v2_lag_48h` | idem UK | |
| `fr_basis_v2_roll_24h_mean` | `basis_v2.rolling(24).mean()` | Niveau moyen base |
| `uk_basis_v2_roll_24h_mean` | idem UK | |
| `fr_price_per_mw_7d` | `roll_168h(spot_la) / roll_168h(residual_load)` | Ratio prix/charge recent |
| `uk_price_per_mw_7d` | idem UK | |

---

## Resume

| Bloc | Count |
|------|-------|
| Cat 1 — Pre-processing | 0 (7 modifiees) |
| Cat 2 — Calendrier | 21 |
| Cat 3 — Jours feries | 10 |
| Cat 4 — Fondamentaux | 23 |
| Cat 5 — Penetration ENR | 12 |
| Cat 6 — Dynamique ENR | 15 |
| Cat 7 — Nucleaire | 11 |
| Cat 8 — Hydro | 9 |
| Cat 9 — Interconnecteurs | 23 |
| Cat 10 — Prix / Lags | 19 |
| Cat 11 — Interactions | 13 |
| Cat 12 — Regional | 11 |
| Cat 13 — Temperature riviere | 12 |
| Cat 14 — Transforms | 11 |
| Cat 15 — Rolling stats | 24 |
| Cat 16 — Momentum | 8 |
| Cat 17 — Offre/Demande avancee | 9 |
| Cat 18 — Ramps 1h/3h | 8 |
| Cat 19 — Spark multi-eff | 4 |
| Cat 20 — Interconnexion avancee | 5 |
| Cat 21 — Z-Scores | 8 |
| Cat 22 — Signaux SDE | 12 |
| Cat 23 — Transforms asinh | 4 |
| Cat 24 — Nuclear shortfall | 2 |
| Cat 25 — ATC/NTC ratios cable | 6 |
| Cat 26 — Market-specific SDAC/N2EX | 18 |
| Cat 27 — Price formation signals | 8 |
| Cat 28 — FR continent territory | 15 |
| Cat 29 — UK island territory | 6 |
| Cat 30 — Regime & structural breaks | 14 |
| **Cat 32 — Advanced Price Proxies** | **22** |
| **TOTAL engineered** | **~360** |
| Colonnes brutes | 112 |
| **GRAND TOTAL** | **~471** |

---

## Dedup par correlation (seuil 0.98)

SHAP v3 : 75 features eliminees pour FR, 78 pour UK (doublons exacts comme `de_gas` ≈ `nl_gas` r=1.000,
`fr_spark_spread` ≈ `fr_spark_ccgt` r=1.000, etc.).

**SHAP v4 (stationary target)** : re-ranking complet avec target = `spot - roll_168h_mean` pour FR.
- 467 → 379 features FR (88 dropped), 467 → 378 features UK (89 dropped)
- Rankings dans `outputs/shap_ranking_v4_stationary.json`

### SHAP v4 — Top-10 FR (target stationnaire)

| Rank | Feature | SHAP |
|------|---------|------|
| 1 | `fr_spot_la_roll_168h_mean` | 7.32 |
| 2 | `fr_residual_zscore_14d` | 4.97 |
| 3 | `uk_residual_zscore_14d` | 3.48 |
| 4 | `fr_spot_la_deviation_168h` | 3.13 |
| 5 | `continental_residual_load` | 2.66 |
| 6 | `euro_scarcity_ratio` | 2.56 |
| 7 | `wind_nuke_deviation_gap` | 2.43 |
| 8 | `fr_residual_change_24h` | 2.33 |
| 9 | `uk_spot_la_deviation_168h` | 2.12 |
| 10 | `uk_price_per_mw_7d` | 2.05 |

**Changement majeur** : Les features de NIVEAU (spark spread, prix voisins) deviennent inutiles. Les features de DEVIATION (z-scores, changes, scarcity ratios) dominent.

## Feature Selection v5 — Noise Probing (Boruta)

Pipeline : 20 features bruit aleatoire × 5 rounds. Une feature est "confirmee" si elle bat le bruit ≥3/5 fois.

| Categorie | Count |
|-----------|-------|
| **Confirmees** (beat noise ≥3/5) | 52 |
| Tentatives (1-2/5) | 45 |
| **Rejetees** (jamais beat noise) | 282 |

**74% des 379 features sont du bruit pur.** CatBoost les ignore naturellement (best_iter~56), mais ca confirme que la majorite des features n'apportent rien.

Les 27 features finales (apres RFE) sont dans `outputs/feature_selection_v5_fr.json`.

---

## Target Transformations

| Cible | Meilleur target | Reconstruction | RMSE |
|-------|----------------|----------------|------|
| **FR** | `spot - fr_spot_la_roll_168h_mean` | `roll_168h_mean + pred` | **17.84** |
| **FR + HBC** | idem + hourly bias correction | idem + correction par heure | **17.44** |
| **UK** | `spot - uk_merit_order_cost` (basis v1) | `merit_order_cost + pred` | **9.87** |

**FR — Stationarity** : la serie brute (spot) a un CV trimestriel de 82% (non-stationnaire). La deviation vs roll_168h a un CV de ~5% (stationnaire en moyenne). Le poids `exp(-2.0*days/365) / rolling_std²` stabilise aussi la variance.

**UK** : Le basis v1 (spot - merit_order_cost) fonctionne car le gaz est marginal ~93% du temps en UK.

---

## Sample Weights

| Config | Formule | Usage |
|--------|---------|-------|
| FR best | `exp(-2.0 * days_ago / 365) / clip(roll_168h_std², 1)` | Downweight crise + haute volatilite |
| UK best | Uniform (pas de weights) | UK est deja stationnaire |

---

## Parametres config.yaml

```yaml
feature_engineering:
  daily_cols: [de_gas, es_gas, fr_gas, nl_gas, uk_gas, eu_emission, uk_emission]
  gas_efficiency: 0.50
  emission_factor: 0.37
  river_hot_threshold_fr: 25.0
  river_hot_threshold_de: 23.0
  nuclear_low_threshold: 35000
  nuclear_very_low_threshold: 25000
  uk_wind_high_threshold: 0.50
  uk_wind_very_high_threshold: 0.65
  scarcity_critical_threshold: 0.85
  scarcity_extreme_threshold: 0.95
  jump_threshold: 50.0
  ocgt_efficiency: 0.40
  ccgt_efficiency: 0.55
  nuclear_marginal_cost: 12.0
  transport_cost_approx: 2.0
  scarcity_barrier_power: 1.5
```

---

## Hyperparametres optimaux (CatBoost FR)

| Param | Valeur | Notes |
|-------|--------|-------|
| iterations | 15000 | avec early stopping 200 |
| learning_rate | 0.03 | lr=0.008 similaire mais plus lent |
| **depth** | **3** | <<< cle du gain (avant: 8) |
| **l2_leaf_reg** | **30** | forte regularisation (avant: 5) |
| subsample | 0.7 | row sampling |
| **colsample_bylevel** | **0.5** | column sampling 50% (nouveau) |
| random_seed | 42 | |

**Pourquoi depth=3 ?** Avec le target stationnaire (deviation centree ~0), la structure a apprendre est simple : corrections lineaires/additives par rapport a l'ancre. Des arbres profonds (depth=8) capturaient du bruit. Des arbres peu profonds capturent les vraies interactions (2-3 splits max).

## Historique des scores FR

| Etape | Config | FR RMSE | Delta |
|-------|--------|---------|-------|
| v1 | Raw spot, 20 feat | 27.52 | — |
| v2 | arcsinh(spot), 20 feat | 26.10 | -1.42 |
| v3 | arcsinh + Cat32, 31 feat | 24.79 | -2.73 |
| v4 | arcsinh + Cat32 + weights(2.0) | 24.45 | -3.07 |
| v5 | dev_168h + Cat32 + weights/std² | 19.99 | -7.53 |
| v6 | + SHAP v4 ranking (all feat, d=8) | 18.55 | -8.97 |
| **v7** | **+ depth=3, l2=30, colsample=0.5** | **17.84** | **-9.68** |
| **v7+HBC** | **+ hourly bias correction** | **17.44** | **-10.08** |

## Historique des scores UK

| Etape | Config | UK RMSE | Delta |
|-------|--------|---------|-------|
| v1 | Raw spot, 75 feat | 10.45 | — |
| v2 | Basis v1 (merit_order), 75 feat | 9.87 | -0.58 |
