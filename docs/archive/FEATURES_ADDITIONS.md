# Features a ajouter — Discussion utilisateur (parties 1-2-3)

## Partie 1 : Transformations + Offre/Demande + Couts + Spatial + Signaux

| # | Feature | Formule | Categorie | Priorite |
|---|---------|---------|-----------|----------|
| 1 | `asinh_spot_la` (FR/UK) | `asinh(spot_la)` | Transform | Remplace sign*log1p |
| 2 | `residual_load_v2` (FR) | `load - wind - solar - hydro_ror` | Fondamental | Haute — must-run soustrait |
| 3 | `security_margin` (FR/UK) | `(nuclear + gas_cap) - residual_load` | Fondamental | Haute — distance au blackout |
| 4 | `load_ramp_1h` (FR/UK) | `load[t] - load[t-1]` | Dynamique | Haute — cout demarrage |
| 5 | `load_ramp_3h` (FR/UK) | `load[t] - load[t-3]` | Dynamique | Haute |
| 6 | `spark_spread_ocgt` | `gas/0.40 + emission*0.37` | Cout marginal | Moyenne — vieille centrale |
| 7 | `spark_spread_ccgt` | `gas/0.55 + emission*0.37` | Cout marginal | Moyenne — moderne |
| 8 | `flow_over_atc` | `flow_la / atc_f.clip(1)` | Interconnexion | Haute — pression cable |

## Partie 2 : Scarcity + Unused Cap + Lag 168h + Accel + Z-Score

| # | Feature | Formule | Categorie | Priorite |
|---|---------|---------|-----------|----------|
| 9 | `scarcity_ratio` (FR/UK) | `residual / (nuclear + gas_cap)` | **CRITIQUE** | Convexite des prix |
| 10 | `scarcity_critical` | `ratio > 0.85` | CRITIQUE | Seuil |
| 11 | `scarcity_extreme` | `ratio > 0.95` | CRITIQUE | Seuil |
| 12 | `unused_capacity` (par cable) | `ATC_f - Flow_la` | Haute | Isolement UK |
| 13 | `lag_reliability_ratio` | `spot_la / spot_lag_168h` | Moyenne | Lag 24h vs 168h |
| 14 | `residual_ramp_1h` (FR/UK) | `residual.diff(1)` | Haute | Pression instantanee |
| 15 | `residual_ramp_3h` (FR/UK) | `residual.diff(3)` | Haute | Duck curve |
| 16 | `residual_zscore_14d` (FR/UK) | `(res - mean_336h) / std_336h` | **Haute** | Detecteur panique |
| 17 | `load_zscore_14d` (FR/UK) | idem sur load | Moyenne | |
| 18 | `wind_zscore_14d` (FR/UK) | idem sur wind | Moyenne | |

## Partie 3 : Cheat Codes Stochastiques (SDE)

| # | Feature | Formule | Categorie | Priorite |
|---|---------|---------|-----------|----------|
| 19 | `jump_count_24h` (FR/UK) | `count(|ΔP| > seuil) sur 24h` | **SDE** | Haute — regime sauts |
| 20 | `jump_count_48h` (FR/UK) | idem 48h | SDE | Moyenne |
| 21 | `jump_magnitude_24h` | `mean(|ΔP|) quand > seuil` | SDE | Moyenne |
| 22 | `vol_ratio` (FR/UK) | `std_24h / std_168h` | SDE | Haute — stress vs calme |
| 23 | `mean_reversion_strength` (FR/UK) | `deviation_168h / std_168h` | SDE | Haute — force rappel normalisee |

## Total : ~23 features nouvelles (x2 pour FR/UK = ~40-45 colonnes)
## Seuil de jump a mettre en config (defaut: 50 EUR)
