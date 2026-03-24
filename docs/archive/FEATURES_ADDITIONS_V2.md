# Features Research-Backed — Cat 24 a 30

> Features ajoutees apres analyse empirique (correlations, correlations partielles),
> recherche internet sur les mecanismes de marche (SDAC/EUPHEMIA, N2EX, crise nucleaire),
> et validation SHAP v2. Les features eliminées par le dedup correlation (seuil 0.98) sont
> marquees ~~barrees~~.

---

## Contexte de recherche

### Mecanisme de prix : SDAC / EUPHEMIA

Le prix spot day-ahead en France est fixe par **marginal pricing** via l'algorithme EUPHEMIA
(Single Day-Ahead Coupling). Meme si 70% de la production francaise est nucleaire, le prix
est dicte par la **derniere unite necessaire** pour equilibrer l'offre et la demande :

- Quand le gaz est marginal : `prix ≈ gas / efficiency + emission × factor`
- Quand le renouvelable est en surplus : prix → 0 ou negatif
- FR est un **price-taker continental** — le prix converge avec DE, BE, NL, ES via FBMC

### UK : marche insulaire independant (N2EX / EPEX GB)

Le Royaume-Uni opere son propre marche via N2EX. Contrairement a FR :
- UK est **gas-driven** (~93% du temps, le gaz est marginal)
- L'isolement depend de la capacite d'interconnexion (4 cables vers FR, BE, NL, DK)
- Quand les cables sont satures, UK est un marche autarcique

### Structural breaks dans la periode d'entrainement

| Evenement | Dates | Impact |
|-----------|-------|--------|
| FBMC Core go-live | 8 juin 2022 | Market coupling Flow-Based |
| Crise nucleaire FR (SCC) | Ete 2022 | 32/56 reacteurs offline, record 743 EUR/MWh |
| Iberian exception | 15 juin 2022 — 31 dec 2023 | Cap sur le prix du gaz en ES/PT |
| EU revenue cap | Juil 2022 — Juin 2023 | Plafond revenus producteurs |
| Regime shift gas→RE | 2022 → 2024 | Gas marginal 34.6% → 8.8% (FR) |

**Consequence critique** : le dataset d'entrainement (Jul 22 — Jun 24) melange deux regimes
fondamentalement differents. Le test (Jul 24 — Fev 25) est uniquement en regime "normal" 2024.

---

## Cat 24 — Nuclear shortfall (2 features)

**Rationale** : Capture l'ecart entre la capacite nucleaire maximale observee et la capacite
actuelle. Mesure directe de l'indisponibilite nucleaire (maintenance, SCC, etc.).

| Feature | Formule | Survived dedup |
|---------|---------|----------------|
| `fr_nuclear_shortfall` | `expanding_max(nuclear_avcap) - nuclear_avcap` | FR #74, UK #483 |
| `uk_nuclear_shortfall` | idem UK | FR #147, UK #594 |

**Validation** : SHAP moyen = 1.02 FR, 0.30 UK. Surtout utile en interaction avec gas (→ Cat 27).

---

## Cat 25 — ATC/NTC ratios par cable (6 features)

**Rationale** : Le ratio ATC/NTC mesure la capacite commerciale disponible rapportee a la
capacite physique. Un ratio bas signale des contraintes reseau (maintenance, congestion
structurelle). Important pour le modele GAT (features d'aretes).

| Feature | Formule | Survived dedup |
|---------|---------|----------------|
| `atc_fr-uk-1_f_ratio` | ATC / NTC cable FR→UK #1 | FR #180, UK #458 |
| `atc_fr-uk-2_f_ratio` | ATC / NTC cable FR→UK #2 | FR #96, UK #586 |
| `atc_fr-uk-3_f_ratio` | ATC / NTC cable FR→UK #3 | FR #251, UK #619 |
| `atc_uk-fr-1_f_ratio` | ATC / NTC cable UK→FR #1 | FR #80, UK #460 |
| `atc_uk-fr-2_f_ratio` | ATC / NTC cable UK→FR #2 | FR #219, UK #482 |
| `atc_uk-fr-3_f_ratio` | ATC / NTC cable UK→FR #3 | FR #253, UK #507 |

**Validation** : SHAP moyen entre 0.15 et 0.75. Contribution surtout en interaction avec
d'autres features d'interconnexion.

---

## Cat 26 — Market-specific SDAC/N2EX (17 features, 1 deduped)

**Rationale** : Features specifiques aux mecanismes de prix SDAC (FR continental) et
N2EX (UK insulaire), basees sur la litterature EPF et l'analyse empirique.

| Feature | Formule | Source | Survived dedup |
|---------|---------|--------|----------------|
| `gas_spread_uk_eu` | uk_gas - nl_gas | Empirique r=-0.74 FR | FR #256, UK #390 |
| ~~`continent_thermal_floor`~~ | ~~avg(DE, FR, NL spark)~~ | ~~r=0.997 avec de_spark_spread~~ | **DEDUPED** |
| `uk_import_ratio` | net_import_uk / uk_load | N2EX theory | FR #71, UK #391 |
| `fr_export_ratio` | fr_uk_net_flow / fr_load | SDAC theory | FR #300, UK #622 |
| `de_wind_high` | de_wind_pen > 0.30 | Empirique cliff -27 EUR | FR #339, UK #694 |
| `de_wind_very_high` | de_wind_pen > 0.50 | Empirique | FR #339*, UK #448 |
| `uk_wind_share_flexible` | wind / (load - nuke - bio) | N2EX merit order | SHAP UK #6 (2.35) |
| `fr_de_decoupled` | \|fr_spot - de_spot\| > 10 | SDAC theory | FR #342, UK #674 |
| `fr_merit_order_cost` | interpolation CCGT↔OCGT | Merit order theory | FR #12, UK #364 |
| `uk_merit_order_cost` | idem UK | Merit order theory | **SHAP UK #1 (17.54!)** |
| `fr_spot_la_h3` | spot_la @ hour=3, ffill | Ziel & Weron 2018 | FR #146, UK #561 |
| `uk_spot_la_h3` | idem UK | Ziel & Weron 2018 | FR #239, UK #434 |
| `fr_spot_la_h19` | spot_la @ hour=19, ffill | Ziel & Weron 2018 | FR #126, UK #436 |
| `uk_spot_la_h19` | idem UK | Ziel & Weron 2018 | FR #122, UK #436 |
| `fr_intraday_amplitude` | h19 - h3 | EPF literature | FR #154, UK #529 |
| `uk_intraday_amplitude` | idem UK | EPF literature | FR #154*, UK #528 |
| `dark_doldrums_fr` | winter × evening × (1-wind) × (1-solar) | Physical | FR #317, UK #674* |
| `dark_doldrums_uk` | winter × evening × (1-wind) | Physical | UK #410 |

**Star feature** : `uk_merit_order_cost` = SHAP #1 pour UK (17.54), loin devant tout le reste.
Interpolation dynamique entre cout CCGT et OCGT en fonction du scarcity_ratio.

---

## Cat 27 — Advanced price formation signals (7 features, 1 deduped)

**Rationale** : Interactions non-lineaires et signaux de formation de prix identifies par
analyse de correlation partielle. Mesurent l'incremental value au-dela des features existantes.

| Feature | Formule | Partial r | Survived dedup |
|---------|---------|-----------|----------------|
| `fr_nuke_shortfall_x_gas` | shortfall × nl_gas | **0.768** vs shortfall seul | FR #117 |
| `uk_nuke_shortfall_x_gas` | shortfall × uk_gas | **0.763** vs shortfall seul | FR #130, UK #579 |
| `fr_implied_re_surplus` | spark - spot_la | 0.547 vs spot_la | FR #89 |
| ~~`uk_implied_re_surplus`~~ | ~~spark - spot_la~~ | ~~r=-1.0 avec uk_spot_minus_spark~~ | **DEDUPED** |
| `uk_cheapest_import` | min(FR+cost, BE+cost, NL+cost, DK1+cost) | 0.433 vs fr_spot_la | FR #264, UK #365 |
| `uk_import_price_range` | max - min import prices | 0.293 raw | FR #193, UK #485 |
| `net_capacity_cost_fr_uk` | cost_fr→uk - cost_uk→fr | -0.278 vs spark | FR #314, UK #... |
| ~~`uk_fossil_or_import_need`~~ | ~~load - (nuke+bio+wind+solar)~~ | ~~r>0.98 avec uk_thermal_need~~ | **DEDUPED** |

**Pourquoi shortfall × gas est si puissant** (partial r = 0.768) :
- `shortfall` seul dit "combien de nucleaire manque"
- `shortfall × gas` dit "combien coute le remplacement du nucleaire manquant par du gaz"
- C'est exactement ce que le merit order fait en pratique

---

## Cat 28 — FR continent territory (12 features, 3 deduped)

**Rationale** : La France fait partie du marche continental couple (SDAC). Son prix est
influence par les fondamentaux de tout le continent via EUPHEMIA. Ces features capturent
la position energetique continentale.

| Feature | Formule | Partial r vs spark | Survived dedup |
|---------|---------|-------------------|----------------|
| ~~`continent_nuclear_total`~~ | ~~FR + DE + BE nuclear~~ | ~~r>0.98 avec fr_nuclear_avcap~~ | **DEDUPED** |
| ~~`continent_thermal_need`~~ | ~~continental_residual - nuclear~~ | ~~0.534 (pre-dedup)~~ | **DEDUPED** |
| `continent_zero_mc_pen` | (wind+solar+nuke+hydro) / load continental | **-0.550** | FR #13, UK #450 |
| `continent_re_pen` | (wind+solar) / load continental | **-0.483** | FR #210, UK #500 |
| `continent_weighted_price` | load-weighted avg (6 marches) | **0.609** | FR #35, UK #432 |
| `carbon_to_gas_ratio` | eu_emission / nl_gas | -0.226 vs spot_la | FR #81, UK #466 |
| `spread_fr_es` | fr_spot_la - es_spot_la | 0.311 | FR #121, UK #585 |
| `euro_scarcity_ratio` | euro_deficit / euro_gas_cap | **0.618** | FR #7, UK #5 |
| `euro_adequacy_deficit` | euro_load - euro_zero_mc (MW) | **0.553** | FR #3, UK #368 |
| `wind_tier1_pen` | (DE+BE wind) / continental_load | **-0.453** | UK #413 |
| `continent_wind_nuke_ratio` | continental_wind / continental_nuke | **-0.437** | FR #208, UK #468 |
| `es_thermal_floor` | es_gas/eff + emission×em | 0.499 vs spot_la | FR #10, UK #... |
| `es_residual_load` | es_load - es_wind - es_solar - es_hydro | 0.324 | FR #31, UK #465 |
| `de_river_high` | de_river_temp_avg > 20degC | **0.334** | FR #311, UK #678 |
| `wind_nuke_deviation_gap` | (wind/norm) - (nuke/norm) | **-0.307** | FR #211, UK #555 |

**Star features** :
- `euro_scarcity_ratio` : SHAP #7 FR (5.62), #5 UK (2.36) — meilleure feature continentale
- `euro_adequacy_deficit` : SHAP #3 FR (10.31) — mesure directe du deficit en MW
- `continent_weighted_price` : SHAP #35 FR — prix moyen pondere par la demande

---

## Cat 29 — UK island territory (5 features, 1 deduped)

**Rationale** : Le UK est un marche insulaire ou le gaz est quasi-toujours marginal (~93%).
Ces features capturent l'etat specifique du systeme electrique britannique : utilisation
des centrales a gaz, marge de capacite, et autosuffisance.

| Feature | Formule | Partial r vs spark | Survived dedup |
|---------|---------|-------------------|----------------|
| `uk_gas_utilization` | gas_gen_proxy / gas_cap | **0.513** | FR #237, UK #388 |
| `uk_gas_headroom` | gas_cap - gas_gen_proxy | **-0.507** | FR #234, UK #392 |
| ~~`uk_capacity_margin`~~ | ~~total_domestic - load~~ | ~~r>0.98 avec uk_security_margin~~ | **DEDUPED** |
| `uk_gas_cost_per_mw` | gas × gas_gen / load | **0.663** vs spot_la | SHAP UK #9 (1.58) |
| `uk_self_sufficiency` | total_domestic / load | **-0.509** | FR #235, UK #487 |
| `uk_load_pct_weekly_peak` | load / rolling_168h_max | 0.427 | FR #171, UK #396 |

**Star feature** : `uk_gas_cost_per_mw` = partial r = 0.663 vs spot_la (apres controle
pour spark_spread). Capture le cout marginal du gaz rapporte a la demande totale — plus
fin que le spark spread seul.

---

## Cat 30 — Regime & structural breaks (14 features)

**Rationale** : Le dataset d'entrainement couvre deux regimes fondamentalement differents.
Ces features permettent au modele de distinguer les regimes et d'adapter ses predictions.
Basees sur la recherche internet (SDAC, crise nucleaire, Iberian exception).

> Note : Cat 30 a ete ajoutee APRES le SHAP v2 + dedup. Pas encore de ranking SHAP.

| Feature | Formule | Stats train | Stats test |
|---------|---------|-------------|------------|
| `iberian_exception` | 1 si 15/06/2022 — 31/12/2023 | **75.1%** des heures | **0%** |
| `fr_thermal_gap` | load - (nuke + wind + solar + hydro) | MW a couvrir par gaz | — |
| `fr_gas_on_margin` | thermal_gap > 0 | **34.6%** | **8.8%** |
| `fr_gas_price_if_marginal` | nl_gas × gas_on_margin | Prix gaz conditionnel | — |
| `uk_thermal_gap` | load - zero_mc_gen UK | MW gaz necessaires | — |
| `uk_gas_on_margin` | thermal_gap > 0 | **93.4%** | — |
| `uk_gas_price_if_marginal` | uk_gas × gas_on_margin | Prix gaz conditionnel | — |
| `fr_oversupply_mw` | max(zero_mc_gen - load, 0) | MW en surplus | — |
| `fr_negative_price_risk` | oversupply > 0 | **63.7%** | — |
| `uk_oversupply_mw` | max(zero_mc_gen - load, 0) | MW surplus UK | — |
| `uk_negative_price_risk` | oversupply > 0 | 6.6% | — |
| `fr_nuclear_avail_ratio` | nuclear / expanding_max | Arc crise → recovery | — |
| `uk_nuclear_avail_ratio` | nuclear / expanding_max | | — |
| `fr_gas_spot_rolling_corr` | rolling_168h_corr(gas, spot) | Regime gaz-driven vs pas | — |

**Insight cle** : `fr_gas_on_margin` passe de 34.6% a 8.8% entre train et test. C'est LE
structural break le plus important. `fr_gas_price_if_marginal` permet au modele d'appliquer
le prix du gaz uniquement quand il est effectivement marginal.

---

## Resume des features deduped (seuil r > 0.98)

| Feature supprimee | Raison | Doublon de |
|-------------------|--------|------------|
| `continent_thermal_floor` | r = 0.997 | `de_spark_spread` |
| `uk_implied_re_surplus` | r = -1.0 | `uk_spot_minus_spark` (signe inverse) |
| `uk_fossil_or_import_need` | r > 0.98 | `uk_thermal_need` |
| `continent_nuclear_total` | r > 0.98 | `fr_nuclear_avcap_f` (FR = 80% nuke EU) |
| `continent_thermal_need` | r > 0.98 | `continental_residual_load` - offset |
| `uk_capacity_margin` | r > 0.98 | `uk_security_margin` |

---

## SHAP v2 — Top-10 des nouvelles features (Cat 24-30)

### Pour FR (fr_spot)

| Rang global | Feature | Cat | Mean \|SHAP\| |
|-------------|---------|-----|--------------|
| #3 | `euro_adequacy_deficit` | 28 | 10.31 |
| #7 | `euro_scarcity_ratio` | 28 | 5.62 |
| #10 | `es_thermal_floor` | 28 | 2.60 |
| #12 | `uk_merit_order_cost` | 26 | 2.09 |
| #13 | `continent_zero_mc_pen` | 28 | 2.05 |
| #35 | `continent_weighted_price` | 28 | 0.75 |
| #74 | `fr_nuclear_shortfall` | 24 | 1.02 |
| #81 | `carbon_to_gas_ratio` | 28 | 0.88 |
| #89 | `fr_implied_re_surplus` | 27 | 0.78 |
| #117 | `fr_nuke_shortfall_x_gas` | 27 | 0.53 |

### Pour UK (uk_spot)

| Rang global | Feature | Cat | Mean \|SHAP\| |
|-------------|---------|-----|--------------|
| **#1** | **`uk_merit_order_cost`** | **26** | **17.54** |
| #5 | `euro_scarcity_ratio` | 28 | 2.36 |
| #6 | `uk_wind_share_flexible` | 26 | 2.35 |
| #9 | `uk_gas_cost_per_mw` | 29 | 1.58 |
| #14 | `euro_adequacy_deficit` | 28 | 1.01 |
| #15 | `uk_supply_demand_ratio` | — | 0.96 |
| #17 | `continental_wind_pen` | — | 0.87 |
| #365 | `uk_cheapest_import` | 27 | — |
| #388 | `uk_gas_utilization` | 29 | — |
| #392 | `uk_gas_headroom` | 29 | — |

---

## Bilan

- **69 features ajoutees** (Cat 24-30), dont **6 eliminees** par correlation dedup
- **63 features actives** dans le pipeline (450 colonnes totales)
- **uk_merit_order_cost** est la feature #1 pour UK — loin devant
- **euro_adequacy_deficit** et **euro_scarcity_ratio** capturent la dynamique continentale
- **Cat 30** (regime) n'a pas encore de ranking SHAP — sera validee au prochain sweep
