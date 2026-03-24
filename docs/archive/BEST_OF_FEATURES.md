# Best-Of Features Techniques — INCOMO 3

> Ce qui fait que notre modele performe. Score Kaggle: **25.87** (SUM RMSE FR+UK), 2e/3e place.
> Val: FR=16.97, UK=9.83, SUM=26.80

---

## Architecture du modele

### 1. Cibles stationnaires (le game changer principal)

Le choix de la cible est la decision #1 du pipeline. Predire le prix brut ne marche pas a cause du changement de regime (crise energetique 2022-23 vs regime normal 2024-25).

**FR — Deviation stationnaire:**
```
y = spot - rolling_168h_mean(spot_la)    # centree ~0, stationnaire
pred_spot = rolling_168h_mean + model.predict(X)
```
- La rolling mean 168h (1 semaine) capture le niveau de prix courant
- Le modele apprend seulement les **deviations** autour de ce niveau
- Poids: `w = exp(-2 * days_ago / 365) / clip(rolling_168h_std^2, 1)` — recence + stabilite

**UK — Basis (spot - merit_order_cost):**
```
y = spot - merit_order_cost              # marge de marche, ~0
pred_spot = merit_order_cost + model.predict(X)
```
- `merit_order_cost` = cout fondamental du mix electrique (gas-driven)
- Le modele apprend la **prime de marche** au-dessus du cout fondamental
- Pas de poids d'echantillonnage (UK: pas de regime de crise dans les donnees)

**Impact:** -10 EUR RMSE sur FR (de 26+ a 17). Sans ca, biais systematique de +20 EUR.

### 2. Ensemble CatBoost + LightGBM

- **CatBoost** (poids 95% FR, 65% UK) : moteur principal
- **LightGBM** (5% FR, 35% UK) : diversite de gradient
- Poids optimises par grid search sur validation

### 3. Hourly Bias Correction (HBC)

Post-processing: `pred_corrige[h] = pred[h] + mean(erreur[h])` par heure (24 params).
- FR: corrections de -5.3 (h18) a +3.9 (h7)
- UK: corrections de -3.8 (h0) a +3.8 (h13)
- Gain: ~0.4 RMSE

---

## Les 28 features FR (selection v5 — SHAP stationnaire + Boruta)

| # | Feature | Formule | Pourquoi ca marche |
|---|---------|---------|-------------------|
| 1 | `fr_spot_la_roll_168h_mean` | `mean(spot_la, 168h)` | Ancre le niveau de prix courant. #1 SHAP de loin. |
| 2 | `fr_residual_zscore_14d` | `(resid_load - mean_14d) / std_14d` | Tension anormale sur le reseau FR en contexte 14j. |
| 3 | `uk_residual_zscore_14d` | Idem pour UK | Contagion UK→FR via interconnexion. |
| 4 | `fr_spot_la_deviation_168h` | `spot_la - roll_168h_mean` | Ecart courant vs tendance — mean-reversion signal. |
| 5 | `continental_residual_load` | `load_EU - wind_EU - solar_EU` | Demande thermique nette europeenne. |
| 6 | `euro_scarcity_ratio` | `(euro_load - zero_MC) / gas_cap` | Tension offre/demande EU: combien du gaz est mobilise. |
| 7 | `wind_nuke_deviation_gap` | `(wind/norm_wind) - (nuke/norm_nuke)` | Ecart normalise vent vs nucleaire — anti-correle aux prix. |
| 8 | `fr_residual_change_24h` | `resid_load - resid_load.shift(24)` | Momentum 24h de la charge residuelle. |
| 9 | `uk_spot_la_deviation_168h` | `uk_spot_la - roll_168h_mean` | Signal cross-market UK. |
| 10 | `uk_price_per_mw_7d` | `mean_price_7d / mean_resid_load_7d` | Cout marginal implicite UK sur 7j. |
| 11 | `uk_spot_la_roll_168h_std` | `std(uk_spot_la, 168h)` | Volatilite UK — risque de contagion. |
| 12 | `fr_spot_la_deviation_24h` | `spot_la - roll_24h_mean` | Signal intraday de deviation. |
| 13 | `fr_residual_ramp_3h` | `resid_load.diff(3)` | Rampe rapide de demande (3h). |
| 14 | `de_residual_load` | `de_load - de_wind - de_solar` | Demande thermique Allemagne. FR est couple a DE. |
| 15 | `fr_load_roll_168h_mean` | `mean(fr_load, 168h)` | Niveau de charge structurel. |
| 16 | `carbon_to_gas_ratio` | `eu_emission / nl_gas` | Ratio carbone/gaz — determine le switching coal/gas. |
| 17 | `fr_mean_reversion_strength` | `deviation_168h / std_168h` | Z-score du prix — force de rappel vers la moyenne. |
| 18 | `uk_load_change_24h` | `uk_load - uk_load.shift(24)` | Choc de demande UK (impact via cable). |
| 19 | `uk_spot_la_roll_168h_mean` | `mean(uk_spot_la, 168h)` | Niveau de prix UK — reference cross-market. |
| 20 | `doy_cos` | `cos(2pi * day_of_year / 365)` | Saisonnalite annuelle. |
| 21 | `fr_spot_la_roll_168h_std` | `std(fr_spot_la, 168h)` | Volatilite FR recente. |
| 22 | `fr_dynamic_marginal` | `w * merit_order + (1-w) * nuclear_MC` | Cout marginal interpole selon scarcity. |
| 23 | `de_gas` | Prix gaz Allemagne (raw) | Cout du combustible marginal EU. |
| 24 | `uk_nuclear_avail_ratio` | `uk_nuclear / max(uk_nuclear)` | Disponibilite nucleaire UK. |
| 25 | `uk_load_roll_168h_mean` | `mean(uk_load, 168h)` | Niveau de charge UK structurel. |
| 26 | `ntc_dk1-uk_f` | NTC Danemark→UK (raw) | Capacite d'import nordique du UK. |
| 27 | `fr_gas_roll_168h_mean` | `mean(fr_gas, 168h)` | Tendance du cout gaz FR sur 7j. |
| 28 | `X_fr_roll_mean_x_uk_ppm` | `roll_168h_mean * uk_price_per_mw_7d` | **Interaction** niveau FR x cout marginal UK. |

---

## Les 150 features UK (Top-30 SHAP basis + Boruta confirmed)

| # | Feature | SHAP | Formule | Pourquoi |
|---|---------|------|---------|----------|
| 1 | `euro_scarcity_ratio` | 5.95 | `(euro_load - zero_MC) / gas_cap` | Tension EU = prix UK (island importe cher) |
| 2 | `gas_spread_uk_eu` | 3.84 | `uk_gas - nl_gas` | Spread gaz UK-EU. UK gas = prix marginal UK. |
| 3 | `uk_wind_x_gas` | 3.60 | `uk_wind * uk_gas` | Interaction vent x gaz : quand le vent tombe + gaz cher = spike |
| 4 | `uk_gas_utilization` | 3.41 | `(ccgt+ocgt+coal+bio) / gas_cap` | Taux d'utilisation des centrales gaz. |
| 5 | `uk_spot_la_deviation_24h` | 3.38 | `spot_la - roll_24h_mean` | Signal momentum intraday. |
| 6 | `uk_wind_share_flexible` | 2.87 | `wind / (load - baseload)` | Part du vent dans la demande flexible. |
| 7 | `uk_scarcity_ratio` | 2.28 | `resid_load / dispatchable` | Tension offre/demande UK specifique. |
| 8 | `atc_uk-fr-3_f_ratio` | 2.27 | `atc / median(atc)` | Capacite cable FR-UK normalise (#3). |
| 9 | `uk_load_zscore_14d` | 2.10 | `(load - mean_14d) / std_14d` | Anomalie de demande en contexte 14j. |
| 10 | `uk_load_pct_weekly_peak` | 1.88 | `load / max(load, 168h)` | Position dans la semaine (pointe = cher). |
| 11 | `uk_self_sufficiency` | 1.70 | `total_domestic / load` | Autosuffisance = besoin d'import. |
| 12 | `uk_supply_demand_ratio` | 1.69 | `(dispatchable + wind) / load` | Ratio S/D global. |
| 13 | `fr_nuclear_rolling_7d_mean` | 1.63 | `mean(fr_nuke, 168h)` | Dispo nucleaire FR = prix cable FR→UK. |
| 14 | `uk_wind_pen_squared` | 1.47 | `(wind / load)^2` | Effet non-lineaire du vent (rendements decroissants). |
| 15 | `uk_security_margin` | 1.46 | `dispatchable - resid_load` | Marge de securite en MW (negatif = stress). |

---

## Concepts cles qui font la difference

### A. Separation fondamental vs prime de marche

Le prix electrique = **cout fondamental** (merit order / rolling mean) + **prime de marche** (deviation). On predit la prime, pas le prix. Ca rend le probleme stationnaire et elimine le biais de regime.

### B. Features cross-market (FR↔UK)

FR et UK sont connectes par 3 cables sous-marins (IFA 1/2, ElecLink). ~20% des features FR sont des signaux UK et vice versa. Les plus importants :
- `uk_residual_zscore_14d` pour FR (#3 SHAP)
- `uk_price_per_mw_7d` pour FR (#10)
- `euro_scarcity_ratio` pour UK (#1 SHAP)
- `fr_nuclear_rolling_7d_mean` pour UK (#13)

### C. Scarcity / supply-demand

Les ratios de rarete (`scarcity_ratio`, `security_margin`, `gas_utilization`) capturent la **tension** sur le systeme. Quand la marge tombe, les prix montent de facon non-lineaire — c'est la forme de la merit order curve (convexe).

### D. Rolling statistics multi-horizon

- **24h** : signal intraday (deviation, ramp, zscore)
- **168h (7j)** : tendance hebdo (mean, std, mean-reversion)
- **14j** : contexte structurel (zscore 14d)

La combinaison des trois horizons permet au modele de distinguer un choc temporaire (deviation 24h) d'un changement de regime (zscore 14d).

### E. Poids d'echantillonnage (FR uniquement)

```
w = exp(-2 * days_ago / 365) / clip(rolling_168h_std^2, 1)
```
- **Recence** : les donnees recentes comptent plus (demi-vie ~6 mois)
- **Stabilite** : les periodes calmes comptent plus (evite le bruit des crises)
- Sur UK : pas de poids (pas de regime de crise a downweighter)

### F. Interaction feature cle (FR)

```
X = fr_spot_la_roll_168h_mean * uk_price_per_mw_7d
```
Capture l'interaction entre le niveau de prix FR et le cout marginal implicite UK. Quand les deux sont eleves → signal fort de marche tendu.

### G. Gas-driven marginal cost

Le gaz naturel fixe le prix marginal dans 70-80% des heures (UK encore plus). Features gas directes :
- `de_gas`, `fr_gas_roll_168h_mean` (FR)
- `gas_spread_uk_eu`, `uk_gas_utilization`, `uk_wind_x_gas` (UK)
- `carbon_to_gas_ratio` (switching coal↔gas)

---

## Hyperparametres optimaux

### FR CatBoost (Optuna v2, 300 trials)

| Param | Valeur | Interpretation |
|-------|--------|----------------|
| depth | 3 | Modele peu profond → regularisation forte |
| learning_rate | 0.059 | Agressif mais early-stopped tot (~50 iters) |
| l2_leaf_reg | 4.42 | Regularisation L2 moderee |
| colsample_bylevel | 0.228 | Dropout de features tres fort (22% par split) |
| subsample | 0.533 | Bagging 53% (anti-overfitting) |
| min_child_samples | 14 | Feuilles suffisamment peuplees |
| random_strength | 0.9 | Bruit dans le splitting |
| **best_iter** | **49** | Seulement 49 arbres ! Modele tres parcimonieux. |

**Insight:** depth=3 + 49 iterations + colsample=0.22 = un modele **tres** regularise. FR est un marche bruyant, le modele gagne en etant conservateur.

### UK CatBoost (config actuelle, pre-Optuna)

| Param | Valeur | Interpretation |
|-------|--------|----------------|
| depth | 8 | Profond — UK a des interactions complexes (island) |
| learning_rate | 0.03 | Plus conservateur |
| l2_leaf_reg | 5 | Moderee |
| colsample_bylevel | 0.8 | Peu de dropout (150 features = besoin de beaucoup) |
| subsample | 0.8 | Bagging leger |
| **best_iter** | ~200-300 | Plus d'arbres que FR |

**Insight:** UK necessite un modele plus complexe (150 features, d=8). Le marche UK a plus de structure (island = gas-marginal quasi-toujours) donc plus de signal a capturer.

---

## Ce qu'on n'a PAS encore exploite

1. **Autocorrelation des residus** (r=0.80 lag-1h) — modele AR sur residus pourrait gagner ~1-2 RMSE
2. **HBC monthly x hour** (120 params vs 24) — -1.34 SUM sur val, risque d'overfit
3. **Retrain iterations** — augmenter de 200 a 500 pour le retrain final (FR: -0.64 RMSE)
4. **Optuna UK** — HP search focalisee pas encore integree
