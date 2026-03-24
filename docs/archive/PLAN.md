# INCOMO 3 — Plan Complet : De l'EDA aux Resultats Finaux

## 0. Vue d'ensemble du projet

**Objectif** : Predire les prix spot horaires Day-Ahead pour FR et UK sur 8 mois (Jul 2024 — Fev 2025).
**Metrique** : RMSE moyenne sur les 2 cibles.
**Contrainte** : Aucune donnee externe (uniquement CSVs fournis + `holidays` lib).
**Donnees** : 17,544 rows train (Jul 2022 — Jun 2024), 5,833 rows test, 111 colonnes brutes.

---

## Phase 1 — EDA [FAIT]

### 1.1 EDA General (`notebooks/01_eda.ipynb`)
- Distribution des cibles FR/UK (histogrammes, box plots par heure/mois)
- Carte de NaN par colonne (DK1-UK 72-74%, NL 61%)
- Correlations top-20 avec les cibles
- Patterns temporels (prix par heure, jour de semaine, mois)
- Analyse des spikes (>500 EUR) et drivers
- Distribution shift train vs test (crise 2022 vs normal 2024)
- Merit order : residual load vs prix, wind vs prix

### 1.2 EDA Deep Dives (`notebooks/02_eda_deep_dives.ipynb`)
- **Spark spread** : r=0.89 FR, r=0.88 UK — biais horaire (nuit << spark)
- **Thermal Need x Gas Price** : interaction bat les features individuelles
- **Ete** : volatilite 3x plus elevee, duck curve solaire, nuclear en maintenance
- **Nucleaire FR** : seuil ~35GW, relation non-lineaire, baseload_gap > thermal_need
- **UK Wind** : non-lineaire, prix negatifs > 50% penetration, rampes 3h
- **Regimes** : crise 2022 (322 EUR) vs normal 2024 (47 EUR), spark-spot stable
- **Congestion** : fr_uk_util r=-0.69, decouplage quand > 90%

---

## Phase 2 — Feature Engineering [FAIT]

### 2.1 Features (471 colonnes, 32 categories)

**Fichier** : `src/feature_engineering.py`
**Pipeline** : `build_features(df, config) -> DataFrame` — stateless, identique train/test.
**Detail complet** : voir `docs/FEATURES_COMPLET.md`

| Bloc | Count | Statut |
|------|-------|--------|
| Cat 1-16 (base) | 221 | Done |
| Cat 17-23 (offre/demande, z-scores, SDE) | ~45 | Done |
| Cat 24-30 (research-backed) | ~73 | Done |
| Cat 32 (advanced price proxies) | 22 | Done |
| Colonnes brutes | 112 | |
| **GRAND TOTAL** | **~471** | |

---

## Phase 3 — Validation (`src/validation.py`) [FAIT]

### 3.1 Holdout chronologique (principal)
- Train : Jul 2022 — Jan 2024 (13,921 rows)
- Val : Feb 2024 — Jun 2024 (3,623 rows)
- Date : `config.validation.holdout_start = "2024-02-01"`

### 3.2 CV Temporelle (validation croisee)
- 4 folds expanding window (scripts/temporal_cv_v1.py)
- v7 (d=3, 27 feat) : Avg RMSE=16.05 (+-2.35) — best config
- v7 wins 2/4 folds vs old (d=8)
- Fold 4 (holdout) coherent avec F1-F3 → pas d'overfitting

---

## Phase 4 — Modeles [FAIT]

### 4.1 CatBoost FR [FAIT]

**Config finale (Optuna v2)** :
- iterations=15000, lr=**0.059**, **depth=3**, l2_leaf_reg=**4.4**, subsample=**0.53**, colsample_bylevel=**0.23**
- min_child_samples=14, random_strength=0.9
- Target : `spot - roll_168h_mean(spot_la)` (stationnaire)
- Weights : `exp_decay(2.0) / std_168h²`
- 28 features (v5-28)
- RMSE val : **17.19** (+HBC : **16.98**)

**Hyperparameter tuning v2** (scripts/hyperparam_tuning_v2_fr.py) :
- Optuna TPE, 300 trials, 7 params + feature set jointly optimized
- Gain vs v1 : 17.84 → **17.19** (-0.65 RMSE)
- fANOVA importance : depth=83%, colsample=7%, feature_set=6%
- Top-10 trials : 100% depth=3, 100% v5-28
- Insight : la vraie regularisation vient du sous-echantillonnage agressif (csbl=0.23, ss=0.53), pas de L2

| Param | v1 | Optuna v2 | Impact |
|-------|-----|-----------|--------|
| depth | 3 | 3 | Confirme (83% importance) |
| learning_rate | 0.03 | **0.059** | Plus rapide + early stop |
| l2_leaf_reg | 30 | **4.4** | Moins de reg L2 |
| colsample | 0.5 | **0.23** | Tres agressif |
| subsample | 0.7 | **0.53** | Plus de stochasticite |
| min_child | 1 | **14** | Regularise par feuilles |

**Feature discovery v6** (scripts/feature_discovery_v6.py) :
- 436 candidats generes (interactions, ratios, seuils, rolling windows)
- 1 candidat utile : `X_fr_spot_la_roll_168h_mean_x_uk_price_per_mw_7d` (delta=-0.14)
- 3 features confirmees par noise probing : l'interaction + `fr_price_per_mw_7d` + `fr_load_price_signal_load`
- Conclusion : les v5-27 sont quasi-optimales

### 4.2 CatBoost UK [FAIT]

**Config finale** :
- iterations=15000, lr=0.02, **depth=6**, l2_leaf_reg=5, colsample_bylevel=0.7, subsample=0.8
- Target : `spot - merit_order_cost` (basis modeling)
- Weights : `exp_decay(2.0)` (pas de variance normalization)
- 100 features (SHAP v4 top-100)
- RMSE val : **10.60**

**Basis vs Stationary** :
- Basis (spot - merit_order_cost) : RMSE=10.60
- Stationary (spot - roll_168h_mean) : RMSE=11.47
- Basis wins car UK a une relation merit-order plus forte

### 4.3 LightGBM [FAIT]

- FR LightGBM : RMSE=18.18 (meme features que CatBoost)
- UK LightGBM : RMSE=11.27 (meme features que CatBoost)
- LGB sert de diversite pour l'ensemble

### 4.4 Experiments realises
- Feature sweep v2 : 20/30/50/75/100/all features
- A/B test 15 configs : target x Cat32 x weights
- 19 stationarity methods testees
- SHAP v4 avec target stationnaire
- Feature selection v5 : Boruta + bucket + RFE → 27 features FR
- Hyperparameter tuning v1 : lr x depth x reg x sampling
- Feature discovery v6 : residual-driven candidate search
- UK tuning v1 : depth + feature selection + basis vs stationary
- CV temporelle : 4-fold expanding window

---

## Phase 5 — Ensemble [FAIT]

**Methode** : Weighted average optimise sur validation.

| Cible | w_CatBoost | w_LightGBM | RMSE |
|-------|-----------|-----------|------|
| FR | 0.70 | 0.30 | 17.59 |
| UK | 0.95 | 0.05 | 10.59 |

---

## Phase 6 — Post-processing [FAIT]

1. **Clipping** : borner aux quantiles 0.1%/99.9% du train (FR: [-40.6, 800], UK: [-22.5, 798.5])
2. **Hourly bias correction** : `bias[h] = mean(y_true[h] - y_pred[h])` calcule sur val → applique au test

---

## Phase 7 — Submission [FAIT]

1. Retrain sur TOUT le training set (17,544 rows) avec hyperparams geles
2. Iterations fixees au best_iteration du val + marge de 50
3. Ensemble CatBoost + LightGBM avec poids optimises
4. HBC + clipping
5. CSV genere : `outputs/submission.csv` (5,833 rows)

---

## Scores finaux

### Validation (Feb-Jun 2024)

| Modele | FR RMSE | UK RMSE | Combined |
|--------|---------|---------|----------|
| CatBoost Optuna v2 | **17.19** | *en cours* | — |
| + HBC | **16.98** | *en cours* | — |

*UK : best ever = 9.84 (basis, d=8, 200f, no weights) — Optuna a faire*

### Evolution FR

| Etape | Config | RMSE | Delta |
|-------|--------|------|-------|
| v1 | Raw spot, 20 feat | 27.52 | — |
| v2 | arcsinh(spot) | 26.10 | -1.42 |
| v3 | + Cat32 (31 feat) | 24.79 | -2.73 |
| v4 | + weights(2.0) | 24.45 | -3.07 |
| v5 | + target stationnaire | 19.99 | -7.53 |
| v6 | + SHAP v4 ranking | 18.55 | -8.97 |
| v7 | + depth=3, l2=30, csbl=0.5 (grid v1) | 17.84 | -9.68 |
| v8 | + interaction feature | 17.70 | -9.82 |
| **v9** | **Optuna v2 (lr=0.06, l2=4.4, csbl=0.23)** | **17.19** | **-10.33** |
| **v9+HBC** | **+ hourly bias correction** | **16.98** | **-10.54** |

### Evolution UK

| Etape | Config | RMSE |
|-------|--------|------|
| v1 | raw spot (Optuna d=7, 100f) | 10.66 |
| **v2** | **basis (d=8, 200f, no weights)** | **9.84** |
| v3 | basis (d=6, 100f, pipeline — regression) | 10.60 |

### Fichiers de sortie

| Fichier | Description |
|---------|-------------|
| `outputs/submission.csv` | Soumission (a regenerer apres UK) |
| `outputs/final_pipeline_results.json` | Resultats pipeline v1 |
| `outputs/hyperparam_tuning_v2_fr.json` | **Optuna v2 FR — 300 trials** |
| `outputs/hyperparam_tuning_v1_fr.json` | Grid search v1 FR |
| `outputs/feature_selection_v5_fr.json` | 27 features FR selectionnees |
| `outputs/feature_discovery_v6.json` | Feature discovery resultats |
| `outputs/uk_tuning_v1.json` | UK tuning v1 (stationary — obsolete) |
| `outputs/shap_ranking_v4_stationary.json` | SHAP ranking apres dedup |
