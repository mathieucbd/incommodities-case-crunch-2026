# INCOMO 3 — Handoff technique

Competition : InCommodities Case Crunch 2026
Objectif : Predire les prix spot horaires (day-ahead) pour **FR** (fr_spot) et **UK** (uk_spot).
Metrique Kaggle : SUM of RMSE (FR_RMSE + UK_RMSE).

---

## Fichiers a copier (ordre de priorite)

### 1. OBLIGATOIRES (pipeline ne tourne pas sans)

| Fichier | Role |
|---------|------|
| `src/data_loading.py` | Charge x_train.csv, y_train.csv, x_test.csv et merge |
| `src/feature_engineering.py` | Construit ~290 features derivees (30 categories) a partir des 111 colonnes brutes. **C'est le coeur du projet.** |
| `config.yaml` | Parametres : seuils feature engineering, hyperparams modeles, dates holdout |
| `scripts/final_pipeline.py` | Pipeline complete : load → features → train CB+LGB → ensemble → HBC → submission |
| `outputs/feature_selection_v5_fr.json` | Liste des 27 features selectionnees pour FR (SHAP v5) |
| `outputs/uk_feature_research.json` | Liste des 150 features confirmees pour UK (SHAP + Boruta) |
| `data/raw/x_train.csv` | Features brutes train (17544 rows x 111 cols, Jul 2022 — Jun 2024) |
| `data/raw/y_train.csv` | Targets train (fr_spot, uk_spot) |
| `data/raw/x_test.csv` | Features brutes test (5833 rows, Jul 2024 — Feb 2025) |

### 2. UTILES (pas strictement necessaires mais aident)

| Fichier | Role |
|---------|------|
| `src/validation.py` | Utilitaires de validation temporelle (holdout, expanding window) |
| `src/feature_selection.py` | Methodes de selection (Lasso, SHAP, Boruta) |
| `src/models/catboost_model.py` | Wrapper OOP CatBoost + Optuna |

---

## Architecture du modele

### FR (France)
- **Target stationnarise** : `y = fr_spot - EMA(fr_spot_la, span=240h)`
  - `fr_spot_la` = prix spot look-ahead (connu au moment de la prediction day-ahead)
  - EMA 240h = ancre qui suit le niveau de prix
  - Reconstruction : `pred_spot = EMA + model.predict(X)`
- **Modele** : CatBoost (depth=3, lr=0.059, Optuna-tuned) + LightGBM ensemble (w_cb=0.70)
- **28 features** selectionnees par SHAP
- **Poids echantillons** : `w = exp(-2 * days_ago / 365) / clip(rolling_168h_std^2, 1)`
  - Recency decay (data recente compte plus)
  - Inverse variance (periodes stables comptent plus)

### UK (Royaume-Uni)
- **Target basis** : `y = uk_spot - uk_merit_order_cost`
  - merit_order_cost = cout marginal gaz (feature deja dans x_train)
  - Reconstruction : `pred_spot = merit_order_cost + model.predict(X)`
- **Modele** : CatBoost (depth=8, lr=0.03, **loss=MAE**) + LightGBM ensemble (w_cb=0.70)
- **150 features** confirmees par SHAP + Boruta
- **Pas de poids** (les poids degradent UK)
- **MAE loss** : UK a des queues tres lourdes (range [-205, +1444]), MAE est plus robuste

### Post-processing (les deux)
- **HBC (Hourly Bias Correction)** : 24 params, `bias[h] = mean(actual[h] - pred[h])` sur val
- **Clipping** : quantiles 0.1% / 99.9% du train
- **Retrain iterations floor** : 500 (le full-data retrain a besoin de plus d'iterations)

---

## Dependances Python

```
pandas numpy catboost lightgbm pyyaml holidays
```

---

## Lancer la pipeline

```bash
cd "INCOMO 3"
python scripts/final_pipeline.py
```

Genere `outputs/submission_v4b_mae_uk.csv`.

---

## Categories de features (feature_engineering.py)

Le fichier construit 30 categories :
1. Time features (hour, dow, month, doy_sin/cos)
2. Lag features (spot_la shifts)
3. Rolling stats (mean, std, min, max sur 24h/168h/336h)
4. Deviation features (spot_la - rolling_mean)
5. Z-scores (residual zscore 7d/14d)
6. Residual load (load - renewables)
7. Continental aggregates (DE+FR+BE+NL residual)
8. Scarcity ratios (load / capacity)
9. Cross-market spreads (FR-UK, FR-DE)
10. Momentum features (change 24h/48h)
11. Ramp features (3h changes)
12. Gas/carbon features (ratios, rolling means)
13. Wind/nuclear deviations
14. Mean reversion strength
15. Dynamic marginal cost
16. NTC (interconnector capacity)
17. Merit order cost features
... et plus.

---

## Resultats actuels

| Version | FR RMSE | UK RMSE | SUM | Kaggle |
|---------|---------|---------|-----|--------|
| v3a | - | - | - | 25.28 |
| v4 (EMA 240h) | 16.91 | 9.83 | 26.74 | pas soumis |
| v4b (+MAE UK) | 16.91 | 9.78 | 26.68 | pas soumis |
| 1ere place | - | - | - | 24.45 |
