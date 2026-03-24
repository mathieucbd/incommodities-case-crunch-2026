# INCOMO 3 — Contexte Complet

## Competition

**InCommodities Case Crunch 2026** — competition organisee par InCommodities (trading d'electricite, Aarhus, Danemark).

**Objectif** : predire les prix spot horaires day-ahead de l'electricite pour la **France** (`fr_spot`) et le **Royaume-Uni** (`uk_spot`) sur 8 mois hors-echantillon (juillet 2024 — fevrier 2025).

**Metrique** : `RMSE(FR) + RMSE(UK) = SUM` — les deux marches comptent egalement.

**Contraintes** :
- Uniquement les donnees fournies (pas de donnees externes, APIs, etc.)
- Feature engineering autorise (rolling, calendrier, holidays via Python `holidays`)
- Le leaderboard Kaggle public utilise 60% du test — ne pas overfitter dessus
- Evaluation finale en live au siege d'InCommodities sur des donnees futures inedites

**Classement** :
| Submission | Score Kaggle | Notes |
|-----------|-------------|-------|
| 1ere place (Team KISS) | 23.14 | — |
| Notre meilleur (v5b) | 24.88 | 4-model regime |
| Notre meilleure val (v7) | 25.12 | 5-model regime, non soumis |

---

## Donnees

### Fichiers bruts (`data/raw/`)

| Fichier | Lignes | Periode | Colonnes |
|---------|--------|---------|----------|
| `x_train.csv` | 17 544 | 2022-07-01 → 2024-06-30 | 111 (3 id + 67 forecast + 34 lagged + 7 daily) |
| `y_train.csv` | 17 544 | idem | id, datetime_CET, datetime_UTC, fr_spot, uk_spot |
| `x_test.csv` | 5 833 | 2024-07-01 → 2025-02-28 | 111 (meme schema, sans targets) |

### Conventions de nommage

| Suffixe | Signification | Resolution |
|---------|---------------|------------|
| `_f` | Forecast (prevision D-1) | Horaire |
| `_la` | Lagged actual (valeur realisee D-1, shift 24h) | Horaire |
| (rien) | Quotidien (gas, emissions) | 1 valeur a 00:00 CET, NaN le reste |

### Colonnes cles

- **Load forecasts** (`*_load_f`) : demande electricite pour 10 pays/zones
- **Renewables** (`*_solar_f`, `*_wind_f`) : generation prevue solaire/eolien
- **Nuclear/Gas capacity** (`*_avcap_f`) : capacite disponible (UMM)
- **Interconnectors** : ATC/NTC (capacite), flows et costs (lagged) pour 6 liens UK
- **Prix lagged** (`*_spot_la`) : prix spot D-1 pour 10 zones europeennes
- **Gas/emissions** (`de_gas`, `uk_gas`, `eu_emission`, `uk_emission`) : quotidiens, sparse

### Split temporel

```
Train:  Jul 2022 ───────────────────────────── Jan 2024
Val:    ·········································· Feb 2024 ── Jun 2024  (3623h)
Test:   ··················································· Jul 2024 ── Feb 2025  (5833h)
```

UK utilise une fenetre de 12 mois (Feb 2023 → Jan 2024) pour le training — le regime post-crise gaz est plus representatif.

---

## Domaine

### Marche FR (SDAC / EPEX SPOT)
- **Nuclear-dominated** : 70% de la generation, prix souvent fixe par le nucleaire
- Enchère day-ahead a 12:00 CET (algorithme EUPHEMIA, couplage europeen SDAC)
- Les prix FR sont lies aux voisins (DE, BE, CH, ES) via le couplage
- **River temperature** >25°C = risque de curtailment nucleaire = spikes de prix
- Feature selection agressive : 32 features suffisent (de 330+ candidates)

### Marche UK (N2EX / Nord Pool)
- **Gas-dominated** : merit order cost = gas/efficiency + emission × 0.37
- Enchère separee du continent (post-Brexit), clearing 09:10 UK time
- 6 interconnecteurs sous-marins : IFA1, IFA2, ElecLink (FR-UK), NEMO (BE-UK), BritNed (NL-UK), Viking Link (DK1-UK)
- Heavy tails : prix de -205 a +1444 EUR/MWh → MAE loss plus robuste
- 154 features necessaires (plus de sources de variabilite)

### Concepts fondamentaux
- **Merit order** : les centrales sont classees par cout marginal croissant ; le prix = cout de la derniere centrale necessaire
- **Residual load** = demande - renouvelables → determine quelle centrale est marginale
- **Spark spread** = prix gaz / rendement + emission × facteur → cout marginal du gaz
- RMSE penalise quadratiquement les spikes → bien predire les extremes est critique

---

## Architecture du Projet

```
INCOMO 3/
├── data/raw/                          # Donnees brutes (x_train, y_train, x_test)
├── docs/
│   ├── Subject.md                     # Enonce de la competition
│   ├── context.md                     # Ce fichier
│   └── audit/AUDIT_PIPELINE.md        # Audit technique du pipeline
├── notebooks/
│   ├── 01_eda.ipynb                   # Exploration initiale
│   ├── 02_eda_deep_dives.ipynb        # Analyses approfondies
│   ├── 03_feature_selection.ipynb     # Selection SHAP/Boruta
│   ├── 04_catboost.ipynb              # Experimentations CatBoost
│   ├── 05_fr_error_diagnostic.ipynb   # Diagnostic erreurs FR
│   ├── 06_basis_modeling.ipynb        # Basis modeling UK (spot - MOC)
│   └── 07_results_recap.ipynb         # Synthese des resultats
├── scripts/
│   └── final_pipeline_v9.py           # Pipeline principal (~800 lignes)
├── src/
│   ├── __init__.py
│   ├── data_loading.py                # Chargement CSV (33 lignes)
│   ├── feature_engineering.py         # Construction features (1800 lignes, 290+ features)
│   └── models/
│       ├── __init__.py                # Re-exports
│       ├── metrics.py                 # RMSE, HBC, HBC monthly (34 lignes)
│       ├── targets.py                 # Preparation target stationnaire (42 lignes)
│       ├── tree_models.py             # CatBoost/LGB/XGB unifie (108 lignes)
│       ├── elastic_net.py             # Elastic Net + scaler (53 lignes)
│       ├── dnn.py                     # DNN PyTorch (80 lignes)
│       └── ensemble.py                # Regime weights optimization (109 lignes)
├── outputs/
│   ├── submission_v9.csv              # Soumission actuelle
│   ├── submission.csv                 # Copie pour Kaggle
│   ├── final_pipeline_v9_results.json # Scores detailles
│   ├── feature_selection_v5_fr.json   # 28 features FR selectionnees
│   └── uk_feature_research.json       # 150 features UK selectionnees
├── config.yaml                        # Hyperparametres et seuils
├── pyproject.toml                     # Dependances Python
├── CLAUDE.md                          # Instructions pour l'assistant IA
├── FINDINGS.md                        # Decisions, ce qui a marche/pas marche
└── SCORES.md                          # Historique des scores
```

---

## Pipeline v9

### Vue d'ensemble

Le pipeline est un orchestrateur (~800 lignes) qui enchaine :

1. **Chargement** : `data_loading.load_data()` → x_train, y_train, x_test
2. **Feature engineering** : `feature_engineering.build_features()` sur concat(train, test) pour eviter le cold-start aux frontieres
3. **Split** : train (Jul 2022 — Jan 2024) / val (Feb — Jun 2024) / test (Jul 2024 — Feb 2025)
4. **Target stationnaire** :
   - FR : `y = spot - EMA(spot_la, span=240h)` — l'EMA 10 jours capture le trend lent
   - UK : `y = spot - merit_order_cost` — le MOC capture ~70% du signal
5. **Entrainement** de 5 modeles par marche (10 total) sur le deviation target
6. **Ensemble** : poids par regime horaire (5 regimes × N modeles), grid search step 0.1
7. **HBC** : correction du biais systematique par heure (24 parametres)
8. **Retrain** sur train+val, prediction test, generation CSV

### Modeles

| Modele | FR | UK | Role |
|--------|-----|-----|------|
| **CatBoost** | depth=3, lr=0.059, RMSE loss, 32 feat | depth=8, MAE loss, 12m window, 154 feat | Meilleur standalone |
| **LightGBM** | 15 leaves, MAE loss | 63 leaves, MAE loss, 12m | Diversite (corr erreur 0.75 avec CB) |
| **XGBoost** | depth=4 (poids=0, hors ensemble FR) | depth=7, 12m | Diversite UK |
| **Elastic Net** | alpha=10, l1=0.9, 14 features non-zero | alpha=1, 32 non-zero, 12m | Lineaire, diversite |
| **DNN** | [192,96], Huber δ=5, dropout=0.2 | [768,384,192], Huber, dropout=0.3 | Non-lineaire, 356 feat |

### Ensemble par regime

5 regimes horaires : night (0-5h), morning (6-9h), day (10-16h), peak (17-21h), late (22-23h).

Les poids varient par regime — par exemple FR morning : EN=0.3, DNN=0.7 (les trees ratent les ramps matinaux), tandis que FR late : CB=0.5, DNN=0.4.

### Post-processing

- **HBC** : 24 corrections horaires (ex: h=7 +3.45, h=18 -3.20 pour FR)
- **Clipping** : percentiles 0.1% / 99.9% du training (FR: [-40.6, 800], UK: [-22.5, 798.5])

---

## Scores Actuels

### Validation (Feb — Jun 2024)

| | RMSE | +HBC |
|---|---|---|
| FR Ensemble | 15.81 | **15.71** |
| UK Ensemble | 9.59 | **9.49** |
| **SUM** | | **25.20** |

### Modeles individuels (+HBC)

| Modele | FR | UK |
|--------|-----|-----|
| CatBoost | 16.64 | 10.08 |
| LightGBM | 17.15 | 10.11 |
| XGBoost | 24.88 | 10.22 |
| Elastic Net | 16.56 | 11.32 |
| DNN | 16.73 | 10.99 |

---

## Historique des Versions

| Version | Changement principal | SUM val |
|---------|---------------------|---------|
| v3a | CatBoost + LightGBM + HBC | ~27.3 |
| v4 | + EMA 240h FR anchor | 26.74 |
| v4b | + MAE loss UK | 26.68 |
| v5 | + XGBoost 3e modele | ~26.2 |
| v5b | + Regime weights (5 regimes) | 26.39 |
| v6 | + Elastic Net 4e modele | 25.89 |
| v7 | + DNN 5e modele (Huber loss) | **25.12** |
| v8 | + rolling_336h, stress_index, UK 12m | 25.15 |
| v9 | Audit fixes + refactoring modulaire | 25.20 |

---

## Ce Qui a Marche / Pas Marche

### Gains confirmes

| Amelioration | Delta RMSE |
|-------------|-----------|
| Basis target UK (spot - MOC) | -0.68 UK |
| DNN comme 5e modele | -0.77 SUM |
| Elastic Net 4e modele | -0.50 SUM |
| XGBoost 3e modele | -0.48 SUM |
| EMA 240h FR (vs rolling 168h) | -0.37 FR |
| Optuna v2 FR (depth=3, lr=0.059) | -0.86 FR |
| UK 12m training window | -0.43 UK (standalone) |

### Echecs

| Tentative | Resultat |
|-----------|----------|
| GRU / GAT-GRU | +0.4 a +1.1 vs trees — sequences inutiles sur features tabulaires |
| Stacking (Ridge) | +2.1 FR — overfit du meta-learner |
| arcsinh transform | +0.3-0.5 — degrade systematiquement |
| Basis modeling FR (spot - MOC) | +4.5 FR — crise gaz fausse le MOC |
| Regime weight regularization | +0.20 val — shrinkage degrade les poids v7 |

---

## Fichiers de Reference

| Fichier | Description |
|---------|-------------|
| `FINDINGS.md` | Decisions detaillees, insights domaine, pistes non explorees |
| `SCORES.md` | Historique complet des scores, A/B tests, regime weights |
| `docs/audit/AUDIT_PIPELINE.md` | Audit technique : bugs trouves, fixes appliques, risques identifies |
| `config.yaml` | Seuils de feature engineering, hyperparametres par defaut |

---

## Dependances

```
Python >=3.10
pandas, numpy, scikit-learn
catboost, lightgbm, xgboost
torch (PyTorch — pour le DNN)
holidays (jours feries FR/UK)
optuna (tuning — notebooks uniquement)
matplotlib, seaborn (visualisation)
```

Package manager : `uv` (pas pip).
