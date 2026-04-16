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

**Classement Final (Private Leaderboard)** :
| Rang | Team | Score Private | Notes |
|------|------|---------------|-------|
| **1er** | **2sigmas** | **19.5475** | gagnant |
| **2e** | **AQTC (nous)** | **20.0781** | **submission_attack_averaged** |
| **3e** | **Team KISS** | **21.4419** | — |
| — | InCommodities benchmark | 21.4333 | battu |

**Gap au 1er** : 0.53 RMSE

**Notre meilleur Kaggle Public** : blend v17_85 + v9_15 = 23.20 (ancien leader public avant private reveal)

---

## Donnees

### Fichiers bruts (`data/raw/`)

**Données originales** (compétition) :
| Fichier | Lignes | Periode | Colonnes |
|---------|--------|---------|----------|
| `x_train.csv` | 17 544 | 2022-07-01 → 2024-06-30 | 111 (3 id + 67 forecast + 34 lagged + 7 daily) |
| `y_train.csv` | 17 544 | idem | id, datetime_CET, datetime_UTC, fr_spot, uk_spot |
| `x_test.csv` | 5 833 | 2024-07-01 → 2025-02-28 | 111 (meme schema, sans targets) |

**Données fulldata** (reçues 26/03/2026 post-compétition) :
| Fichier | Lignes | Periode | Colonnes |
|---------|--------|---------|----------|
| `x_train_full.csv` | 23 377 | 2022-01-01 → 2025-02-28 | 111 |
| `y_train_full.csv` | 23 377 | idem | id, datetime_CET, datetime_UTC, fr_spot, uk_spot |

Les fichiers fulldata incluent les actuals du test period (Jul 2024 → Feb 2025), permettant de valider les submissions sur les vraies valeurs.

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

**Original (compétition)** :
```
Train:  Jul 2022 ───────────────────────────── Jan 2024  (17544 samples)
Val:    ·········································· Feb 2024 ── Jun 2024  (3623h)
Test:   ··················································· Jul 2024 ── Feb 2025  (5833h)
```

**Fulldata (post-compétition)** :
```
Full:   Jan 2022 ────────────────────────────────────────────── Feb 2025  (23377 samples)
        ↑
        +6 mois additionnels
```

**Nouveaux holdouts (ACTION 3)** :
- **SPRING** : Train 2022-07-01→2024-01-31, Val 2024-02-01→2024-06-30
- **WINTER** : Train 2022-07-01→2024-01-31, Val 2023-07-01→2023-11-30 (antisaisonnier)
- **SPRING NEW (fulldata)** : Train 2022-01-01→2024-09-30, Val 2024-10-01→2024-12-31
- **WINTER NEW (fulldata)** : Train 2022-01-01→2024-06-30, Val 2024-11-01→2025-02-28

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

## Pipeline v17 (BEST)

### Vue d'ensemble

Le pipeline v17 = v16 (STL target FR) + Fix 2 (STL cohérent retrain) est un orchestrateur qui enchaine :

1. **Chargement** : `data_loading.load_data()` → x_train, y_train, x_test
2. **Feature engineering** : `feature_engineering.build_features()` (~290 features)
3. **Split** : train (Jul 2022 — Jan 2024) / val (Feb — Jun 2024) / test (Jul 2024 — Feb 2025)
4. **Target stationnaire** :
   - **FR** : `y = spot - STL_trend(spot_la, 168h)` — décomposition saisonnière (v16+)
   - **UK** : `y = spot - merit_order_cost` — le MOC capture ~70% du signal
5. **Entrainement** de 7 modeles FR + 8 modeles UK (15 total) sur le deviation target
6. **Ensemble** : poids par regime horaire (5 regimes × N modeles), grid search step 0.1
7. **HBC** : correction du biais systematique par heure (24 parametres)
8. **Retrain** sur train+val avec STL cohérent (Fix 2), prediction test, generation CSV

### Architecture v13/v17 (7 FR + 8 UK models)

**France (7 modèles)** :
| Modele | Config | Role |
|--------|--------|------|
| **CatBoost** | depth=3, Quantile:0.6, 28 feat | Base tree, asymétrique |
| **LightGBM** | 15 leaves, MAE loss | Diversité loss |
| **XGBoost** | depth=4, PseudoHuber:20 | Diversité loss |
| **Elastic Net** | alpha=10, l1=0.9, 14 non-zero | Linéaire, corrélation 0.60-0.70 avec trees |
| **DNN** | [192,96], Huber:5, dropout=0.2 | Non-linéaire, 356 feat, corr 0.77-0.82 |
| **RidgeF** | Features fondamentales uniquement | Très décorrélé (corr=0.65 FR, 0.41 UK) |
| **SR** | Ridge meta-learner sur erreurs v9 | Stacking résiduel, T2 models (55 membres) |

**UK (8 modèles)** :
| Modele | Config | Role |
|--------|--------|------|
| **CatBoost** | depth=8, MAE loss, 12m window, 154 feat | Base tree |
| **LightGBM** | 63 leaves, Huber:5, 12m | Diversité |
| **XGBoost** | depth=7, PseudoHuber:20, 12m | Diversité |
| **Elastic Net** | alpha=1, 32 non-zero, 12m | Linéaire |
| **DNN** | [768,384,192], MSE, dropout=0.3, 12m | Non-linéaire |
| **RidgeF** | Features fondamentales uniquement | Décorrélé |
| **SR** | Ridge meta-learner, T2 models (65 membres avec combos) | Stacking résiduel |
| **XGB_cluster** | XGB sur shifted 6h clusters (4 clusters) | Segmentation temporelle (v13) |

### Ensemble par regime (5 regimes)

5 regimes horaires : night (0-5h), morning (6-9h), day (10-16h), peak (17-21h), late (22-23h).

**Exemple FR morning (submission_attack_spring)** :
- CB=0.1, LGB=0.0, XGB=0.0, EN=0.1, DNN=0.3, RidgeF=0.0, SR=0.5

**Exemple FR morning (submission_attack_winter)** :
- CB=0.1, LGB=0.0, XGB=0.0, EN=0.1, DNN=0.1, RidgeF=0.0, SR=0.7

Les poids divergent radicalement selon la saison (ACTION 3). DNN disparaît en winter, SR ultra-dominant.

### Post-processing

- **HBC** : 24 corrections horaires (ex: h=7 +3.45, h=18 -3.20 pour FR)
- **Clipping** : percentiles 0.1% / 99.9% du training (FR: [-40.6, 800], UK: [-22.5, 798.5])

---

## Scores Actuels

### Validation — BEST (attack_averaged, dual holdout)

| Submission | FR +HBC | UK +HBC | SUM | Notes |
|------------|---------|---------|-----|-------|
| **attack_averaged** | **14.57** | **8.80** | **23.37** | moyenne spring+winter |
| attack_spring | 14.35 | 8.83 | 23.18 | baseline v17 |
| attack_winter | 14.79 | 8.77 | 23.56 | poids winter seuls |

### Kaggle (Private Leaderboard)

| Submission | Private Score | Rank |
|------------|---------------|------|
| **attack_averaged** | **20.0781** | **2e/68** |
| attack_winter | 20.2397 | — |
| blend v17_85 + v9_15 | 20.4818 | ancien best public |

### Modeles individuels v17 (+HBC, holdout spring)

| Modele | FR | UK |
|--------|-----|-----|
| CatBoost | 16.08 | 9.78 |
| LightGBM | 17.58 | 10.15 |
| XGBoost | 18.67 | 10.11 |
| Elastic Net | 16.44 | 13.10 |
| DNN | 16.69 | 10.96 |
| RidgeF | ~17.5 | ~11.5 |
| SR (stacking résiduel) | 16.55 | 9.80 |

---

## Historique des Versions

| Version | Changement principal | SUM val | Kaggle |
|---------|---------------------|---------|--------|
| v3a | CatBoost + LightGBM + HBC | ~27.3 | 25.28 |
| v4 | + EMA 240h FR anchor | 26.74 | — |
| v4b | + MAE loss UK | 26.68 | — |
| v5 | + XGBoost 3e modele | ~26.2 | — |
| v5b | + Regime weights (5 regimes) | 26.39 | 24.88 |
| v6 | + Elastic Net 4e modele | 25.89 | — |
| v7 | + DNN 5e modele (Huber loss) | 25.12 | — |
| v8 | + rolling_336h, stress_index, UK 12m | 25.15 | — |
| v9 | Audit fixes + loss diversity | 25.10 | — |
| v10 | + Nystroem kernel features | 25.07 | — |
| v11 | + RidgeF + Stacking Résiduel T2 | 23.83 | 24.23 (+0.40 overfit) |
| v11b | Anti-overfitting (3+3 combos, alpha=500) | 23.85 | 23.92 (+0.07 gap) |
| v13 | + XGB_cluster UK (8e membre) | 23.73 | 23.94 (+0.21 gap) |
| v16 | + STL target FR (remplace EMA 240h) | 23.08 | 23.26 (+0.18 STL bias) |
| **v17** | **+ Fix 2 STL cohérent retrain** | **23.08** | **23.20** (+0.12 gap) |
| **attack_averaged** | **Dual holdout spring+winter** | **23.37** | **20.0781** ⭐ |

**Blending** :
- blend v17_85 + v9_15 : Kaggle 20.4818 (ancien best public)
- blend v13_85 + v9_15 : Kaggle 21.2563

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
