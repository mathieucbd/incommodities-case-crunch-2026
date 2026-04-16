# Cleanup Plan — InCommodities Case Crunch 2026
*Objectif : repo GitHub propre documentant la solution gagnante (2e/68, RMSE 20.08)*

---

## Résultats finaux (à rappeler)

| Métrique | Score |
|----------|-------|
| **Kaggle Private LB** | **20.0781 (2e/68)** |
| 1er (2sigmas) | 19.5475 |
| Benchmark InCommodities | 21.4333 |
| Soumission gagnante | `submission_attack_averaged.csv` |
| Val RMSE (proxy) | ~23.37 (FR=14.57, UK=8.80) |

---

## Architecture de la solution gagnante

```
7 modèles FR :  CatBoost (Quantile:0.6) + LightGBM (MAE) + XGBoost (PseudoHuber)
                + ElasticNet + DNN[192,96] + Ridge(fondamentales) + SR
8 modèles UK :  idem + XGB_cluster (4 clusters temporels 6h-décalés)

Target FR  : spot - STL_trend(spot_la, period=168h)
Target UK  : spot - merit_order_cost (gas/0.50 + emission×0.37)

SR (Stacking Résiduel) : 39-44 modèles légers par groupe thématique
                         → Ridge meta-learner prédit l'erreur de l'ensemble v9

Ensemble   : poids par régime horaire (5 régimes : night/morning/day/peak/late)
HBC        : correction biais par heure (24 valeurs)

Clé du succès : double calibration saisonnière
  - Holdout SPRING (→Jan'24, Val Feb-Jun'24) → poids printemps
  - Holdout WINTER (→Jan'24, Val Jul-Nov'23) → poids hiver
  - Prédiction finale = moyenne des 2 ensembles → -2.74 RMSE vs baseline
```

---

## État actuel du repo

### Fichiers trackés (branch v2, dirty)
```
M  config.yaml                      → garder (hyperparams)
M  pyproject.toml                   → garder
M  scripts/final_pipeline_v11.py    → garder (étape clé SR)
M  src/feature_engineering.py       → garder
M  src/models/__init__.py           → garder
M  src/models/cnn_lstm.py           → SUPPRIMER (jamais utilisé en production)
M  src/models/dnn.py                → garder
M  src/models/elastic_net.py        → garder
M  src/models/ensemble.py           → garder
M  src/models/targets.py            → garder
M  src/models/tree_models.py        → garder
D  notebooks/04-10_*                → déjà supprimés (OK)
D  scripts/compare_detrending.py    → déjà supprimés (OK)
D  scripts/feature_reselection.py   → déjà supprimés (OK)
D  src/models/cnn.py, lstm.py       → déjà supprimés (OK)
D  tmp/                             → déjà supprimé (OK)
```

### Fichiers non trackés (35 untracked — décision requise)

**→ GARDER et commit :**
```
scripts/attack_winter_holdout.py          ← PIPELINE GAGNANT (975 lignes)
scripts/attack_winter_holdout_fulldata.py ← variante données complètes
scripts/final_pipeline_v17.py             ← étape clé (STL + Fix2 cohérent)
scripts/final_pipeline_v9.py              ← baseline référence
scripts/finals_inference_v2.py            ← pipeline finals (pretrain+inference)
scripts/calculate_real_rmse_all_submissions.py ← utilitaire analyse
notebooks/04_catboost_results.ipynb       ← EDA catboost
notebooks/05_fr_error_diagnostic.ipynb   ← analyse erreurs FR
notebooks/06_basis_modeling.ipynb         ← analyse target UK
notebooks/07_results_recap.ipynb          ← récap résultats
docs/SCORES.md                            ← historique complet scores
docs/FINDINGS.md                          ← findings clés
docs/context.md                           ← contexte marché
FINALS_GUIDE.md                           ← guide finals
configs/hp_cnn_lstm.yaml                  ← (optionnel)
```

**→ SUPPRIMER :**
```
scripts/blend_v13_v9.py          ← blending one-shot, inutile
scripts/blend_v16_v9.py          ← idem
scripts/blend_winter_x_atk_avg.py ← idem
scripts/final_pipeline_v10.py    ← version intermédiaire (1306 lignes)
scripts/final_pipeline_v13.py    ← version intermédiaire
scripts/final_pipeline_v18.py    ← version intermédiaire
scripts/final_pipeline_v19.py    ← version intermédiaire
scripts/final_pipeline_v20.py    ← version intermédiaire
scripts/final_pipeline_v21.py    ← version intermédiaire
scripts/fullretrain_averaged.py  ← remplacé par finals_inference_v2.py
scripts/fullretrain_winter_only.py ← remplacé
attack_verification.log          ← log temporaire
investigate_test_lags.py         ← analyse ponctuelle
STRUCTURE_FINALE.md              ← remplacé par ce plan + README
FINAL_RETRAIN_README.md          ← remplacé par README
docs/SESSION_REPORT_optimization.md ← log de session
docs/V16_VALIDATION_RESULTS.md  ← intégré dans SCORES.md
docs/BENCHMARK_STL_RESULTS.md   ← intégré dans SCORES.md
"docs/antigravity report.md"     ← rapport temporaire
docs/audit/AUDIT_AMELIORATION.md ← log de session
```

**→ GITIGNORE (garder en local, ne pas pousser sur GitHub) :**
```
archive/                         ← contient notebooks_old, scripts_old, outputs_old
                                    → ajouter à .gitignore (déjà untracked, jamais committé)
                                    → propre sur GitHub, accessible en local
                                    ⚠️  ne pas cloner ailleurs sans backup préalable
```

### Outputs à garder (via outputs/.gitkeep seulement — pas commités)
```
outputs/feature_selection_v5_fr.json   ← REQUIS par le pipeline (feat_fr)
outputs/uk_feature_research.json        ← REQUIS par le pipeline (feat_uk)
outputs/submission_attack_averaged.csv  ← soumission gagnante (pour référence)
```
⚠️ Ces fichiers doivent être dans `.gitignore` mais documentés dans le README.

---

## Structure cible pour GitHub

```
incommodities-case-crunch-2026/
├── README.md                          ← À CRÉER (voir plan ci-dessous)
├── CLAUDE.md                          ← garder (instructions projet)
├── config.yaml                        ← hyperparamètres
├── pyproject.toml                     ← dépendances (uv)
├── data/
│   ├── raw/.gitkeep                   ← données non committées
│   └── outputs/.gitkeep
├── docs/
│   ├── Subject.md                     ← énoncé compétition
│   ├── SCORES.md                      ← historique scores
│   ├── FINDINGS.md                    ← findings clés
│   └── context.md                     ← contexte marché électricité
├── notebooks/
│   ├── 01_eda.ipynb                   ← EDA initial
│   ├── 02_eda_deep_dives.ipynb        ← EDA approfondi
│   ├── 03_feature_selection.ipynb     ← sélection features
│   ├── 04_catboost_results.ipynb      ← benchmarks modèles
│   ├── 05_fr_error_diagnostic.ipynb   ← diagnostic erreurs FR
│   ├── 06_basis_modeling.ipynb        ← modélisation target UK
│   └── 07_results_recap.ipynb         ← récap résultats finaux
├── scripts/
│   ├── final_pipeline_v9.py           ← baseline (5 modèles, regime weights, HBC)
│   ├── final_pipeline_v11.py          ← + Stacking Résiduel (SR)
│   ├── final_pipeline_v17.py          ← + STL target FR + Fix2 cohérent
│   ├── attack_winter_holdout.py       ← SOLUTION GAGNANTE (double calibration)
│   ├── attack_winter_holdout_fulldata.py ← variante données complètes (finals)
│   ├── finals_inference_v2.py         ← pipeline finals (pretrain + inference)
│   └── calculate_real_rmse_all_submissions.py
├── src/
│   ├── __init__.py
│   ├── data_loading.py
│   ├── feature_engineering.py
│   └── models/
│       ├── __init__.py
│       ├── metrics.py
│       ├── targets.py
│       ├── ensemble.py
│       ├── tree_models.py
│       ├── elastic_net.py
│       ├── dnn.py
│       ├── gat.py                     ← (optionnel, jamais utilisé en prod)
│       └── cnn_lstm.py                ← (optionnel, jamais utilisé en prod)
└── outputs/
    ├── .gitignore ou .gitkeep
    ├── feature_selection_v5_fr.json   ← requis pipeline (à documenter)
    └── uk_feature_research.json       ← requis pipeline (à documenter)
```

---

## Plan README.md

Structure suggérée :

```markdown
# InCommodities Case Crunch 2026 — 2nd Place Solution

**Competition**: Forecast hourly day-ahead electricity spot prices (FR + UK)
**Metric**: RMSE(FR) + RMSE(UK) averaged | **Result**: 2nd/68, score 20.0781

## Solution Overview
[Architecture diagram / bullet points de la solution]

## Key Innovations
1. STL decomposition as FR target (−1.84 RMSE standalone)
2. Stacking Résiduel with 39-44 thematic sub-models (−0.76 vs v9)
3. Double seasonal calibration: spring + winter holdouts averaged (−2.74 vs baseline)

## Setup
uv sync && ...

## Reproduce
# Pretrain (10 min)
python scripts/attack_winter_holdout.py

## Pipeline Evolution
v3→v9→v11→v16/v17→attack_winter (avec scores à chaque étape)

## Files
[table des fichiers clés]
```

---

## Checklist de nettoyage (ordre recommandé)

- [ ] 1. Commit les fichiers M trackés (src/, config.yaml, etc.)
- [ ] 2. Supprimer les scripts intermédiaires listés ci-dessus
- [ ] 3. Ajouter `archive/` à `.gitignore` (garder en local, invisible sur GitHub)
- [ ] 4. Nettoyer `outputs/` (garder seulement les 2 JSON requis + submission gagnante)
- [ ] 5. Ajouter `outputs/` à `.gitignore` (sauf les 2 JSON requis)
- [ ] 6. Créer `README.md`
- [ ] 7. Vérifier que `src/models/cnn_lstm.py` et `gat.py` méritent d'être gardés (jamais produits)
- [ ] 8. Squash ou rebase les commits pour une histoire propre (optionnel)
- [ ] 9. Supprimer la branche `v2`, tout committer sur `main`
- [ ] 10. Push sur GitHub (repo public ou privé ?)

---

## Points d'attention

**outputs/ requis par le pipeline** : `feature_selection_v5_fr.json` et
`uk_feature_research.json` sont chargés dans `attack_winter_holdout.py` (lignes 593-610).
Sans ces fichiers, le pipeline ne tourne pas. Options :
- Les committer dans `outputs/` (petits fichiers JSON, ~KB)
- Intégrer les feature lists directement dans le script (plus autonome)

**Branch v2** : tout le développement récent est sur `v2`, pas `main`.
Merger dans `main` avant de publier.

**Données** : `.gitignore` exclut déjà `data/raw/*`. OK.

**`.venv/`** : bien dans `.gitignore`. OK.

**`uv.lock`** : actuellement dans `.gitignore` — pour reproducibilité GitHub,
envisager de le committer (débat habituel lock file).
