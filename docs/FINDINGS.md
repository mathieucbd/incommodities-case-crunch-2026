# INCOMO 3 — Findings & Decisions

---

## Decisions Cles

### 1. Target Engineering

**FR : `y = spot - STL_trend(spot_la, 168h)`** ✅ v16 (remplace EMA 240h)
- **STL** (Seasonal-Trend decomposition using Loess) bat EMA 240h de **-1.84 RMSE +HBC**
- Period 168h = 1 semaine, seasonal=13 (odd integer > period/2)
- Décompose spot_la = Trend + Seasonal + Residual
- **Avantages vs EMA** :
  - Capture mieux les patterns hebdomadaires (weekend vs weekday)
  - Décomposition adaptative (Loess) vs EMA rigide
  - Trend + Seasonal séparés → prédiction plus précise
- **Benchmark standalone** (CatBoost Quantile:0.6, holdout Feb-Jun 2024):
  - EMA 240h: 17.92 RMSE +HBC
  - STL trend: 16.08 RMSE +HBC
  - Delta: **-1.84** (>10% réduction d'erreur)
- **Gain attendu v16** (composition 60%): ~-1.10 → score ≈ 22.60

**FR ancien (v9-v13) : `y = spot - EMA(spot_la, 240h)`**
- EMA 240h bat rolling mean 168h de -0.37 RMSE
- EMA s'adapte aux changements de regime (crise gaz 2022 → normalisation 2024)
- L'EMA est calculee sur le spot lagged (pas le spot actuel) → pas de leakage

**UK : `y = spot - merit_order_cost`** ✅ Optimal (STL testé et rejeté)
- merit_order_cost = gas/efficiency + emission × 0.37
- Bat EMA 240h de -0.68 RMSE (enorme)
- **Bat STL trend de +0.49** → MOC reste meilleur pour UK
- Raison : UK est un marche gas-dominated, le MOC capture ~70% du signal de prix
- Le residual (spot - MOC) est plus stationnaire que le spot brut
- **STL échoue pour UK** car MOC = driver fondamental (causalité), STL = pattern temporel

### 2. Profondeur des arbres

**FR : depth=3** (peu profond)
- 83% d'importance dans l'optimisation Optuna (300 trials)
- FR a peu de features (28) → profondeur faible evite l'overfitting
- lr=0.059 (rapide), l2_reg=4.4 (leger)

**UK : depth=8** (profond)
- UK a 150+ features avec beaucoup d'interactions
- Le basis target (spot - MOC) est plus simple → le modele peut aller plus profond
- MAE loss au lieu de RMSE (robuste aux tails lourds UK : range [-205, +1444])

### 3. Ensemble 5 modeles + Regime Weights + Loss Diversity (v9)

- **CatBoost** : Quantile:0.6 (FR+UK) — meilleur standalone, biais asymetrique
- **LightGBM** : MAE (FR) / Huber_5 (UK) — robuste aux outliers
- **XGBoost** : PseudoHuber_20 (FR+UK) — compromis MSE/MAE
- **Elastic Net** : MSE (fixe) — modele lineaire, correlation 0.60-0.70 avec trees
- **DNN** : Huber_5 (FR) / MSE (UK) — 356 features, correlation 0.70-0.80 avec trees

Chaque modele utilise une loss differente → erreurs decorrelees → meilleur ensemble.
FR sample weights supprimees (ratio 23,895x tuait les loss non-MSE).

Les poids varient par heure :
- **Matin (6-9h)** : EN+DNN dominent (CB a 0.2) — trees ratent les ramps matinaux
- **Nuit** : CB+DNN dominent FR, CB+XGB+EN UK
- **Peak** : EN+DNN pour FR, CB+LGB pour UK

### 4. Stacking Residuel T2 (v11)

**Architecture** : v9 ensemble + Ridge(fondamentales) + Stacking Residuel T2
- T2 = modeles legers (Ridge, ElasticNet, LGB small, XGB small) entraines par groupes thematiques
- Le meta-learner Ridge predit l'**erreur du v9 ensemble** (pas le spot directement)
- Le mode residuel bat le mode independant de -0.19 SUM

**Decomposition des gains (v9 → v11)** :
```
v9 baseline                  25.10
  + Ridge(fondamentales)     24.85  (-0.25)
  + Stacking T2              24.60  (-0.25)
  + mode Residuel            24.41  (-0.19)
  + Per-country optim        24.19  (-0.22) FR -cb_small, UK -lgb_small, enriched meta UK
  + Group splits + combos    23.85  (-0.34)
  TOTAL                      -1.25
```

**Per-country optimisations** :
- FR : alpha=1, 4 algos (sans cb_small), 9 groupes (fr_load split en raw+residual)
- UK : alpha=500 (anti-overfitting), 4 algos (sans lgb_small), 9 groupes (uk_wind split en core+continent), meta-learner enrichi (+hour/dow sin/cos)

**Anti-overfitting (v11 → v11b)** :
- v11 aggressive (7+5 combos) : val=23.83, Kaggle=24.23 → gap +0.40
- v11b conservative (3+3 combos, UK alpha 100→500) : val=23.85, Kaggle=23.92 → gap +0.07
- Reduire le nombre de combos et augmenter la regularisation UK a divise le gap par 6

### 5. Segmentation Temporelle UK (v13)

**Principe** : Entrainer les memes algos sur des sous-ensembles d'heures pour creer des predictions decorrelees.

**Split** : Shifted 6h → [3-8], [9-14], [15-20], [21-2] (~3480 samples/cluster)

**Screening (H2)** : Seul XGB beneficie de la segmentation (err corr=0.838, RMSE<105% global).
CB (0.915), LGB (0.902), EN (0.915) → trop correles avec le global.

**Integration gagnante (H3)** : XGB_cluster comme 8e membre de l'ensemble regime UK.
- 7 models → 8 models (CB+LGB+XGB+EN+DNN+RidgeF+SR+XGB_cluster)
- UK : 8.99 → 8.87 (-0.12)
- SUM : 23.85 → 23.73 (-0.12)

**Rejete** :
- H4 (cluster preds comme features SR) : +0.02 — le Ridge alpha=500 dilue le signal
- H5 (pre-blend w=0.5) : -0.08 — inferieur a H3
- H6 (24 hourly LGB comme feature SR) : -0.05 — marginal

**UK seulement** — FR a des correlations >0.95 entre strategies, la segmentation n'aide pas.

### 6. HBC (Hourly Bias Correction)

- Corrige le biais systematique par heure (ex: surestimer h=18, sous-estimer h=7)
- Gain moyen : 0.2-0.5 RMSE
- 3 variantes testees : standard (24 params), monthly (120 params), dampened
- Monthly HBC gagne ~0.8 RMSE en val mais risque d'overfitting au test

---

## Ce Qui a Marche

| Amelioration | Delta RMSE | Impact |
|-------------|-----------|--------|
| Basis target UK (spot - MOC) | -0.68 UK | majeur |
| EMA 240h FR (vs rolling 168h) | -0.37 FR | significatif |
| DNN 5e modele (v7) | -0.77 SUM | majeur |
| Elastic Net 4e modele (v6) | -0.50 SUM | significatif |
| XGBoost 3e modele (v5) | -0.48 SUM | significatif |
| Regime weights (v5b) | -0.13 FR | modere |
| Optuna v2 FR (depth=3, lr=0.059) | -0.86 FR | significatif |
| MAE loss UK | -0.06 UK | modere |
| Loss diversity + no FR weights (v9) | -0.18 SUM | significatif |
| Stacking Residuel T2 (v11) | -1.25 SUM | majeur |
| Anti-overfitting combos (v11b) | gap /6 | majeur (0.40→0.07) |
| **STL target FR (v16)** | **-0.65 val, -0.68 Kaggle** | **majeur** (23.94→23.26) |
| **Fix 2 STL cohérent retrain (v17)** | **-0.04 Kaggle** | modéré (23.24→23.20 via blend) |
| **Blend 85% v17 + 15% v9** | **-0.06 Kaggle** | **meilleur score** (23.26→23.20) |
| Blend 85% v13 + 15% v9 | -0.14 Kaggle | majeur (23.92→23.78, gap 0.64) |
| XGB cluster 8e membre UK (v13) | -0.12 val, **+0.02 Kaggle** | overfit (gap 0.21 vs 0.07) |
| UK 12m window (CatBoost seul) | -0.43 UK | significatif |
| Nouvelles features (rolling_336h etc.) | -0.40 SUM | significatif (CB seul) |
| Per-hour models | -0.30 SUM | modere (CB+LGB seul) |

**Note :** UK 12m, nouvelles features, et per-hour models montrent des gains en isolation (CB/LGB) mais **ne composent pas** dans le pipeline 5-model avec regime weights. Le regime ensemble absorbe deja ces gains.

---

## Ce Qui N'a Pas Marche

| Tentative | Resultat | Raison |
|-----------|----------|--------|
| GRU (12 configs) | +0.8 FR, +0.4 UK vs CB | Trees > sequences sur features tabulaires |
| GAT-GRU (9 configs) | +1.1 FR, +0.8 UK vs CB | Graph attention n'aide pas ici |
| GRU comme 6e modele | 0.00 SUM | Correlation erreur 0.88 avec DNN (pas de diversite) |
| Stacking independant (Ridge, v11 H2) | +0.2 vs residuel | Mode residuel systematiquement meilleur |
| arcsinh transform | +0.3-0.5 | Degrade systematiquement tous les modeles |
| Monthly x Hour HBC (V11) | 0.00 OOF | Overfit massif in-sample (-0.60), inutile OOF. Le T2 absorbe déjà le biais. |
| Features "Same-Hour Lookback" | +0.27 à +0.73 | Les modèles de grande profondeur (UK depth 8) captent déjà nativement l'interaction heure×lag. |
| k-NN Analog Days (Mémoire physique) | +0.02 (neutre) | Retrouver le prix des jours physiquement similaires (L1, K=3) n'ajoute aucune info exploitable au-delà de la V11. |
| Custom Asymmetric Loss (Pics/Creux) | +0.40 à +3.00 | Essayer de "forcer" le modèle (via $alpha \cdot \text{Huber}$) à chasser les pics détruit la baseline continue. Le GBDT est déjà optimal en loss symétrique. |
| IVW sample weights (v14, 4 formules) | -0.02 SUM pipeline | CB FR -0.18 standalone mais CB UK +0.41. XGB degrade (+0.12). Seul LGB gagne (-0.03). Le T2 stacking absorbe les gains. |
| T1 Features dans pipeline complet (v13+T1) | val -0.02, test +0.02 | Gain isolé -0.29 +HBC sur ensemble 5 modèles, mais dans pipeline v13 complet les hyperparamètres optimisés captent déjà le signal via features corrélées. Cause overfitting (gap +0.23 vs +0.21). |
| Cold-Start Fix (v14, concat train+test) | test +0.58 (CATASTROPHE) | build_features(concat(train, test)) pour historique complet → DATA LEAKAGE massif. Rolling/shift features utilisent données futures du test. Val 23.69 semblait bon mais test 24.27 révèle le désastre. |
| Ridge Damping (v15, alpha FR 1→10, UK 500→1000) | val +0.01 (inefficace) | Objectif: réduire dominance SR (poids 0.4-0.8). Résultat: SR reste dominant, val 23.74 vs 23.73. L'augmentation alpha dilue le signal sans réduire l'overfitting. |

---

## Les Révélations Structurelles (Edge)

### 1. Inverse Variance Weighting — Debunked (v14)

**Decouverte initiale (Holdout + LOMO 4 folds)** : Le benchmark `benchmark_v12_weighting.py` montrait des gains spectaculaires (-2.51 FR XGB, -0.35 UK XGB). MAIS le benchmark etait **defectueux** :
- Baselines STD jamais entrainees (comparaison contre des zeros)
- Seulement 4 mois, 1000 iter (v11b=15000), ES=30 (v11b=200)

**Test rigoureux (v14, 5 hypotheses)** : En utilisant les params v11b exacts (15000 iter, ES=200) sur le holdout standard :

| Algo | FR delta | UK delta | Verdict |
|------|----------|----------|---------|
| CB (Quantile:0.6) | **-0.18** | **+0.41** | FR OK, UK detruit |
| LGB (MAE/Huber) | -0.03 | -0.03 | Marginal |
| XGB (PseudoHuber) | +0.12 | +0.11 | Degrade (contrairement au benchmark!) |

**Pourquoi CB UK est detruit** : Quantile:0.6 a deja un biais asymetrique. Ajouter IVW (qui downweight les extremes) cree un double biais → le modele early-stoppe a 305 iter au lieu de 698.

**Pipeline complete (H4)** : SUM 23.85 → **23.83** (delta=-0.02). Le T2 stacking absorbe presque tout le gain standalone.

**4 formules testees** : F1_base (1/(1+vol/mean)), F2_clip, F3_exp, F4_rank. F1_base est la meilleure mais toutes sont marginales dans le pipeline complet.

**Conclusion** : IVW ne compose pas avec le pipeline v11b. Les gains standalone (~-0.18 CB FR) sont absorbes par le regime ensemble + T2 stacking.
| Basis modeling FR (spot - MOC) | +4.5 FR | basis_shift = +40 EUR (crise gaz change le MOC) |
| Multi-window averaging | -0.05 SUM | Gain trop faible, full window suffit |
| Huber loss (CatBoost seul) | +0.10 FR | RMSE standard est optimal pour le metric RMSE |
| Meme loss partout (Quant:0.6) | +0.09 SUM | Modeles trop correles → ensemble perd en diversite |
| Per-hour features (v8 dans pipeline) | +0.03 SUM | Absorbe par le regime ensemble |
| Regime simplification (3 regimes) | +0.03 SUM | 5 regimes step=0.1 est optimal |
| Regime step=0.2 | +0.08 SUM | Granularite plus fine toujours meilleure |
| UK alpha sweep (1→1000) | ±0.01 SUM | Alpha=300 best (-0.001), pas significatif |
| UK greedy 8 combos (val only) | -0.19 SUM | Fort gain val mais risque d'overfitting test |
| Cluster preds comme features SR (v13 H4) | +0.02 SUM | Ridge alpha=500 dilue le signal cluster |
| Pre-blend global+cluster (v13 H5) | -0.08 SUM | Inferieur a H3 (ajout direct comme 8e membre) |
| 24 hourly LGB comme feature SR (v13 H6) | -0.05 SUM | Marginal, RMSE standalone 22.67 (2x pire) |
| CB/LGB/EN cluster models (v13 H2) | neutre | Corr erreur >0.90 avec global → pas de diversite |

---

## Insights Domaine

### Marche FR (SDAC / EPEX SPOT)
- **Nuclear-dominated** : 70% de la generation, prix souvent fixe par le nucleaire
- **EMA comme anchor** : le prix FR suit un trend lent (EMA 240h = 10 jours)
- **River temperature** : >25C = risque de curtailment nucleaire (feature `fr_river_temp_risk`)
- **Continental coupling** : le prix FR est plafonne par les imports voisins (DE, BE, CH, ES)
- **28 features suffisent** : feature selection agressive (de 330 → 28) ameliore la stabilite

### Marche UK (N2EX / Nord Pool)
- **Gas-dominated** : merit order cost = gas/0.50 + emission × 0.37
- **Heavy tails** : range [-205, +1444], MAE loss plus robuste
- **6 interconnectors** : IFA1/2, ElecLink, NEMO, BritNed, Viking Link
- **150 features necessaires** : plus de features que FR car plus de sources de variabilite
- **12m window optimal** : le regime post-crise (2023+) est plus representatif du test

### Dynamiques Cross-Market
- `euro_scarcity_ratio` : top feature pour les deux marches
- `continental_residual_load` : demande nette europeenne = driver de prix
- Les z-scores 14 jours (`fr_residual_zscore_14d`) capturent les anomalies

---

## Lecons Anti-Overfitting

1. **Plus de combos ≠ mieux** : 7+5 combos greedy val=-0.40, mais Kaggle gap +0.40. 3+3 combos conservative : gap +0.07
2. **Alpha eleve = securite** : UK alpha=500 (vs 100) coute -0.004 val mais protege le test
3. **Les 4 leviers testes (v12)** n'apportent rien de plus :
   - Regime simplification → pire
   - Alpha sweep → marginal (±0.01)
   - FR combos → deja optimal
   - UK greedy combos → -0.19 val mais risque d'overfitting identique a v11 aggressive
4. **Le gap val→Kaggle est LE metric de qualite**, pas le score val absolu
5. **Ajouter des membres d'ensemble = plus de combos = plus d'overfitting** : v13 ajoute XGB_cluster comme 8e membre UK → 8 models × 5 regimes × grid search step=0.1. Le gain val (-0.12) ne transfere pas (Kaggle +0.02, gap 0.07→0.21)
6. **Les gains standalone ne composent pas** : IVW donne -0.18 CB FR standalone, mais -0.02 SUM dans le pipeline complet. Le T2 stacking + regime ensemble absorbent deja ce que l'IVW apporterait. De même, T1 features donnent -0.29 +HBC isolées mais seulement -0.02 dans v13 complet.
7. **Les benchmarks defectueux trompent** : Le benchmark LOMO IVW montrait -2.51 FR XGB car les baselines STD n'etaient jamais entrainees (zeros). Toujours valider les baselines
8. **Blending models réduit l'overfitting** : Mélanger v13 (overfit gap +0.21) avec v9 (plus stable, moins complexe) dans ratio 85/15 réduit le gap et améliore le test de -0.14 (23.92→23.78). Le modèle simple tempère les excès du modèle complexe.
9. **Data leakage via concat est catastrophique** : build_features(concat(train, test)) pour "cold-start fix" semblait logique (val -0.04) mais détruit le test (+0.58). Les rolling/shift features voient le futur. JAMAIS concat avant feature engineering.
10. **STL target engineering = gros gain FR, inefficace UK** : STL (décomposition saisonnière 168h) bat EMA 240h de -1.84 standalone FR, mais perd face à Merit Order Cost pour UK (+0.49). Les patterns temporels (STL) fonctionnent quand ils capturent la structure réelle des prix (cycles hebdo FR). Les anchors fondamentaux (MOC UK) gagnent quand ils reflètent la causalité économique (gas+carbon). Benchmark avant intégration = crucial pour éviter les fausses pistes.
11. **Le gap STL = biais in-sample, pas overfitting modèle** : v16 gap +0.18 (val 23.08 → test 23.26). Fix 1 (STL walk-forward, fit sur train-only) dégrade la val de exactement +0.18 → le gap est 100% dû à STL bidirectionnel qui utilise des données futures en validation. Le modèle lui-même n'overfit PAS. Implication : on ne peut pas réduire le gap en régularisant le modèle.
12. **Fix 2 STL cohérent = petit gain test réel** : Utiliser un seul fit STL sur concat(train,test) pour le retrain (au lieu de mixer 2 fits différents) donne -0.04 sur Kaggle (23.24→23.20 via blend). Petit mais gratuit.
13. **Le blending v16/v17+v9 est moins efficace qu'avec v13** : blend v13+v9 85/15 améliorait de -0.14. blend v16+v9 85/15 n'améliore que de -0.02. v16 est déjà trop bon → v9 (val 25.02) dilue plus qu'il n'aide. Le blending a des rendements décroissants quand le modèle principal s'améliore.
14. **Dual holdout recalibration > validation single-season** : Tester sur 2 holdouts saisonniers (spring Feb-Jun 2024 + winter Jul-Nov 2023) révèle que les poids de régime divergent radicalement. DNN disparaît en winter (0.0-0.2), SR ultra-dominant (0.7-0.8). Submission attack_averaged (moyenne arithmétique des 2 configs) gagne -2.74 Kaggle vs meilleur blend précédent → 2e place (20.0781). Leçon : la validation sur une seule saison est biaisée, moyenner plusieurs holdouts saisonniers = meilleure généralisation.
15. **Stacking Residual meta-learning > early stopping** : Le SR (Ridge meta-learner) apprend les patterns d'erreurs des base learners et les corrige dynamiquement. Il récupère -0.19 RMSE (H2→H3) même si l'early stopping est précoce (CB_FR 147 iter, LGB_FR 731). Le meta-learning residuel capture ce que les base learners ratent → surpasse la pure régularisation via early stopping.
16. **Fulldata révèle non-stationnarité UK Q4 2024** : Nouveau dataset 2022→2025 (reçu 26/03/2026) expose UK RMSE explosion 8.83 → 33.51 sur holdout Oct-Dec 2024. Le vrai test Jul 2024-Feb 2025 donne UK=27-29 (hors-distribution mais moins extrême). Train 2022-2024 ne contient pas ces régimes de prix. Leçon : l'électricité est non-stationnaire, les shifts de distribution majeurs post-training period sont possibles.

---

## Pistes Non Explorees

1. **Conformalized predictions** : intervalles de prediction pour le scoring Winkler (si le metric change)
2. **Online learning** : reentrainer incrementalement sur le test set (si autorise)
3. **Cross-validation temporelle stricte** : expanding window avec re-optimisation des poids
4. **Blend de submissions** : v11b + v7 + v9 (correlation test ~0.96-0.99, diversite limitee)
5. **Feature engineering deep** : interactions automatiques (PolynomialFeatures + selection)
6. **Transformer models** : attention-based pour les sequences temporelles
7. **Quantile regression ensemble** : predire la mediane au lieu de la moyenne
8. **External data** : weather forecasts, electricity futures, capacity auctions
