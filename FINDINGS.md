# INCOMO 3 — Findings & Decisions

---

## Decisions Cles

### 1. Target Engineering

**FR : `y = spot - EMA(spot_la, 240h)`**
- EMA 240h bat rolling mean 168h de -0.37 RMSE
- EMA s'adapte aux changements de regime (crise gaz 2022 → normalisation 2024)
- L'EMA est calculee sur le spot lagged (pas le spot actuel) → pas de leakage

**UK : `y = spot - merit_order_cost`**
- merit_order_cost = gas/efficiency + emission × 0.37
- Bat EMA 240h de -0.68 RMSE (enorme)
- Raison : UK est un marche gas-dominated, le MOC capture ~70% du signal de prix
- Le residual (spot - MOC) est plus stationnaire que le spot brut

### 2. Profondeur des arbres

**FR : depth=3** (peu profond)
- 83% d'importance dans l'optimisation Optuna (300 trials)
- FR a peu de features (28) → profondeur faible evite l'overfitting
- lr=0.059 (rapide), l2_reg=4.4 (leger)

**UK : depth=8** (profond)
- UK a 150+ features avec beaucoup d'interactions
- Le basis target (spot - MOC) est plus simple → le modele peut aller plus profond
- MAE loss au lieu de RMSE (robuste aux tails lourds UK : range [-205, +1444])

### 3. Ensemble 5 modeles + Regime Weights

- **CatBoost** : meilleur standalone
- **LightGBM** : diversite (correlation erreur 0.75 avec CB)
- **XGBoost** : correlation encore plus basse (0.70 avec CB FR)
- **Elastic Net** : modele lineaire, correlation 0.60-0.70 avec trees
- **DNN** : Huber loss, 349 features, correlation 0.70-0.80 avec trees

Les poids varient par heure :
- **Matin (6-9h)** : EN+DNN dominent (CB a 0%) — trees ratent les ramps matinaux
- **Nuit** : LGB domine FR, CB domine UK
- **Peak** : EN+DNN pour FR, CB+LGB pour UK

### 4. HBC (Hourly Bias Correction)

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
| Stacking (Ridge) | +2.1 FR | Overfit severe sur le meta-learner |
| arcsinh transform | +0.3-0.5 | Degrade systematiquement tous les modeles |
| Basis modeling FR (spot - MOC) | +4.5 FR | basis_shift = +40 EUR (crise gaz change le MOC) |
| Multi-window averaging | -0.05 SUM | Gain trop faible, full window suffit |
| Huber loss (CatBoost) | +0.10 FR | RMSE standard est optimal pour le metric RMSE |
| Per-hour features (v8 dans pipeline) | +0.03 SUM | Absorbe par le regime ensemble |

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

## Pistes Non Explorees

1. **Conformalized predictions** : intervalles de prediction pour le scoring Winkler (si le metric change)
2. **Online learning** : reentrainer incrementalement sur le test set (si autorise)
3. **Cross-validation temporelle stricte** : expanding window avec re-optimisation des poids
4. **Ensemble de submissions** : blender v3a + v5b + v7 + v8 (diversite maximale)
5. **Feature engineering deep** : interactions automatiques (PolynomialFeatures + selection)
6. **Transformer models** : attention-based pour les sequences temporelles
7. **Quantile regression ensemble** : predire la mediane au lieu de la moyenne
8. **External data** : weather forecasts, electricity futures, capacity auctions
