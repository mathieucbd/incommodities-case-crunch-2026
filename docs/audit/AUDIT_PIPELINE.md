# Audit Complet — Pipeline v7/v8

Date : 2026-03-24
Scope : final_pipeline.py (v7), final_pipeline_v8.py, feature_engineering.py, data_loading.py, config.yaml

---

## Resume Executif

| Severite | Nb | Description |
|----------|-----|-------------|
| CRITIQUE | 1 | Cold-start features au test (build_features separe) |
| HAUTE | 2 | XGBoost FR casse, regime weights overfitting |
| MOYENNE | 4 | DNN retrain, Monthly HBC, HBC saisonnier, EN sans poids |
| BASSE | 2 | Scaler DNN partage (v7), feature lists stale |
| OK | 3 | Pas de leakage, clipping raisonnable, EMA target correct |

**Impact estime sur RMSE : 0.3 - 0.8 points potentiellement recuperables.**

---

## 1. CRITIQUE — Cold-Start Features au Test

### Probleme

`build_features()` est appele **separement** sur train et test :

```python
train_fe = build_features(train, config)   # rolling features = historique train seulement
test_fe = build_features(x_test, config)   # rolling features = REPART DE ZERO
```

Consequence : toutes les features rolling/shift dans `test_fe` n'ont **aucun historique** du training set. Au debut du test :

| Feature | Window | Lignes affectees | % du test |
|---------|--------|-----------------|-----------|
| `*_roll_24h_*` | 24h | 23 rows degradees | 0.4% |
| `*_lag_48h` | shift(24) | 24 rows = NaN | 0.4% |
| `*_roll_168h_*` | 168h | 167 rows degradees | 2.9% |
| `*_zscore_14d` | 336h | 335 rows degradees | 5.7% |
| `*_roll_336h_*` | 336h | 335 rows degradees | 5.7% |
| `*_lag_168h` | shift(144) | 144 rows = NaN | 2.5% |
| EWM features | span=24 | ~72h pour converger | 1.2% |

Les features degradees ne sont pas juste "moins precises" — elles sont **fondamentalement differentes** de ce que le modele a vu en training. Un `fr_spot_la_roll_168h_mean` calcule sur 10 heures de donnees au lieu de 168h est un signal completement different.

### Impact estime

Les 2 premieres semaines (~336h = 5.7% du test) ont des features degradees. Si le RMSE y est 30-50% plus eleve :

```
RMSE_degraded = RMSE_normal * 1.3 (estimation conservative)
Impact total = sqrt(0.057 * (1.3)^2 + 0.943 * 1^2) - 1 ≈ +2.5%
Pour SUM=25 : delta ≈ +0.6 RMSE
```

### Fix

Concatener avant build_features, splitter apres :

```python
# Concatener SANS les targets (pas de leakage)
all_x = pd.concat([x_train, x_test], axis=0)
all_fe = build_features(all_x, config)

# Splitter et ajouter les targets au train seulement
n_train = len(x_train)
train_fe = all_fe.iloc[:n_train].copy()
train_fe = train_fe.join(y_train[["fr_spot", "uk_spot"]])
test_fe = all_fe.iloc[n_train:].copy()
```

Verification : `build_features()` n'utilise **jamais** `fr_spot` ou `uk_spot` directement — toutes les features sont construites a partir de `*_la` (lagged) et `*_f` (forecast). Donc pas de risque de leakage.

### Fichiers concernes

- `scripts/final_pipeline.py` : lignes 139-143
- `scripts/final_pipeline_v8.py` : lignes 139-143
- La section retrain (lignes 773+) recalcule correctement l'EMA sur `all_data`, mais les features de `df_test_pred` viennent toujours de `test_fe` separe

---

## 2. HAUTE — XGBoost FR Casse

### Probleme

```
XGB FR: RMSE=28.34, +HBC=24.88, iter=15000
```

XGBoost FR **atteint le cap de 15000 iterations** sans early stopping. Son RMSE brut (28.34) est 69% pire que CatBoost (16.79). Le HBC compense partiellement (24.88), mais c'est quand meme terrible.

Resultat : XGBoost recoit **poids = 0.0 dans TOUS les regimes FR**. C'est du code mort — il est entraine pour rien.

### Diagnostic

| Symptome | Cause probable |
|----------|---------------|
| Pas d'early stopping | LR trop haute (0.05) avec regularisation forte (alpha=5, lambda=10) → le modele n'arrive pas a apprendre assez vite pour trigger early stopping |
| RMSE brut 28.34 | Le modele underfit massivement |
| HBC corrige a 24.88 | Le modele a un biais systematique enorme (+11.5 EUR/h en moyenne) |

### Options

1. **Supprimer XGBoost FR** — gagne ~30s de temps d'entrainement, aucun impact sur les predictions
2. **Re-tuner XGBoost FR** — lr=0.01, alpha=0.5, depth=6, n_estimators=30000
3. **Garder mais ne pas retrainer** si poids = 0 dans tous les regimes (deja fait au retrain via `fr_xgb_needs`)

### Impact

Nul actuellement (poids=0), mais si XGBoost FR etait correctement tune, il pourrait ajouter de la diversite et ameliorer l'ensemble de 0.1-0.3 RMSE.

---

## 3. HAUTE — Regime Weights Overfitting

### Probleme

25 parametres (5 regimes x 5 modeles) optimises par grid search exhaustif sur 3623 observations de validation :

| Regime | Heures | N observations | Combinaisons testees |
|--------|--------|---------------|---------------------|
| Night | 0-5 | 905 | 14641 |
| Morning | 6-9 | 604 | 14641 |
| Day | 10-16 | 1057 | 14641 |
| Peak | 17-21 | 755 | 14641 |
| **Late** | **22-23** | **302** | **14641** |

Le regime "Late" n'a que **302 observations** pour choisir parmi 14641 combinaisons de poids. Avec un ratio obs/combinaisons de 0.02, le risque d'overfitting est enorme.

### Exemples suspects

```
FR late: CB=0.5, LGB=0.1, XGB=0.0, EN=0.0, DNN=0.4  → n=302
UK late: CB=1.0, LGB=0.0, XGB=0.0, EN=0.0, DNN=0.0  → n=302
```

UK late donne 100% au CatBoost. Avec seulement 302 observations, est-ce robuste ou juste du bruit ?

### Comparaison val vs test (v5b)

La seule soumission Kaggle avec regime weights (v5b) a montre :
- Val SUM : 26.39
- Kaggle SUM : 24.88

Le Kaggle est **meilleur** que la val, ce qui suggere que l'overfitting des regime weights n'est pas catastrophique. Mais la difference pourrait venir d'autres facteurs (distribution du test plus facile).

### Fix propose

1. **Grid plus grossier** : step=0.2 au lieu de 0.1 → 6^4 = 1296 combinaisons (11x moins)
2. **Shrinkage** : melanger les poids optimaux avec l'uniforme (1/5)
   ```python
   shrinkage = 0.7  # 70% optimal, 30% uniforme
   w_final = shrinkage * w_optimal + (1 - shrinkage) * (1/5)
   ```
3. **Cross-validation temporelle** : 3 folds expanding window pour robustifier
4. **Supprimer le regime "Late"** : fusionner h22-23 avec "Night" (meme profil de prix)

---

## 4. MOYENNE — DNN Retrain Sans Validation Propre

### Probleme

Lors du retrain sur full data, le DNN utilise une partie des donnees d'entrainement comme "validation" :

```python
# Ligne 885-888 (v7)
dnn_fr_final, _ = train_dnn(
    dnn_fr_final, X_dnn_full,
    y_dev_fr_full[valid_fr_full],
    X_dnn_full[:256],                              # <-- TRAIN DATA comme validation!
    y_dev_fr_full[valid_fr_full][:256],             # <-- TRAIN DATA comme validation!
    max_epochs=dnn_fr_epochs + 5,
    patience=dnn_fr_epochs + 5)                     # <-- patience = max_epochs → pas d'early stopping
```

Le `patience` egal a `max_epochs` garantit que l'early stopping ne trigger jamais. Le modele s'entraine pour exactement `dnn_fr_epochs + 5` epochs.

### Risque

- Le nombre d'epochs est approximativement correct (base sur la validation), mais :
  - Le full training set est ~4x plus grand → chaque epoch voit plus de donnees
  - Le nombre optimal d'epochs sur plus de donnees est potentiellement different
  - Pas de garde-fou contre l'overfitting

### Fix

- Utiliser la derniere portion du training set (ex: 10% dernieres heures) comme holdout interne pour l'early stopping du DNN retraine
- Ou : fixer le nombre d'epochs a `int(dnn_fr_epochs * 0.8)` (80% des epochs de validation, car plus de donnees = convergence plus rapide par epoch)

---

## 5. MOYENNE — Monthly HBC Overfitting

### Probleme

| Methode HBC | Params | Obs/param | Val SUM | Risque |
|-------------|--------|-----------|---------|--------|
| Standard | 24 | ~151 | 25.12 | Faible |
| Monthly x Hour | 120 | ~30 | **24.36** | **Eleve** |
| Dampened (0.7) | 120 | ~30 | 24.45 | Moyen |

Le Monthly HBC gagne **0.76 RMSE** en validation (25.12 → 24.36), ce qui est enorme. Mais avec seulement ~30 observations par cellule (mois, heure), c'est probablement de l'overfitting.

### Pourquoi c'est risque

La validation couvre Fev-Juin 2024 (5 mois). Le test couvre Jul 2024-Fev 2025 (8 mois). Les mois de test incluent :
- Jul, Aug, Sep, Oct, Nov, Dec, Jan, Feb
- Seul **Feb** est commun entre val et test

Donc les corrections Monthly HBC pour Jul-Jan sont basees sur des mois absents du test. Elles ne generaliseront pas.

### Fix

Si on veut soumettre avec Monthly HBC :
- Dampened alpha=0.5 au lieu de 1.0 (reduire l'amplitude des corrections)
- Utiliser le standard HBC (24 params) comme fallback pour les mois non vus
- Ou : calculer l'HBC par mois en utilisant les **memes mois des annees precedentes** du training set (seasonality-aware HBC)

---

## 6. MOYENNE — HBC Mismatch Saisonnier

### Probleme

Les corrections HBC sont calibrees sur **Fev-Juin 2024** mais appliquees a **Jul 2024 - Fev 2025**.

Le biais systematique du modele par heure **depend de la saison** :
- **Ete** (Jul-Sep) : solaire fort, prix bas → le modele tend a surestimer
- **Hiver** (Dec-Feb) : chauffage, nuits longues → le modele tend a sous-estimer les peaks

Les corrections HBC calibrees sur le printemps (Mars-Juin) peuvent etre incorrectes pour l'ete et l'hiver.

### Exemple concret

```
FR HBC h=7 : +3.69  (corrige la sous-estimation du morning ramp)
FR HBC h=18: -3.12  (corrige la surestimation du evening peak)
```

Ces corrections de ~3 EUR sont specifiques au printemps. En hiver, le morning ramp est plus raide (h=7 pourrait necessiter +5 au lieu de +3.69). En ete, le evening peak est plus faible (h=18 pourrait necessiter -1 au lieu de -3.12).

### Impact estime

Si le biais saisonnier change de 1-2 EUR/h entre saisons, l'HBC applique des corrections suboptimales pour ~50% du test (les saisons non vues). Impact : 0.1-0.3 RMSE.

### Fix

- **Cross-validated HBC** : calculer les corrections sur plusieurs folds temporels (expanding window) et moyenner
- **Seasonality-aware HBC** : utiliser les memes mois du training set (ex: HBC de Jul 2023 pour Jul 2024)

---

## 7. MOYENNE — Elastic Net Sans Sample Weights

### Probleme

CatBoost et LightGBM FR sont entraines avec des sample weights (time_decay / variance) :
```python
Pool(X, y, weight=fr_stat["weights"])    # CatBoost
lgb_fr.fit(X, y, sample_weight=weights)  # LightGBM
```

L'Elastic Net FR ignore les poids :
```python
en_fr.fit(X_fr_tr_scaled, fr_stat["y_dev_tr"][fr_stat["valid_tr"]])  # pas de poids
```

### Impact

L'Elastic Net est entraine sur une distribution non-ponderee, donnant autant d'importance aux donnees de la crise (Jul 2022, high variance) qu'aux donnees recentes. Les trees, eux, downweightent les vieilles donnees.

Quand l'ensemble mixe EN (non-pondere) avec les trees (ponderes), c'est une source de diversite involontaire mais potentiellement benefique.

### Verdict

Probablement neutre ou legerement benefique (diversite). Mais si on voulait etre coherent, on pourrait utiliser `sample_weight` dans sklearn via une version ponderee de l'EN.

---

## 8. BASSE — Scaler DNN Partage FR/UK (v7 seulement)

### Probleme (v7)

```python
dnn_scaler = StandardScaler()
X_dnn_tr = dnn_scaler.fit_transform(df_train[feat_dnn_final])  # MEME scaler pour FR et UK
```

Le DNN FR et UK partagent le meme scaler, fit sur le full training set. Mais les distributions FR et UK sont differentes (prix, volumes, etc.).

### Statut

**Corrige en v8** avec des scalers separes :
```python
dnn_scaler_fr = StandardScaler()  # fit sur df_train (full)
dnn_scaler_uk = StandardScaler()  # fit sur df_train_uk (12m)
```

Au retrain (v7), des scalers separes sont deja utilises. Donc le bug v7 n'affecte que le DNN de validation (pas la soumission finale).

---

## 9. BASSE — Feature Lists Potentiellement Stale

### Probleme

Les features FR (27+1) et UK (150) viennent de fichiers JSON de selection :
- `outputs/feature_selection_v5_fr.json` — cree en v5
- `outputs/uk_feature_research.json` — cree en v?

Depuis la selection, le feature engineering a ete modifie :
- Ajout des categories 28-30 (FR continent, UK island, regime features)
- Ajout des proxies avancees (cat 32)
- V8 : rolling_336h, stress_index, load_surprise

### Risque

Certaines features des nouvelles categories pourraient etre plus utiles que des features selectionnees a l'epoque. Par exemple :
- `euro_scarcity_ratio` est documente comme "top feature" mais est-il dans les 150 UK features ?
- `fr_dynamic_marginal` pourrait battre `fr_spark_spread` dans la liste FR

### Fix

Re-executer la feature selection (SHAP + Boruta) sur l'etat actuel du feature engineering pour mettre a jour les listes.

---

## 10. OK — Pas de Feature Leakage Detecte

### Verification

Toutes les features utilisent exclusivement :
- `*_la` : valeurs actuelles lagged (prix D-1 publie avant l'enchère D)
- `*_f` : forecasts day-ahead (disponibles avant l'enchère)
- Features derivees des ci-dessus (rolling, diff, ratios)

Aucune feature n'utilise `fr_spot` ou `uk_spot` (les targets) directement.

L'EMA pour le target FR est calculee sur `fr_spot_la` (pas `fr_spot`), confirmant l'absence de leakage.

Le merit_order_cost pour le target UK est calcule a partir de `uk_gas` et `uk_emission` (pas `uk_spot`), confirmant l'absence de leakage.

---

## 11. OK — Clipping Raisonnable

```python
fr_q_low = np.percentile(train_fe["fr_spot"].dropna(), 0.1)   # -40.6
fr_q_high = np.percentile(train_fe["fr_spot"].dropna(), 99.9)  # 800.0
uk_q_low = np.percentile(train_fe["uk_spot"].dropna(), 0.1)   # -22.5
uk_q_high = np.percentile(train_fe["uk_spot"].dropna(), 99.9)  # 798.5
```

Les predictions test sont dans les bornes : FR mean=78.8, UK mean=104.3. Le clipping ne devrait affecter que des cas extremes. Pas de risque ici.

---

## 12. OK — EMA Target Reconstruction Correcte

La reconstruction `pred_spot = EMA(spot_la) + model.predict(X)` est correcte :
- L'EMA est causale (utilise seulement le passe)
- L'EMA au test est continue avec le training (pas de cold-start)
- L'EMA utilise `fr_spot_la` (pas `fr_spot`), donc pas de leakage

La reconstruction UK `pred_spot = MOC + model.predict(X)` est egalement correcte :
- MOC est calcule a partir de gas/emission (disponibles a la prediction)
- Pas de leakage de `uk_spot`

---

## Plan d'Action Prioritise

| Priorite | Action | Gain estime | Effort |
|----------|--------|-------------|--------|
| **P0** | Fix cold-start features (concatener avant build_features) | -0.3 a -0.6 RMSE | 30 min |
| **P1** | Regime weights : coarser grid (0.2) + shrinkage | -0.1 a -0.3 RMSE | 15 min |
| **P1** | Supprimer ou re-tuner XGBoost FR | 0 (cleanup) ou -0.1 | 10 min |
| **P2** | DNN retrain avec holdout interne | -0.05 a -0.15 RMSE | 20 min |
| **P2** | HBC seasonality-aware (ou cross-validated) | -0.1 a -0.3 RMSE | 45 min |
| **P3** | Re-selectionner features avec SHAP actuel | -0.1 a -0.5 RMSE | 2h |
| **P3** | Monthly HBC dampened pour soumission | -0.2 a -0.5 RMSE (risque) | 10 min |

**Total potentiel : -0.5 a -1.5 RMSE** (si tous les gains composent, ce qui est optimiste)

Meilleur Kaggle actuel : 24.88
1ere place : 23.14
Gap : 1.74

Les fixes P0+P1 pourraient reduire le gap a ~1.0-1.2 RMSE.
