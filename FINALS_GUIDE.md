# Guide Finals InCommodities 2026

## Préparation (AVANT les finals — demain matin)

### 1. Entraîner et sauvegarder les modèles

```bash
cd "INCOMO 3"
python -u scripts/finals_save_models.py
```

**Durée estimée**: ~15 minutes
**Sortie**: Tous les modèles sauvegardés dans `outputs/finals_models/`

**Fichiers générés**:
- `features.json` — Listes de features
- `stl_params.json` — Paramètres STL
- `regime_weights.json` — Poids de régime WINTER
- `cluster_config.json` — Configuration clusters UK
- Modèles FR: `cb_fr.pkl`, `lgb_fr.pkl`, `xgb_fr.pkl`, `en_fr.pkl`, `dnn_fr.pt`, `dnn_scaler_fr.pkl`, `ridge_fr.pkl`
- Modèles UK: `cb_uk.pkl`, `lgb_uk.pkl`, `xgb_uk.pkl`, `xgb_cluster_uk.pkl`, `en_uk.pkl`, `dnn_uk.pt`, `dnn_scaler_uk.pkl`, `ridge_uk.pkl`

### 2. Vérifier que tous les fichiers sont présents

```bash
ls -lh outputs/finals_models/
```

Vous devriez voir **17 fichiers** au total.

---

## Pendant les Finals (15h00 demain)

### Workflow par round

**Chaque round dure < 3 minutes. Vous recevrez**:
- `x_test.csv` (24h à prédire)
- `x_validation.csv` (30 derniers jours)
- `y_validation.csv` (30 derniers jours)

### Commande unique pour générer les prédictions

```bash
cd "INCOMO 3"
python scripts/finals_inference.py data/finals/x_test.csv data/finals/x_validation.csv data/finals/y_validation.csv
```

**Sortie**: `y_test.csv` (prêt à soumettre)

### Exemple de workflow complet

```bash
# 1. Les organisateurs envoient les fichiers
# 2. Copier les fichiers dans data/finals/
mkdir -p data/finals
cp ~/Downloads/x_test.csv data/finals/
cp ~/Downloads/x_validation.csv data/finals/
cp ~/Downloads/y_validation.csv data/finals/

# 3. Lancer l'inférence
python scripts/finals_inference.py data/finals/x_test.csv data/finals/x_validation.csv data/finals/y_validation.csv

# 4. Soumettre y_test.csv
```

---

## Architecture du pipeline

### Modèles (7 FR + 8 UK)

**FR (7 modèles)**:
- CB (CatBoost)
- LGB (LightGBM)
- XGB (XGBoost)
- EN (ElasticNet)
- DNN (Deep Neural Network)
- RidgeF (Ridge Fundamentals)
- SR (Stacking Résiduel)

**UK (8 modèles)**:
- CB, LGB, XGB, EN, DNN, RidgeF, SR
- **XGB_C** (XGBoost Cluster — 4 modèles horaires)

### Poids de régime (WINTER)

**FR**:
- **night** (0-6h): SR=80%, CB=20%
- **morning** (7-10h): SR=70%, EN=10%, DNN=10%, CB=10%
- **day** (11-16h): SR=50%, CB=20%, LGB=10%, DNN=10%, RidgeF=10%
- **peak** (17-20h): **CB=60%**, DNN=20%, SR=10%, LGB=10%
- **late** (21-23h): SR=70%, CB=20%, DNN=10%

**UK**:
- **night** (0-6h): XGB=50%, LGB=40%, RidgeF=10%
- **morning** (7-10h): LGB=40%, CB=20%, XGB=20%, RidgeF=10%, DNN=10%
- **day** (11-16h): XGB=60%, XGB_C=10%, CB=10%, LGB=10%, DNN=10%
- **peak** (17-20h): XGB=40%, CB=20%, DNN=20%, RidgeF=10%, LGB=10%
- **late** (21-23h): XGB=40%, CB=20%, LGB=20%, RidgeF=10%, DNN=10%

### Features (fenêtres max = 336h = 14 jours)

**Toutes les features sont compatibles avec 30 jours de validation**:
- Fenêtre max: 336h (14 jours)
- Lags typiques: 24h, 48h, 168h (1 semaine)
- Rolling windows: 24h, 168h

### STL Decomposition

- **Période**: 168h (1 semaine)
- **Seasonal**: 13
- Appliqué sur `fr_spot_la` (concat validation + test)
- Target FR: `y = spot - STL_trend`

---

## Vérifications

### Avant les finals

```bash
# 1. Tous les modèles sont sauvegardés
ls outputs/finals_models/*.pkl outputs/finals_models/*.pt outputs/finals_models/*.json | wc -l
# Résultat attendu: 17

# 2. Test sur données fictives (optionnel)
python scripts/finals_inference.py data/raw/x_test.csv data/raw/x_train.csv data/raw/y_train.csv
# Devrait générer y_test.csv en < 3 min
```

### Pendant les finals

- ✅ Temps < 180s (3 minutes)
- ✅ `y_test.csv` contient 24 lignes (24h)
- ✅ Colonnes: `id`, `fr_spot`, `uk_spot`
- ✅ Pas de NaN dans les prédictions

---

## Troubleshooting

### Erreur "FileNotFoundError: outputs/finals_models/..."
→ Relancer `python scripts/finals_save_models.py`

### Temps > 3 minutes
→ Réduire le nombre de modèles dans `finals_inference.py` (supprimer SR)

### NaN dans les prédictions
→ Vérifier que `x_validation` contient au moins 14 jours (fenêtre max = 336h)

### "ModuleNotFoundError"
→ Vérifier l'environnement Python et les dépendances

---

## Checklist Finals

### Avant 15h00
- [ ] Lancer `finals_save_models.py` (1 fois)
- [ ] Vérifier que 17 fichiers sont dans `outputs/finals_models/`
- [ ] Tester `finals_inference.py` sur données fictives
- [ ] Chronométrer le temps d'exécution

### Pendant les rounds
- [ ] Recevoir x_test.csv, x_validation.csv, y_validation.csv
- [ ] Copier dans `data/finals/`
- [ ] Lancer `python scripts/finals_inference.py ...`
- [ ] Vérifier `y_test.csv` (24 lignes, pas de NaN)
- [ ] Soumettre avant la deadline

---

## Performance attendue

- **Kaggle best**: 23.20 (blend v17+v9 85/15)
- **Pipeline v17 seul**: 23.26
- **Finals (winter only)**: ~23.3-23.5 (robuste sur hiver)

---

Bon courage pour les finals ! 🚀
