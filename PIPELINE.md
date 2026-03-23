## The End-to-End Pipeline

### Step 1: Data Ingestion & Preprocessing
* **Data Source:** Load `x_train.csv` and `y_train.csv` via the directory specified in `config.yaml`. 
* **Missing Data Imputation:** Handle the sparsely populated daily columns (e.g., `eu_emission`, `nl_gas`) which only appear at 00:00 CET. Use causal Forward-Fill (`ffill`) to propagate the daily values across the remaining 23 hours.
* **Target Leakage Prevention:** Do not use `_la` (lagged actual) columns as unshifted current-day features, though they are inherently shifted 24 hours. Be aware of the explicit warning about exploiting the test set.

### Step 2: Feature Engineering & Selection
* **Domain-Specific Engineering:** * `fr_residual_load` = `fr_load_f` - (`fr_solar_f` + `fr_wind_f`).
    * `uk_residual_load` = `uk_load_f` - (`uk_solar_f` + `uk_wind_f`).
* **Interconnector Dynamics:** Calculate the Available Transfer Capacity (ATC) versus Net Transfer Capacity (NTC) across the 6 UK interconnectors.
* **Calendar Features:** Extract `hour`, `dayofweek`, `month` from `datetime_CET`. Integrate France and UK public holidays via the `holidays` package.

### Step 3: Point Forecasting (The Base Models)
* **Model 1: LEAR (Baseline):** A Lasso Estimated Autoregressive model using `LassoLarsIC` to automatically determine the L1 penalty, serving as the statistical floor. 
* **Models 2, 3 & 4: XGBoost, LightGBM & CatBoost:** Multi-output tree regressors natively handling NaN values from sparse gas/emissions data.
* **Model 5: Deep Neural Network (Multivariate PyTorch):** Train a Neural Network to predict both `fr_spot` and `uk_spot` simultaneously. Requires strict `StandardScaler` application on all features to prevent exploding gradients.

### Step 3.5: Bayesian Hyperparameter Optimization
* **Algorithm:** Use Hyperopt (TPE) to optimize `n_estimators`, `learning_rate`, `depth`, and regularization. Evaluate against the Validation RMSE.

### Step 4: The Ensemble & Final Submission
* **Ensembling:** Blend the predictions of LightGBM, XGBoost, and the PyTorch DNN using a ridge regression or a weighted average to minimize the final RMSE.
* **Submission Formatting:** Generate a final output matching `sample_submission.csv` exactly, with columns `id`, `fr_spot`, and `uk_spot`.