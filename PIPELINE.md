## 2. The End-to-End Pipeline

### Step 1: Data Ingestion & Preprocessing
* **Multi-Zone Integration:** Merge the 12 zones (AT, BE, CH, CZ, DE, DK1, DK2, FR, NL, NO2, PL, SE4) from the directory specified in `config.yaml`. 
* **Spike/Outlier Treatment:** Raw electricity prices are extremely volatile. Use a Recursive Filter or Median Absolute Deviation (MAD) to detect spikes and replace them with the median of similar days.
* **Scaling:** Apply Min-Max or Standard scaling for the LEAR and DNN models. Tree models generally do not require this.
* **Technical Data Contract (No Lookahead Bias):**
    * **Granularity:** All 15-minute data (Load, Generation) must be resampled to hourly (`.resample('1h').mean()`) to match the Day-Ahead auction frequency.
    * **Missing Data:** Never interpolate. For gaps ≤ 3 hours, use Forward-Fill (`ffill`). For gaps > 3 hours, impute using the value from exactly 24 hours prior or 168 hours prior.

### Step 2: Feature Engineering & Selection
* **Time Lags:** Create 24-hour, 48-hour, and 168-hour (1 week) lags for all zones.
* **Generation Aggregation & Residual Load:** Do NOT pass all 17 ENTSO-E generation types into the model directly. Engineer the following features:
    * `Renewables` = Solar + Wind Onshore + Wind Offshore
    * `Residual_Load` = Total Load - Renewables (This is the most critical feature for European price volatility).
    * `Baseload` = Nuclear + Run-of-River Hydro + Biomass
    * `Dispatchable` = Fossil-Gas + Hard-Coal + Lignite
* **The "Greedy Algorithm" for Market Integration:** Do not throw all 12 zones into a target zone's model at once, as this causes severe overfitting. Start with a target zone's domestic features. Iteratively add neighboring country bundles and evaluate if the validation error decreases.

### Step 3: Point Forecasting (The Base Models)
* **Model 1: LEAR (Baseline):** The Lasso Estimated Autoregressive model handles automatic feature selection via the L1 penalty and serves as the ultimate statistical baseline. Train 24 separate univariate models (one for each hour) and adapt the implementation from the open-source `epftoolbox`. 
* **Models 2, 3 & 4: XGBoost, LightGBM & CatBoost:** Utilize tree-based models to handle sudden non-linear market shifts and extreme volatility. CatBoost specifically will excel at handling the categorical calendar features natively.
* **Model 5: Deep Neural Network (DNN):** Train a single multivariate network with 24 output nodes to predict the entire day-ahead curve at once, capturing intra-day correlations. 
* **Optimization:** Use the Tree-structured Parzen Estimator (TPE) algorithm for Bayesian hyperparameter tuning via `hyperopt`. Avoid Grid Search.

### Step 4: Probabilistic Forecasting (The Ensemble)

* **Method:** Quantile Regression Averaging (QRA).
* **Execution:** Feed the point forecasts of LEAR, the Tree Models, and the DNN into a quantile regression layer.
* **Outputs:** Generate the 5%, 50%, and 95% quantiles to provide trading confidence intervals. Ensure quantile crossing is prevented via post-processing smoothing.

### Step 5: Evaluation Metrics
* **Point Forecast Metrics:** Use sMAPE (Symmetric Mean Absolute Percentage Error) and rMAE (Relative Mean Absolute Error). Do not use standard MAPE, as prices close to zero will break the denominator.
* **Probabilistic Metrics:** Pinball Loss (for sharpness and reliability) and Winkler Score (penalizes wide intervals). Do not use the Coverage Width-based Criterion (CWC), as it is an improper scoring rule.
* **Statistical Significance:** Execute the Diebold-Mariano (DM) test to prove your advanced models statistically outperform the LEAR baseline. **Note:** Run this test separately for each of the 24 hours, as day-ahead forecast errors are serially correlated.

### Step 6: Explainability
* **SHAP Values:** Use SHAP (SHapley Additive exPlanations) on the XGBoost/LightGBM/CatBoost models to explain exactly how much specific cross-border flows or weather events shifted the hourly price.