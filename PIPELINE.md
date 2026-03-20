## The End-to-End Pipeline

### Step 1: Data Ingestion & Preprocessing
* **Multi-Zone Integration:** Merge the 12 zones (AT, BE, CH, CZ, DE, DK1, DK2, FR, NL, NO2, PL, SE4) from the directory specified in `config.yaml`. 
* **Spike/Outlier Treatment:** Raw electricity prices are extremely volatile. Use a 24-hour rolling Median Absolute Deviation (MAD, z=3) to detect spikes and cap them to preserve loss function integrity.
* **Technical Data Contract (No Lookahead Bias):**
    * **Granularity:** All 15-minute data (Load, Generation) must be resampled to hourly (`.resample('1h').mean()`) to match the Day-Ahead auction frequency.
    * **Missing Data:** Never interpolate. For gaps ≤ 3 hours, use Forward-Fill (`ffill`). For gaps > 3 hours, impute using the value from exactly 24 hours prior or 168 hours prior.
* **Chronological Splitting & Scaling:** Time-series data cannot be randomly shuffled. Split strictly chronologically. Apply `StandardScaler` fitted *only* on the Training variance to prevent lookahead leakage. **Crucial:** Target prices (`y`) must also be scaled for Neural Networks to prevent exploding gradients.

### Step 2: Feature Engineering & Selection
* **Time Lags:** Create 24-hour, 48-hour, and 168-hour (1 week) lags for the Filtered Price and Residual Load. Safely drop NaNs specific to active features.
* **Generation Aggregation & Residual Load:** Do NOT pass all 17 ENTSO-E generation types into the model directly. Engineer the following features:
    * `Renewables` = Solar + Wind Onshore + Wind Offshore
    * `Residual_Load` = Total Load - Renewables (This is the most critical feature for European price volatility).
    * `Baseload` = Nuclear + Run-of-River Hydro + Biomass
    * `Dispatchable` = Fossil-Gas + Hard-Coal + Lignite
* **The "Greedy Algorithm" for Market Integration:** Do not throw all 12 zones into a target zone's model at once, as this causes severe overfitting. Start with a target zone's domestic features. Iteratively add neighboring country bundles and evaluate if the Walk-Forward CV error mathematically decreases.

### Step 3: Point Forecasting (The Base Models)
* **Model 1: LEAR (Baseline):** The Lasso Estimated Autoregressive model handles automatic feature selection via the L1 penalty and serves as the ultimate statistical baseline. Train 24 separate univariate models (one for each hour) using a moving 182-day calibration window.
* **Models 2, 3 & 4: XGBoost, LightGBM & CatBoost:** Utilize tree-based models to handle sudden non-linear market shifts. CatBoost natively encodes categorical calendar features (Hour, DayOfWeek, Month) and acts as the SOTA tree baseline.
* **Model 5: Deep Neural Network (Multivariate PyTorch):** Train a single multivariate network. Reshape 1D hourly data into daily tensors `(N_days, 24*Features)`. Output 24 continuous hourly predictions simultaneously `Linear(24)` utilizing L1 Loss (MAE) and Early Stopping.
* **[OPTIONAL] Data Augmentation for DNN:** To prevent overfitting in Deep Learning, artificially expand the training dataset by shifting the "start of the day" by 1 hour, 24 times. This increases the training size by 24x but severely increases computational cost. Enable only if time and hardware permit.

### Step 3.5: Bayesian Hyperparameter Optimization
* **Methodology:** Default model weights are mathematically insufficient for final DAM trading. We must execute a rigorous hyperparameter search to ensure fair architectural comparison.
* **Algorithm:** Use the Tree-structured Parzen Estimator (TPE) algorithm via the `hyperopt` library. Strictly avoid Grid Search due to its computational inefficiency in high-dimensional spaces.
* **Execution:** Optimize `n_estimators`, `learning_rate`, `depth`, and `dropout` against the Validation Set MAE. Lock the best parameters into `config.yaml` before proceeding to the Ensemble.

### Step 4: Probabilistic Forecasting (The Ensemble)
* **Method:** Quantile Regression Averaging (QRA) via LightGBM `objective='quantile'`.
* **Execution:** Extract Validation Set predictions from CatBoost, LightGBM, and the PyTorch DNN. Use these as input features to train a meta-model that predicts the 5%, 50%, and 95% actual price quantiles. Apply post-processing to mathematically prevent quantile crossing.

### Step 5: Evaluation Metrics
* **Point Forecast Metrics:** Use sMAPE (Symmetric Mean Absolute Percentage Error) and rMAE (Relative Mean Absolute Error relative to a 168h persistence baseline). Do not use standard MAPE, as prices close to zero will break the denominator.
* **Probabilistic Metrics:** Pinball Loss (for sharpness and reliability) and Winkler Score (penalizes wide intervals). Do not use the Coverage Width-based Criterion (CWC), as it is an improper scoring rule.
* **Statistical Significance:** Execute the Diebold-Mariano (DM) test to prove your advanced models statistically outperform the LEAR baseline. Run this test separately for each of the 24 hours.

### Step 6: Explainability
* **SHAP Values:** Use SHAP (SHapley Additive exPlanations) on the XGBoost/LightGBM/CatBoost models to explain exactly how much specific cross-border flows or weather events shifted the hourly price.