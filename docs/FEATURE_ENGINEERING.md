# InCommodities Case Crunch 2026: Feature Engineering Pipeline Guide

This formal documentation details the quantitative logic, physical grid assumptions, and rigorous leakage-prevention mechanisms implemented in the InCommodities Case Crunch 2026 feature engineering pipeline.

## 1. Physical Grid Fundamentals

### Thermal Floor (Marginal Cost Proxy)
The **Thermal Floor** feature captures the short-run marginal cost (SRMC) of electricity production from natural gas-fired Combined Cycle Gas Turbines (CCGT), which frequently set the clearing price in European Day-Ahead Markets. 
The mathematical proxy is defined as:
```text
Cost_Thermal = (gas_price / 0.5) + (emission_price * (0.202 / 0.5))
```
- **0.5**: Represents the baseline thermal efficiency (50%) of a modern CCGT plant, meaning it takes 2 MWh of continuous thermal gas energy to produce 1 MWh of electricity.
- **0.202**: Represents the specific emission factor (tonnes of CO2 per MWh of thermal natural gas burned). 

By forward-filling sparse gas and emission targets to create this floor, the model learns the lower bound of thermal bidding behavior without capturing irrelevant intraday noise.

### Nuclear Shortfall
The **Nuclear Shortfall** feature shifts focus from raw available capacity to *missing* capacity. The French grid is heavily dominated by a 56-reactor nuclear fleet with a roughly 61,400 MW installed capacity limit. 
We explicitly hardcode this constant limit (`61400 - fr_nuclear_avcap_f`) rather than using dynamic `max()` calculations across the dataframe, definitively eliminating lookahead bias.

*Note: UK Scarcity Metrics (total capacity, security margin, scarcity ratio) were surgically pruned following Mutual Information (MI) validation. UK spot variance is overwhelmingly gas-price driven (thermal floor) rather than physical capacity-driven, rendering those specific metrics statistical dead weight.*

## 2. Advanced Statistical Features

### `asinh` Transformation
To successfully manage heavy-tailed volatility distributions in the Day-Ahead Market, we implement the inverse hyperbolic sine (`np.arcsinh()`) rather than traditional `np.log1p()`. This is quantitatively superior because:
1. It smoothly handles the 3.13% incidence rate of **negative prices** specifically seen on the French grid.
2. It effectively compresses the extreme right-tail kurtosis (11.31) observed in UK pricing during stress events, scaling variance without corrupting the underlying order of magnitude.

### Residual Ramps
Ramps are modeled via a 3-hour differenced residual load (`.diff(3)`). This explicitly maps the extreme thermal generation stress caused by the "Duck Curve"—rapid shifts required to meet peak evening demand as solar generation inevitably collapse.

### Rolling Z-Scores & Volatility
1. **Regime Normalization**: A 14-day (336-hour) rolling Z-score algorithm isolates fundamental shifts in load regimes independently of raw seasonality, normalizing demand across varying structural conditions.
2. **Volatility Indicators**: The pipeline computes 24-hour and 72-hour rolling standard deviations of lagged target prices to embed short-run stress persistence directly into the feature matrix. Both implementations strictly utilize retrospective trailing windows (`center=False`) to guarantee zero future peeking.

### Weekly Periodicity
To capture the massive autocorrelation at the T-168 mark (predicting Monday at 12:00 using last Monday at 12:00), we lag the target features by shifting the existing T-24 features back a further 144 hours. Positive integer shifts explicitly lock the feature into the strictly observable past.

## 3. Structural Regimes & Pruning

### Interconnector Masks
High Voltage Direct Current (HVDC) interconnectors (e.g., Viking Link) possess enormous structural importance but often contain massive sequential blocks of `NaN` values resulting from physical outages or pre-commissioning dates. 
Instead of simple mean-imputation, we compute a binary `is_online` flag. This prevents tree-based algorithms from erroneously learning split mappings on arbitrary imputed values, allowing the network architecture to dynamically switch regimes based on structural availability.

### Feature Pruning
Following exhaustive Mutual Information analysis against dual targets:
- **Distant Grid Drops**: Residual load calculations for highly distant, low-correlation zones (e.g., DE, PL, CZ) and derivative ATC ratio features were fully pruned to drastically condense the feature space.
- **Tree-Interaction Defense**: Critical `calendar` features (hour, month, holiday flags) were explicitly protected from linear pruning lists. Traditional pairwise MI fails to capture deep, non-linear interactions necessary for tree architectures (e.g., "If hour=19 AND is_uk_holiday=1"). 

## 4. Pipeline Security (Zero Leakage)

### Dual Scaler Architecture
To maintain mathematical integrity and absolute strict chronological separation:
1. **Target Isolation**: `fr_spot` and `uk_spot` are meticulously isolated and stored before any feature transformations apply to prevent target distribution leakage into the input schema.
2. **Dual Transformers**: Our architecture deploys two independent `StandardScaler` instances (`feature_scaler` and `target_scaler`). Scaling input and output variances independently protects deeper DNN gradients in PyTorch.
3. **Strict Validation Fitting**: Both scalers are strictly instantiated and updated (`.fit_transform()`) directly onto the `train_df`. The ensuing validation (`val_df`) and holdout (`test_df`) arrays are subsequently processed via `.transform()` only, locking all distributional moments into the unobserved past.
