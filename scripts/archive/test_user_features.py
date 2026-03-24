import sys
import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
from pathlib import Path
import json

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.data_loading import load_data, merge_train
from src.feature_engineering import build_features
from omegaconf import OmegaConf

def compute_expert_features(df, df_feat):
    new_f = pd.DataFrame(index=df.index)
    
    # 1. Structure
    new_f['new_fr_spot_lag_168'] = df['fr_spot_la'].shift(168)
    new_f['new_fr_daily_range_la'] = (df['fr_spot_la'].rolling(24).max() - df['fr_spot_la'].rolling(24).min()).shift(24)
    # Morning ramp proxied by difference
    
    # 2. Regimes
    gas_thresh = df['fr_gas'].quantile(0.66)
    new_f['new_gas_regime_high'] = (df['fr_gas'] > gas_thresh).astype(int)
    new_f['new_carbon_momentum'] = df['eu_emission'].diff(24)
    
    load_thresh = df['fr_load_f'].quantile(0.66)
    new_f['new_high_gas_x_high_load'] = new_f['new_gas_regime_high'] * (df['fr_load_f'] > load_thresh).astype(int)

    # 3. Market Coupling
    # Misalignment: sign(spread) != sign(flow)
    spread = df['fr_spot_la'] - df['uk_spot_la']
    flow_fr_uk_total = df[['flow_fr-uk-1_la', 'flow_fr-uk-2_la', 'flow_fr-uk-3_la']].sum(axis=1) - df[['flow_uk-fr-1_la', 'flow_uk-fr-2_la', 'flow_uk-fr-3_la']].sum(axis=1)
    new_f['new_flow_misalignment'] = (np.sign(spread) != np.sign(flow_fr_uk_total)).astype(int)
    
    # Congestion persistence
    fr_uk_atc_total = 3000 # hardcoded approx capacity for the sum of cables
    utilization = flow_fr_uk_total.abs() / fr_uk_atc_total
    new_f['new_congestion_persistence'] = (utilization > 0.9).rolling(24).sum().shift(24)

    # 4. Basis Markup
    # Using 0.55 efficiency CCGT and 0.37 carbon factor
    merit_order = (df['fr_gas']/0.55) + (df['eu_emission']*0.37)
    new_f['new_fr_markup_la'] = df['fr_spot_la'] - merit_order
    new_f['new_fr_markup_lag_168h'] = new_f['new_fr_markup_la'].shift(168)
    
    return new_f

def main():
    print("Loading data...")
    x_train, y_train, x_test = load_data(project_root / "data" / "raw")
    df = merge_train(x_train, y_train)
    
    config = OmegaConf.load(project_root / "config.yaml")
    
    print("Building base features...")
    df_feat = build_features(df, config)
    
    print("Adding expert features...")
    df_expert = compute_expert_features(df, df_feat)
    
    # Merge
    X = pd.concat([df_feat.drop(columns=['fr_spot', 'uk_spot', 'id', 'day', 'date', 'Unnamed: 0'], errors='ignore'), df_expert], axis=1)
    # The actual target is in df['fr_spot']
    y = df['fr_spot']
    
    X = X.select_dtypes(include=[np.number])
    print(f"Shape before dropna: {X.shape}")
    X = X.dropna(axis=1, how='all')
    print(f"Shape after dropna(how='all'): {X.shape}")
    
    # Fill NAs to avoid losing all rows (especially because of 168h lags creating NaNs at the beginning)
    X = X.fillna(method='bfill').fillna(0)
    
    valid_mask = y.notna()
    X = X[valid_mask]
    y = y[valid_mask]
    print(f"Shape after y.notna(): {X.shape}")
    
    # Train / Val split
    split_date = pd.to_datetime("2024-02-01", utc=True)
    df['datetime_UTC'] = pd.to_datetime(df['datetime_UTC'], utc=True)
    train_mask = df.loc[X.index, 'datetime_UTC'] < split_date
    val_mask = df.loc[X.index, 'datetime_UTC'] >= split_date
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    print(f"X_train shape: {X_train.shape}, X_val shape: {X_val.shape}")
    
    print(f"Training LightGBM on {X_train.shape[1]} features...")
    model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=31, random_state=42)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50)])
    
    print("Computing SHAP values on Validation set...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_val)
    
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_ranking = pd.DataFrame({
        'feature': X_val.columns,
        'importance': mean_abs_shap
    }).sort_values('importance', ascending=False).reset_index(drop=True)
    
    shap_ranking['rank'] = shap_ranking.index + 1
    
    expert_cols = df_expert.columns.tolist()
    expert_ranks = shap_ranking[shap_ranking['feature'].isin(expert_cols)]
    
    print("\n" + "="*50)
    print("  EXPERT FEATURES RANKING (out of {})".format(len(X_val.columns)))
    print("="*50)
    print(expert_ranks.to_string(index=False))
    
    top_5 = shap_ranking.head(5)
    print("\nTOP 5 GLOBAL FEATURES:")
    print(top_5.to_string(index=False))

    results = {
        'expert_ranks': expert_ranks.to_dict('records'),
        'top_5': top_5.to_dict('records')
    }
    
    with open(project_root / "outputs" / "expert_features_ranking.json", "w") as f:
        json.dump(results, f, indent=2)
        
if __name__ == "__main__":
    main()
