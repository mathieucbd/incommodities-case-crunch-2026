import logging
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from pathlib import Path
import sys

# Ensure src is in standard path for execution
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# Ingestion & Preprocessing
from src.data_ingestion import load_and_merge_zone
from src.features import create_lags, add_deterministic_features, apply_mad_filter
from src.preprocessing import chronological_train_val_test_split
from src.constants import TARGET_COL
from src.models.tree_models import train_catboost

logger = logging.getLogger(__name__)

def generate_shap_analysis(target_zone="DE"):
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    raw_directory = config.get('data', {}).get('raw_dir', 'data/raw/auhack_legacy/')
    outputs_dir = Path(config.get('data', {}).get('output_dir', 'data/outputs/'))
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load & Preprocess Data
    logger.info(f"Generating SHAP Explainability for {target_zone}...")
    
    df = load_and_merge_zone(target_zone, raw_directory)
    df['Spot_Price_Filtered'] = apply_mad_filter(df[TARGET_COL], window='24h', z=3.0)
    df = add_deterministic_features(df)
    
    lag_targets = ['Spot_Price_Filtered', 'Residual_Load']
    lags_list = [24, 48, 168]
    df = create_lags(df, lag_targets, lags_list)
    
    active_features = ['Hour', 'DayOfWeek', 'Month']
    for col in lag_targets:
        for lag in lags_list:
            active_features.append(f'{col}_lag_{lag}')
            
    df = df.dropna(subset=active_features + [TARGET_COL])
    
    train_df, val_df, test_df = chronological_train_val_test_split(df, val_ratio=0.15, test_ratio=0.15)
    
    X_train = train_df[active_features]
    y_train = train_df[TARGET_COL]
    X_val = val_df[active_features]
    y_val = val_df[TARGET_COL]
    X_test = test_df[active_features]
    
    # 2. Train the Best Tree Model (CatBoost) with Optimal Params
    cat_params = config.get('model_settings', {}).get('trees', {}).get('cat', {}).copy()
    cat_params['early_stopping_rounds'] = 50
    cat_params['train_dir'] = 'data/outputs/catboost_info'
    
    logger.info("Training CatBoost for SHAP exploration...")
    model = train_catboost(X_train, y_train, X_val, y_val, params=cat_params)
    
    # 3. Compute SHAP Values
    logger.info("Computing SHAP values (TreeExplainer)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # 4. Generate & Save Plots
    logger.info("Saving SHAP visualizations to data/outputs/...")
    
    # Summary Plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, show=False)
    summary_path = outputs_dir / "shap_summary.png"
    plt.savefig(summary_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {summary_path}")
    
    # Dependence Plot for Residual Load (Most critical feature)
    if 'Residual_Load_lag_24' in X_test.columns:
        plt.figure(figsize=(10, 6))
        shap.dependence_plot("Residual_Load_lag_24", shap_values, X_test, show=False)
        dep_path = outputs_dir / "shap_dependence_residual.png"
        plt.savefig(dep_path, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved: {dep_path}")

    # Bar plot for global importance
    plt.figure(figsize=(12, 8))
    shap.plots.bar(explainer(X_test), show=False)
    bar_path = outputs_dir / "shap_bar_importance.png"
    plt.savefig(bar_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {bar_path}")

    logger.info("Explainability Analysis Complete.")

if __name__ == "__main__":
    generate_shap_analysis()
