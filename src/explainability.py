import logging
import yaml
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
from src.models.tree_models import train_xgboost

logger = logging.getLogger(__name__)


def generate_shap_analysis():
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    raw_directory = config.get("data", {}).get("raw_dir", "data/raw/auhack_legacy/")
    outputs_dir = Path(config.get("data", {}).get("output_dir", "data/outputs/"))
    outputs_dir.mkdir(parents=True, exist_ok=True)
    target_zones = config.get("data", {}).get("target_zones", ["DE"])

    explainability_root = outputs_dir / "explainability"
    explainability_root.mkdir(parents=True, exist_ok=True)

    zone_explanations = []

    for zone in target_zones:
        zone_dir = explainability_root / zone
        zone_dir.mkdir(parents=True, exist_ok=True)

        # 1. Load & Preprocess Data
        logger.info(f"Generating SHAP Explainability for {zone}...")

        df = load_and_merge_zone(zone, raw_directory)
        df["Spot_Price_Filtered"] = apply_mad_filter(
            df[TARGET_COL], window="24h", z=3.0
        )
        df = add_deterministic_features(df)

        lag_targets = ["Spot_Price_Filtered", "Residual_Load"]
        lags_list = [24, 48, 168]
        df = create_lags(df, lag_targets, lags_list)

        active_features = ["Hour", "DayOfWeek", "Month"]
        for col in lag_targets:
            for lag in lags_list:
                active_features.append(f"{col}_lag_{lag}")

        df = df.dropna(subset=active_features + [TARGET_COL])

        train_df, val_df, test_df = chronological_train_val_test_split(
            df, val_ratio=0.15, test_ratio=0.15
        )

        X_train = train_df[active_features]
        y_train = train_df[TARGET_COL]
        X_val = val_df[active_features]
        y_val = val_df[TARGET_COL]
        X_test = test_df[active_features]

        # 2. Train the Best Tree Model (XGBoost) with Optimal Params
        xgb_params = (
            config.get("model_settings", {}).get("trees", {}).get("xgb", {}).copy()
        )
        xgb_params["early_stopping_rounds"] = 50

        logger.info("Training XGBoost for SHAP exploration...")
        model = train_xgboost(X_train, y_train, X_val, y_val, params=xgb_params)

        # 3. Compute SHAP Values
        logger.info("Computing SHAP values (TreeExplainer)...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        shap_explanation = explainer(X_test)
        zone_explanations.append((zone, shap_explanation))

        # 4. Generate & Save Individual Zone Plots
        logger.info(f"Saving SHAP visualizations for {zone}...")

        # Summary Plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test, show=False)
        summary_path = zone_dir / "shap_summary.png"
        plt.savefig(summary_path, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved: {summary_path}")

        # Dependence Plot for Residual Load (Most critical feature)
        if "Residual_Load_lag_24" in X_test.columns:
            plt.figure(figsize=(10, 6))
            shap.dependence_plot(
                "Residual_Load_lag_24", shap_values, X_test, show=False
            )
            dep_path = zone_dir / "shap_dependence_residual.png"
            plt.savefig(dep_path, bbox_inches="tight")
            plt.close()
            logger.info(f"Saved: {dep_path}")

    # 5. Pan-European SHAP importance (3x3 grid)
    if not zone_explanations:
        logger.warning("No SHAP explanations generated. Skipping pan-European plot.")
        return

    n_slots = 9
    fig, axes = plt.subplots(3, 3, figsize=(24, 18))
    axes = axes.flatten()

    if len(zone_explanations) > n_slots:
        logger.warning(
            "More than 9 zones detected; plotting the first 9 zones in the 3x3 grid."
        )

    for idx, (zone, zone_exp) in enumerate(zone_explanations[:n_slots]):
        ax = axes[idx]
        shap.plots.bar(zone_exp, max_display=10, show=False, ax=ax)
        ax.set_title(zone)

    for idx in range(len(zone_explanations[:n_slots]), n_slots):
        axes[idx].axis("off")

    fig.suptitle("Pan-European SHAP Feature Importance (Top 10 per Zone)", fontsize=20)
    fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.96))
    pan_path = explainability_root / "pan_european_shap_importance.png"
    fig.savefig(pan_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {pan_path}")

    logger.info("Pan-European Explainability Analysis Complete.")


if __name__ == "__main__":
    generate_shap_analysis()
