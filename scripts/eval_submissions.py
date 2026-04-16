"""
Calculate REAL RMSE for all submissions using y_train_full (actuals Jul 2024 -> Feb 2025).
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

def compute_rmse(pred, actual):
    """Compute RMSE between predictions and actuals."""
    mask = np.isfinite(pred) & np.isfinite(actual)
    if mask.sum() == 0:
        return np.nan
    return np.sqrt(np.mean((pred[mask] - actual[mask])**2))

# Load y_train_full (actuals for test period)
print("Loading y_train_full...")
y_full = pd.read_csv("data/raw/y_train_full.csv")
print(f"  y_train_full: {len(y_full)} samples")
print(f"  ID range: {y_full['id'].min()} -> {y_full['id'].max()}")

# Original train ends at 2024-06-30 23:00:00 (ID 0-17543, 17544 samples)
# Test period: ID 17544 -> end (Jul 2024 -> Feb 2025)

# Filter y_full to test period (IDs >= 17544)
y_test_actual = y_full[y_full["id"] >= 17544].copy()

print(f"\nTest actuals (ID >= 17544): {len(y_test_actual)} samples")
print(f"  ID range: {y_test_actual['id'].min()} -> {y_test_actual['id'].max()}")
print(f"  FR spot: {y_test_actual['fr_spot'].notna().sum()} valid")
print(f"  UK spot: {y_test_actual['uk_spot'].notna().sum()} valid")

# Find all submission files
submission_files = sorted(Path("outputs").glob("submission*.csv"))
print(f"\nFound {len(submission_files)} submission files")

results = []

for sub_file in submission_files:
    # Load submission
    sub = pd.read_csv(sub_file)

    # Merge with actuals
    merged = sub.merge(y_test_actual[["id", "fr_spot", "uk_spot"]],
                       on="id", suffixes=("_pred", "_actual"))

    # Calculate RMSE
    rmse_fr = compute_rmse(merged["fr_spot_pred"].values, merged["fr_spot_actual"].values)
    rmse_uk = compute_rmse(merged["uk_spot_pred"].values, merged["uk_spot_actual"].values)
    rmse_sum = rmse_fr + rmse_uk

    results.append({
        "submission": sub_file.name,
        "rmse_fr": rmse_fr,
        "rmse_uk": rmse_uk,
        "rmse_sum": rmse_sum,
        "n_samples": len(merged),
    })

# Sort by RMSE sum
results_df = pd.DataFrame(results).sort_values("rmse_sum")

# Display results
print("\n" + "=" * 100)
print("REAL RMSE — ALL SUBMISSIONS (vs y_train_full actuals)")
print("=" * 100)
print(results_df.to_string(index=False))

# Save to JSON
output = {
    "test_period": {
        "id_min": int(y_test_actual["id"].min()),
        "id_max": int(y_test_actual["id"].max()),
        "n_samples": int(len(y_test_actual)),
    },
    "results": results_df.to_dict(orient="records"),
}

with open("outputs/real_rmse_all_submissions.json", "w") as f:
    json.dump(output, f, indent=2)

print("\n✅ Results saved to outputs/real_rmse_all_submissions.json")

# Display top 10
print("\n" + "=" * 100)
print("TOP 10 SUBMISSIONS (by RMSE sum)")
print("=" * 100)
for i, row in results_df.head(10).iterrows():
    print(f"{row['submission']:60s} | FR={row['rmse_fr']:6.2f} UK={row['rmse_uk']:6.2f} SUM={row['rmse_sum']:6.2f}")
