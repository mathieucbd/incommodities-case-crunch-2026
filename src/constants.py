import yaml
from pathlib import Path

config_path = Path(__file__).resolve().parent.parent / "config.yaml"
with open(config_path, "r") as f:
    _config = yaml.safe_load(f)

TARGET_COLS = _config.get("data", {}).get("target_cols", ["fr_spot", "uk_spot"])