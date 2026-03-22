import pandas as pd
from pathlib import Path
import yaml
from src.constants import (
    TARGET_COL,
    RENEWABLE_COLS,
    BASELOAD_COLS,
    DISPATCHABLE_COLS,
    FILE_SUFFIXES,
)


def normalize_dst(df: pd.DataFrame, tz_str: str = "Europe/Berlin") -> pd.DataFrame:
    """
    Normalizes timezone conversions mechanically eliminating 23-hour or 25-hour DST calendar faults.
    Spring explicitly pads continuous bounds mapping `.ffill()`.
    Autumn implicitly merges continuous bounds mapping `.resample('1h').mean()`.
    """
    df = df.copy()

    # Lock exact logical timezone vector natively capturing physical borders
    df.index = df.index.tz_convert(tz_str)

    # Rip out awareness mapping exact chronological integer boundaries uniformly
    df.index = df.index.tz_localize(None)

    # Automatically averages duplicate 25th boundaries internally natively!
    # And exposes exclusively the physical gap during 23h Spring offsets!
    df = df.resample("1h").mean()

    # Impute the exposed node natively mapping prior clock sequences
    df = df.ffill()

    return df


def load_and_merge_zone(target_zone: str, raw_dir: str) -> pd.DataFrame:
    """
    Loads spot price, total load, and generation data for a specified zone.
    Standardizes timezone to UTC, resamples to hourly frequency, merges,
    imputes missing data strictly without lookahead bias, and engineers Residual Load.
    """
    base_path = Path(raw_dir)

    # 1. Spot Price
    price_file = base_path / "spot-price" / f"{target_zone}{FILE_SUFFIXES['price']}"
    if not price_file.exists():
        raise FileNotFoundError(f"Missing spot price file: {price_file}")

    df_price = pd.read_csv(price_file)
    df_price["time"] = pd.to_datetime(df_price["time"], utc=True)
    df_price = df_price.rename(columns={"value (EUR/MWh)": TARGET_COL})
    df_price = df_price.set_index("time")
    df_price = df_price.resample("1h").mean()

    # 2. Total Load
    # The load file name might be e.g. "DE-total-load.csv"
    load_file = base_path / "total-load" / f"{target_zone}{FILE_SUFFIXES['load']}"
    if not load_file.exists():
        raise FileNotFoundError(f"Missing total load file: {load_file}")

    df_load = pd.read_csv(load_file)
    df_load["time"] = pd.to_datetime(df_load["time"], utc=True)
    df_load = df_load.rename(columns={"value (MW)": "Total_Load"})
    df_load = df_load.set_index("time")
    df_load = df_load.resample("1h").mean()

    # 3. Generation
    gen_file = base_path / "generation" / f"{target_zone}{FILE_SUFFIXES['generation']}"
    if not gen_file.exists():
        raise FileNotFoundError(f"Missing generation file: {gen_file}")

    df_gen_raw = pd.read_csv(gen_file)
    df_gen_raw["time"] = pd.to_datetime(df_gen_raw["time"], utc=True)

    # Pivot generation
    df_gen = df_gen_raw.pivot_table(
        index="time", columns="type", values="value (MW)", aggfunc="mean"
    )

    # Replace underscores with dashes to cleanly map to constants.py
    df_gen.columns = [str(c).replace("_", "-") for c in df_gen.columns]

    df_gen = df_gen.resample("1h").mean()

    # 4. Meteorological data
    weather_dir = base_path / "weather"
    weather_candidates = []
    if weather_dir.exists():
        weather_candidates.extend(
            sorted(weather_dir.glob(f"{target_zone}-open-meteo-*.csv"))
        )
        weather_candidates.extend(
            sorted(weather_dir.glob(f"{target_zone}*open-meteo*.csv"))
        )
    # Fallback: some datasets place weather files in root.
    weather_candidates.extend(sorted(base_path.glob(f"{target_zone}-open-meteo-*.csv")))
    weather_candidates.extend(sorted(base_path.glob(f"{target_zone}*open-meteo*.csv")))

    weather_file = next((p for p in weather_candidates if p.exists()), None)
    if weather_file is not None:
        # Open-Meteo exports contain metadata rows before the actual header.
        df_meteo = pd.read_csv(weather_file, skiprows=2)
        df_meteo["time"] = pd.to_datetime(df_meteo["time"], utc=True)
        df_meteo = df_meteo.set_index("time")
        # Remove physical units from names, e.g., "temperature_2m (°C)" -> "temperature_2m".
        df_meteo.columns = [str(col).split(" (")[0] for col in df_meteo.columns]
        df_meteo = df_meteo.resample("1h").mean()
    else:
        df_meteo = pd.DataFrame(index=df_price.index)

    # 5. Physical flows-in data
    flow_path = base_path / "flows" / f"{target_zone}-physical-flows-in.csv"
    if flow_path.exists():
        df_flows = pd.read_csv(flow_path)
        df_flows["time"] = pd.to_datetime(df_flows["time"], utc=True)
        df_flows = df_flows.pivot(index="time", columns="zone", values="value (MW)")
        df_flows = df_flows.rename(columns=lambda c: f"Flow_{c}")
        df_flows = df_flows.resample("1h").mean()
        df_flows = df_flows.ffill(limit=2)
    else:
        df_flows = pd.DataFrame(index=df_price.index)

    # 6. Merge all component datasets into one cohesive matrix.
    df_master = df_price.join([df_load, df_gen, df_meteo, df_flows], how="inner")

    # 7. Impute missing values (Strict NO LOOKAHEAD rule)
    # Forward fill up to 3 hours
    df_master = df_master.ffill(limit=3)

    # Fill remaining with T-24 (shift 24)
    df_master = df_master.fillna(df_master.shift(24))

    # Fill final remaining with T-168 (shift 168)
    df_master = df_master.fillna(df_master.shift(168))

    # 8. Feature Engineering
    # Helper to calculate sum safely ignoring missing generation types
    def safe_sum(df, cols):
        existing = [c for c in cols if c in df.columns]
        return df[existing].sum(axis=1) if existing else pd.Series(0, index=df.index)

    df_master["Renewables"] = safe_sum(df_master, RENEWABLE_COLS)
    df_master["Baseload"] = safe_sum(df_master, BASELOAD_COLS)
    df_master["Dispatchable"] = safe_sum(df_master, DISPATCHABLE_COLS)

    df_master["Residual_Load"] = df_master["Total_Load"] - df_master["Renewables"]

    # Execute identical DST normalization locking exact sequence bounds exclusively
    df_master = normalize_dst(df_master)

    zone = target_zone
    print(
        f"[INGESTION] {zone} loaded with columns: {df_master.columns.tolist()[:10]}..."
    )

    return df_master


if __name__ == "__main__":
    # Load config file
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    raw_directory = config["data"]["raw_dir"]
    test_zone = "DE"

    print(f"Testing load_and_merge_zone for '{test_zone}' from '{raw_directory}'...")

    df = load_and_merge_zone(test_zone, raw_directory)

    print(f"\nShape: {df.shape}")
    print("\nHead (First 5 rows):")
    print(df.head()[[TARGET_COL, "Total_Load", "Renewables", "Residual_Load"]])

    print("\nMissing Values Summary:")
    missing = df.isna().sum()
    if missing.sum() == 0:
        print("Perfect! 0 missing values remain.")
    else:
        print(missing[missing > 0])
