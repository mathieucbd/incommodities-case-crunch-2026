"""Feature engineering pipeline for INCOMO 3.

Builds ~290 derived features across 27 categories on top of the 111 raw columns.
Stateless: identical behavior on train and test, no target access.
CatBoost handles NaN natively — leading NaN from shift/rolling are left as-is.

Categories 1-16: original features (221)
Categories 17-23: advanced features added from discussion (~45)
  17 — Supply/Demand avancee (scarcity ratio, security margin, residual v2)
  18 — Load & Residual ramps 1h/3h
  19 — Multi-efficiency spark spreads (OCGT/CCGT)
  20 — Interconnexion avancee (flow/ATC, unused capacity)
  21 — Z-Scores & anomalies (14-day z-scores)
  22 — Signaux stochastiques / SDE (jump count, vol ratio, mean reversion)
  23 — Transforms ameliorees (asinh)
Categories 24-29: research-backed additions
  24 — Nuclear shortfall (expanding max gap)
  25 — ATC/NTC ratios per cable
  26 — Market-specific features (SDAC/N2EX, 10 sub-features)
  27 — Advanced price formation signals (partial-r validated)
  28 — FR continent territory (SDAC merit order, zero-MC pen, weighted price)
  29 — UK island territory (gas fleet stress, capacity margin, self-sufficiency)
  30 — Regime & structural breaks (iberian exception, gas-on-margin, oversupply)
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

try:
    import holidays as holidays_lib
except ImportError:
    holidays_lib = None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Build all engineered features for electricity price forecasting.

    Args:
        df: Raw dataframe from load_data(). Must have ``datetime_CET`` column.
        config: Parsed ``config.yaml`` dictionary.

    Returns:
        DataFrame with all original columns plus ~220 engineered features.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
        return _build_features_impl(df, config)


def _build_features_impl(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    cfg = config["feature_engineering"]
    df = df.copy()

    df = _preprocess(df, cfg)
    df = _add_calendar_features(df)
    df = _add_holiday_features(df)
    df = _add_fundamental_market_features(df, cfg)
    df = _add_renewable_penetration_features(df)
    df = _add_renewable_dynamics_features(df)
    df = _add_nuclear_features(df, cfg)
    df = _add_hydro_features(df)
    df = _add_interconnector_features(df)
    df = _add_price_features(df)
    df = _add_interaction_features(df, cfg)
    df = _add_regional_features(df)
    df = _add_river_temperature_features(df, cfg)
    df = _add_nonlinear_transforms(df)
    df = _add_rolling_statistics(df)
    df = _add_momentum_features(df)
    df = df.copy()  # defragment at the midpoint (~150 columns added so far)
    df = _add_advanced_supply_demand(df, cfg)
    df = _add_load_residual_ramps(df)
    df = _add_multi_efficiency_spark(df, cfg)
    df = _add_advanced_interconnection(df)
    df = _add_zscore_anomalies(df)
    df = _add_stochastic_signals(df, cfg)
    df = _add_improved_transforms(df)
    df = _add_nuclear_shortfall(df)
    df = _add_atc_ratios_per_cable(df)
    df = _add_market_specific_features(df, cfg)
    df = _add_price_formation_signals(df, cfg)
    df = _add_fr_continent_features(df, cfg)
    df = _add_uk_island_features(df, cfg)
    df = _add_regime_features(df)
    df = _add_advanced_price_proxies(df, cfg)
    df = _add_daily_renewable_surplus_features(df)

    # Defragment the DataFrame to avoid PerformanceWarning
    df = df.copy()

    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_sum(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Sum columns that exist in df, ignoring missing ones."""
    existing = [c for c in cols if c in df.columns]
    if not existing:
        return pd.Series(0.0, index=df.index)
    return df[existing].sum(axis=1)


def _safe_mean(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Mean of columns that exist in df, ignoring missing ones."""
    existing = [c for c in cols if c in df.columns]
    if not existing:
        return pd.Series(np.nan, index=df.index)
    return df[existing].mean(axis=1)


def _safe_col(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    """Return column if it exists, else a constant series."""
    if col in df.columns:
        return df[col]
    return pd.Series(default, index=df.index)


# ---------------------------------------------------------------------------
# Cat 1 — Pre-processing
# ---------------------------------------------------------------------------

def _preprocess(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Sort chronologically and forward-fill daily columns."""
    df = df.sort_values("datetime_CET").copy()

    for col in cfg.get("daily_cols", []):
        if col in df.columns:
            df[col] = df[col].ffill()

    return df


# ---------------------------------------------------------------------------
# Cat 2 — Calendar / Time features (21)
# ---------------------------------------------------------------------------

def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    dt = df["datetime_CET"]
    h = dt.dt.hour
    dow = dt.dt.dayofweek
    m = dt.dt.month
    doy = dt.dt.dayofyear

    # Raw integers
    df["hour"] = h
    df["day_of_week"] = dow
    df["month"] = m
    df["day_of_year"] = doy
    df["week_of_year"] = dt.dt.isocalendar().week.astype(int).values
    df["quarter"] = dt.dt.quarter

    # Cyclical encodings
    df["hour_sin"] = np.sin(2 * np.pi * h / 24)
    df["hour_cos"] = np.cos(2 * np.pi * h / 24)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    df["month_sin"] = np.sin(2 * np.pi * m / 12)
    df["month_cos"] = np.cos(2 * np.pi * m / 12)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    # Binary indicators
    df["is_weekend"] = (dow >= 5).astype(np.int8)
    df["is_business_hour"] = ((dow < 5) & (h >= 8) & (h <= 19)).astype(np.int8)
    df["is_morning_ramp"] = ((h >= 6) & (h <= 9)).astype(np.int8)
    df["is_evening_peak"] = ((h >= 17) & (h <= 20)).astype(np.int8)
    df["is_night"] = ((h >= 23) | (h <= 5)).astype(np.int8)
    df["is_solar_hours"] = ((h >= 10) & (h <= 16)).astype(np.int8)

    # Composite
    df["hour_x_dow"] = h * 7 + dow

    return df


# ---------------------------------------------------------------------------
# Cat 3 — Holiday features (10)
# ---------------------------------------------------------------------------

def _add_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    dates = df["datetime_CET"].dt.date
    years = df["datetime_CET"].dt.year
    yr_min, yr_max = int(years.min()), int(years.max())
    year_range = range(yr_min, yr_max + 2)  # +2 for safety

    if holidays_lib is None:
        # Fallback: no holiday features
        for col in [
            "is_holiday_fr", "is_holiday_uk", "is_holiday_de",
            "is_holiday_be", "is_holiday_nl",
            "is_bridge_day_fr", "is_holiday_or_weekend_fr",
            "is_holiday_or_weekend_uk",
            "days_to_next_holiday_fr", "days_since_last_holiday_fr",
        ]:
            df[col] = 0
        return df

    # Build holiday sets
    hol_fr = holidays_lib.France(years=year_range)
    hol_uk = holidays_lib.UnitedKingdom(years=year_range)
    hol_de = holidays_lib.Germany(years=year_range)
    hol_be = holidays_lib.Belgium(years=year_range)
    hol_nl = holidays_lib.Netherlands(years=year_range)

    df["is_holiday_fr"] = dates.map(lambda d: int(d in hol_fr)).astype(np.int8)
    df["is_holiday_uk"] = dates.map(lambda d: int(d in hol_uk)).astype(np.int8)
    df["is_holiday_de"] = dates.map(lambda d: int(d in hol_de)).astype(np.int8)
    df["is_holiday_be"] = dates.map(lambda d: int(d in hol_be)).astype(np.int8)
    df["is_holiday_nl"] = dates.map(lambda d: int(d in hol_nl)).astype(np.int8)

    # Bridge day: Monday after Sunday holiday or Friday before Saturday holiday
    dow = df["datetime_CET"].dt.dayofweek
    date_series = pd.to_datetime(df["datetime_CET"].dt.date)
    prev_day = (date_series - pd.Timedelta(days=1)).dt.date
    next_day = (date_series + pd.Timedelta(days=1)).dt.date
    df["is_bridge_day_fr"] = (
        ((dow == 0) & prev_day.map(lambda d: d in hol_fr))
        | ((dow == 4) & next_day.map(lambda d: d in hol_fr))
    ).astype(np.int8)

    # Combined
    df["is_holiday_or_weekend_fr"] = (
        (df["is_holiday_fr"] == 1) | (df["is_weekend"] == 1)
    ).astype(np.int8)
    df["is_holiday_or_weekend_uk"] = (
        (df["is_holiday_uk"] == 1) | (df["is_weekend"] == 1)
    ).astype(np.int8)

    # Distance to next / since last FR holiday
    fr_holiday_dates = sorted(hol_fr.keys())
    fr_hol_array = np.array([np.datetime64(d) for d in fr_holiday_dates])
    date_vals = date_series.values.astype("datetime64[D]")

    days_to_next = np.full(len(date_vals), 30, dtype=np.int32)
    days_since_last = np.full(len(date_vals), 30, dtype=np.int32)

    for i, d in enumerate(date_vals):
        future = fr_hol_array[fr_hol_array >= d]
        if len(future) > 0:
            days_to_next[i] = min(int((future[0] - d) / np.timedelta64(1, "D")), 30)
        past = fr_hol_array[fr_hol_array <= d]
        if len(past) > 0:
            days_since_last[i] = min(int((d - past[-1]) / np.timedelta64(1, "D")), 30)

    df["days_to_next_holiday_fr"] = days_to_next
    df["days_since_last_holiday_fr"] = days_since_last

    return df


# ---------------------------------------------------------------------------
# Cat 4 — Fundamental market features (23)
# ---------------------------------------------------------------------------

def _add_fundamental_market_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    eff = cfg.get("gas_efficiency", 0.50)
    em = cfg.get("emission_factor", 0.37)

    # Residual loads
    df["fr_residual_load"] = df["fr_load_f"] - _safe_col(df, "fr_solar_f") - _safe_col(df, "fr_wind_f")
    df["uk_residual_load"] = df["uk_load_f"] - _safe_col(df, "uk_solar_f") - _safe_col(df, "uk_wind_f")
    df["de_residual_load"] = df["de_load_f"] - _safe_col(df, "de_solar_f") - _safe_col(df, "de_wind_f")

    # Thermal needs
    df["fr_thermal_need"] = df["fr_residual_load"] - _safe_col(df, "fr_nuclear_avcap_f")
    df["uk_thermal_need"] = df["uk_residual_load"] - _safe_col(df, "uk_nuclear_avcap_f")

    # Spark spreads
    df["fr_spark_spread"] = _safe_col(df, "fr_gas") / eff + _safe_col(df, "eu_emission") * em
    df["uk_spark_spread"] = _safe_col(df, "uk_gas") / eff + _safe_col(df, "uk_emission") * em
    df["de_spark_spread"] = _safe_col(df, "de_gas") / eff + _safe_col(df, "eu_emission") * em
    df["nl_spark_spread"] = _safe_col(df, "nl_gas") / eff + _safe_col(df, "eu_emission") * em

    # Hydro total
    df["fr_hydro_total"] = _safe_col(df, "fr_hydro_res_f") + _safe_col(df, "fr_hydro_ror_f")

    # Baseload gaps
    df["fr_baseload_gap"] = df["fr_load_f"] - _safe_col(df, "fr_nuclear_avcap_f") - df["fr_hydro_total"]
    df["uk_baseload_gap"] = df["uk_load_f"] - _safe_col(df, "uk_nuclear_avcap_f") - _safe_col(df, "uk_biomass_avcap_f")

    # Positive parts
    df["fr_thermal_need_pos"] = df["fr_thermal_need"].clip(lower=0)
    df["uk_thermal_need_pos"] = df["uk_thermal_need"].clip(lower=0)
    df["fr_baseload_gap_pos"] = df["fr_baseload_gap"].clip(lower=0)

    # Spot minus spark (lagged)
    df["fr_spot_minus_spark"] = _safe_col(df, "fr_spot_la") - df["fr_spark_spread"]
    df["uk_spot_minus_spark"] = _safe_col(df, "uk_spot_la") - df["uk_spark_spread"]

    # Gas margin
    df["fr_gas_margin"] = _safe_col(df, "fr_gas_avcap_f") - df["fr_thermal_need_pos"]
    df["uk_gas_margin"] = _safe_col(df, "uk_gas_avcap_f") - df["uk_thermal_need_pos"]

    # Total dispatchable
    df["fr_total_dispatchable"] = (
        _safe_col(df, "fr_nuclear_avcap_f")
        + _safe_col(df, "fr_gas_avcap_f")
        + df["fr_hydro_total"]
    )
    df["uk_total_dispatchable"] = (
        _safe_col(df, "uk_nuclear_avcap_f")
        + _safe_col(df, "uk_gas_avcap_f")
        + _safe_col(df, "uk_biomass_avcap_f")
    )

    # Supply-demand ratio
    df["fr_supply_demand_ratio"] = df["fr_total_dispatchable"] / df["fr_load_f"].clip(lower=1)
    df["uk_supply_demand_ratio"] = (
        (df["uk_total_dispatchable"] + _safe_col(df, "uk_wind_f"))
        / df["uk_load_f"].clip(lower=1)
    )

    return df


# ---------------------------------------------------------------------------
# Cat 5 — Renewable penetration features (12)
# ---------------------------------------------------------------------------

def _add_renewable_penetration_features(df: pd.DataFrame) -> pd.DataFrame:
    # Individual penetrations
    df["fr_wind_pen"] = _safe_col(df, "fr_wind_f") / df["fr_load_f"].clip(lower=1)
    df["uk_wind_pen"] = _safe_col(df, "uk_wind_f") / df["uk_load_f"].clip(lower=1)
    df["fr_solar_pen"] = _safe_col(df, "fr_solar_f") / df["fr_load_f"].clip(lower=1)
    df["uk_solar_pen"] = _safe_col(df, "uk_solar_f") / df["uk_load_f"].clip(lower=1)

    # Total renewable penetration
    df["fr_renewable_pen"] = (
        (_safe_col(df, "fr_wind_f") + _safe_col(df, "fr_solar_f"))
        / df["fr_load_f"].clip(lower=1)
    )
    df["uk_renewable_pen"] = (
        (_safe_col(df, "uk_wind_f") + _safe_col(df, "uk_solar_f"))
        / df["uk_load_f"].clip(lower=1)
    )

    # DE penetrations (continental coupling)
    df["de_wind_pen"] = _safe_col(df, "de_wind_f") / df["de_load_f"].clip(lower=1)
    df["de_solar_pen"] = _safe_col(df, "de_solar_f") / df["de_load_f"].clip(lower=1)
    df["de_renewable_pen"] = (
        (_safe_col(df, "de_wind_f") + _safe_col(df, "de_solar_f"))
        / df["de_load_f"].clip(lower=1)
    )

    # Pan-European wind penetration
    cont_wind = _safe_sum(df, ["fr_wind_f", "de_wind_f", "be_wind_f", "nl_wind_f"])
    cont_load = _safe_sum(df, ["fr_load_f", "de_load_f", "be_load_f", "nl_load_f"])
    df["continental_wind_pen"] = cont_wind / cont_load.clip(lower=1)

    # UK wind thresholds (from EDA: negative prices above 50%)
    df["uk_wind_high"] = (df["uk_wind_pen"] > 0.50).astype(np.int8)
    df["uk_wind_very_high"] = (df["uk_wind_pen"] > 0.65).astype(np.int8)

    return df


# ---------------------------------------------------------------------------
# Cat 6 — Renewable dynamics features (15)
# ---------------------------------------------------------------------------

def _add_renewable_dynamics_features(df: pd.DataFrame) -> pd.DataFrame:
    # UK wind ramps
    df["uk_wind_ramp_1h"] = _safe_col(df, "uk_wind_f").diff(1)
    df["uk_wind_ramp_3h"] = _safe_col(df, "uk_wind_f").diff(3)
    df["uk_wind_ramp_6h"] = _safe_col(df, "uk_wind_f").diff(6)

    # FR wind / solar ramps
    df["fr_wind_ramp_3h"] = _safe_col(df, "fr_wind_f").diff(3)
    df["fr_solar_ramp_3h"] = _safe_col(df, "fr_solar_f").diff(3)

    # DE wind ramp
    df["de_wind_ramp_3h"] = _safe_col(df, "de_wind_f").diff(3)

    # Day-on-day changes
    df["fr_wind_change_24h"] = _safe_col(df, "fr_wind_f") - _safe_col(df, "fr_wind_f").shift(24)
    df["uk_wind_change_24h"] = _safe_col(df, "uk_wind_f") - _safe_col(df, "uk_wind_f").shift(24)
    df["fr_solar_change_24h"] = _safe_col(df, "fr_solar_f") - _safe_col(df, "fr_solar_f").shift(24)
    df["fr_load_change_24h"] = df["fr_load_f"] - df["fr_load_f"].shift(24)
    df["uk_load_change_24h"] = df["uk_load_f"] - df["uk_load_f"].shift(24)

    # Residual load changes
    df["fr_residual_change_24h"] = df["fr_residual_load"] - df["fr_residual_load"].shift(24)
    df["uk_residual_change_24h"] = df["uk_residual_load"] - df["uk_residual_load"].shift(24)

    # Wind volatility
    df["fr_wind_volatility_24h"] = _safe_col(df, "fr_wind_f").rolling(24, min_periods=1).std()
    df["uk_wind_volatility_24h"] = _safe_col(df, "uk_wind_f").rolling(24, min_periods=1).std()

    return df


# ---------------------------------------------------------------------------
# Cat 7 — Nuclear features (11)
# ---------------------------------------------------------------------------

def _add_nuclear_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    nuke_fr = _safe_col(df, "fr_nuclear_avcap_f")
    nuke_uk = _safe_col(df, "uk_nuclear_avcap_f")

    nuke_low = cfg.get("nuclear_low_threshold", 35000)
    nuke_vlow = cfg.get("nuclear_very_low_threshold", 25000)

    # Changes
    df["fr_nuclear_change_24h"] = nuke_fr - nuke_fr.shift(24)
    df["fr_nuclear_change_48h"] = nuke_fr - nuke_fr.shift(48)
    df["fr_nuclear_change_168h"] = nuke_fr - nuke_fr.shift(168)
    df["uk_nuclear_change_24h"] = nuke_uk - nuke_uk.shift(24)

    # Thresholds
    df["fr_nuclear_low"] = (nuke_fr < nuke_low).astype(np.int8)
    df["fr_nuclear_very_low"] = (nuke_fr < nuke_vlow).astype(np.int8)

    # Share of load
    df["fr_nuclear_pct_of_load"] = nuke_fr / df["fr_load_f"].clip(lower=1)
    df["uk_nuclear_pct_of_load"] = nuke_uk / df["uk_load_f"].clip(lower=1)

    # Rolling + deviation
    df["fr_nuclear_rolling_7d_mean"] = nuke_fr.rolling(168, min_periods=1).mean()
    df["fr_nuclear_deviation_from_7d"] = nuke_fr - df["fr_nuclear_rolling_7d_mean"]

    # Ramp magnitude
    df["fr_nuclear_ramp_magnitude"] = df["fr_nuclear_change_24h"].abs()

    return df


# ---------------------------------------------------------------------------
# Cat 8 — Hydro features (9)
# ---------------------------------------------------------------------------

def _add_hydro_features(df: pd.DataFrame) -> pd.DataFrame:
    # fr_hydro_total already computed in cat 4

    # Alpine hydro
    df["ch_hydro_total"] = _safe_col(df, "ch_hydro_res_f") + _safe_col(df, "ch_hydro_ror_f")
    df["at_hydro_total"] = _safe_col(df, "at_hydro_res_f") + _safe_col(df, "at_hydro_ror_f")
    df["alpine_hydro_total"] = df["ch_hydro_total"] + df["at_hydro_total"]

    # Changes
    df["fr_hydro_change_24h"] = df["fr_hydro_total"] - df["fr_hydro_total"].shift(24)
    df["alpine_hydro_change_168h"] = df["alpine_hydro_total"] - df["alpine_hydro_total"].shift(168)

    # Reservoir share
    hydro_total_safe = df["fr_hydro_total"].clip(lower=1)
    df["fr_hydro_res_share"] = _safe_col(df, "fr_hydro_res_f") / hydro_total_safe

    # Nuclear + hydro combined
    df["fr_nuclear_plus_hydro"] = _safe_col(df, "fr_nuclear_avcap_f") + df["fr_hydro_total"]
    df["fr_baseload_surplus"] = df["fr_nuclear_plus_hydro"] - df["fr_load_f"]

    return df


# ---------------------------------------------------------------------------
# Cat 9 — Interconnector features (23)
# ---------------------------------------------------------------------------

def _add_interconnector_features(df: pd.DataFrame) -> pd.DataFrame:
    # --- Aggregated capacities ---
    fr_uk_atc_cols = ["atc_fr-uk-1_f", "atc_fr-uk-2_f", "atc_fr-uk-3_f"]
    uk_fr_atc_cols = ["atc_uk-fr-1_f", "atc_uk-fr-2_f", "atc_uk-fr-3_f"]
    fr_uk_ntc_cols = ["ntc_fr-uk-1_f", "ntc_fr-uk-2_f", "ntc_fr-uk-3_f"]
    uk_fr_ntc_cols = ["ntc_uk-fr-1_f", "ntc_uk-fr-2_f", "ntc_uk-fr-3_f"]

    df["fr_uk_atc_total"] = _safe_sum(df, fr_uk_atc_cols)
    df["uk_fr_atc_total"] = _safe_sum(df, uk_fr_atc_cols)
    df["fr_uk_ntc_total"] = _safe_sum(df, fr_uk_ntc_cols)
    df["uk_fr_ntc_total"] = _safe_sum(df, uk_fr_ntc_cols)

    all_to_uk = ["atc_be-uk_f", "atc_dk1-uk_f"] + fr_uk_atc_cols + ["atc_nl-uk_f"]
    all_from_uk = ["atc_uk-be_f", "atc_uk-dk1_f"] + uk_fr_atc_cols + ["atc_uk-nl_f"]
    df["all_to_uk_atc"] = _safe_sum(df, all_to_uk)
    df["all_from_uk_atc"] = _safe_sum(df, all_from_uk)

    # --- Utilization rates ---
    df["fr_uk_utilization"] = 1 - (df["fr_uk_atc_total"] / df["fr_uk_ntc_total"].clip(lower=1))
    df["uk_fr_utilization"] = 1 - (df["uk_fr_atc_total"] / df["uk_fr_ntc_total"].clip(lower=1))

    be_uk_ntc = _safe_col(df, "ntc_be-uk_f").clip(lower=1)
    nl_uk_ntc = _safe_col(df, "ntc_nl-uk_f").clip(lower=1)
    df["be_uk_utilization"] = 1 - (_safe_col(df, "atc_be-uk_f") / be_uk_ntc)
    df["nl_uk_utilization"] = 1 - (_safe_col(df, "atc_nl-uk_f") / nl_uk_ntc)

    df["max_utilization_to_uk"] = df[
        ["fr_uk_utilization", "be_uk_utilization", "nl_uk_utilization"]
    ].max(axis=1)

    # --- Lagged flow aggregates ---
    fr_to_uk_flows = ["flow_fr-uk-1_la", "flow_fr-uk-2_la", "flow_fr-uk-3_la"]
    uk_to_fr_flows = ["flow_uk-fr-1_la", "flow_uk-fr-2_la", "flow_uk-fr-3_la"]
    df["fr_uk_net_flow_la"] = _safe_sum(df, fr_to_uk_flows) - _safe_sum(df, uk_to_fr_flows)
    df["be_uk_net_flow_la"] = _safe_col(df, "flow_be-uk_la") - _safe_col(df, "flow_uk-be_la")
    df["nl_uk_net_flow_la"] = _safe_col(df, "flow_nl-uk_la") - _safe_col(df, "flow_uk-nl_la")
    df["dk1_uk_net_flow_la"] = _safe_col(df, "flow_dk1-uk_la") - _safe_col(df, "flow_uk-dk1_la")

    df["total_net_import_uk_la"] = (
        df["fr_uk_net_flow_la"]
        + df["be_uk_net_flow_la"]
        + df["nl_uk_net_flow_la"]
        + df["dk1_uk_net_flow_la"]
    )

    # --- Lagged cost aggregates ---
    fr_uk_cost_cols = ["cost_fr-uk-1_la", "cost_fr-uk-2_la", "cost_fr-uk-3_la"]
    uk_fr_cost_cols = ["cost_uk-fr-1_la", "cost_uk-fr-2_la", "cost_uk-fr-3_la"]
    df["fr_uk_avg_cost_la"] = _safe_mean(df, fr_uk_cost_cols)
    df["uk_fr_avg_cost_la"] = _safe_mean(df, uk_fr_cost_cols)
    df["fr_uk_cost_spread_la"] = df["fr_uk_avg_cost_la"] - df["uk_fr_avg_cost_la"]

    # --- Congestion indicators ---
    df["fr_uk_congested"] = (df["fr_uk_utilization"] > 0.9).astype(np.int8)
    df["uk_fr_congested"] = (df["uk_fr_utilization"] > 0.9).astype(np.int8)
    df["any_direction_congested"] = (
        (df["fr_uk_congested"] == 1) | (df["uk_fr_congested"] == 1)
    ).astype(np.int8)

    # ATC change
    df["fr_uk_atc_change_24h"] = df["fr_uk_atc_total"] - df["fr_uk_atc_total"].shift(24)

    return df


# ---------------------------------------------------------------------------
# Cat 10 — Price-based features (19)
# ---------------------------------------------------------------------------

def _add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    # Multi-horizon lags (la is already 24h lagged)
    df["fr_spot_lag_48h"] = _safe_col(df, "fr_spot_la").shift(24)
    df["fr_spot_lag_168h"] = _safe_col(df, "fr_spot_la").shift(144)
    df["uk_spot_lag_48h"] = _safe_col(df, "uk_spot_la").shift(24)
    df["uk_spot_lag_168h"] = _safe_col(df, "uk_spot_la").shift(144)

    # Neighbor lags
    df["de_spot_lag_48h"] = _safe_col(df, "de_spot_la").shift(24)
    df["be_spot_lag_48h"] = _safe_col(df, "be_spot_la").shift(24)
    df["nl_spot_lag_48h"] = _safe_col(df, "nl_spot_la").shift(24)

    # Cross-zone spreads
    df["spread_fr_uk_la"] = _safe_col(df, "fr_spot_la") - _safe_col(df, "uk_spot_la")
    df["spread_fr_de_la"] = _safe_col(df, "fr_spot_la") - _safe_col(df, "de_spot_la")
    df["spread_uk_nl_la"] = _safe_col(df, "uk_spot_la") - _safe_col(df, "nl_spot_la")
    df["spread_uk_be_la"] = _safe_col(df, "uk_spot_la") - _safe_col(df, "be_spot_la")
    df["spread_de_fr_abs_la"] = (_safe_col(df, "fr_spot_la") - _safe_col(df, "de_spot_la")).abs()

    # Continental average
    df["continental_avg_spot_la"] = _safe_mean(
        df, ["de_spot_la", "be_spot_la", "nl_spot_la", "fr_spot_la"]
    )
    df["fr_vs_continental_la"] = _safe_col(df, "fr_spot_la") - df["continental_avg_spot_la"]
    df["uk_vs_continental_la"] = _safe_col(df, "uk_spot_la") - df["continental_avg_spot_la"]

    # Price changes
    df["fr_spot_change_24h_la"] = _safe_col(df, "fr_spot_la") - _safe_col(df, "fr_spot_la").shift(24)
    df["uk_spot_change_24h_la"] = _safe_col(df, "uk_spot_la") - _safe_col(df, "uk_spot_la").shift(24)
    df["fr_spot_change_168h_la"] = _safe_col(df, "fr_spot_la") - _safe_col(df, "fr_spot_la").shift(144)
    df["uk_spot_change_168h_la"] = _safe_col(df, "uk_spot_la") - _safe_col(df, "uk_spot_la").shift(144)

    return df


# ---------------------------------------------------------------------------
# Cat 11 — Interaction features (13)
# ---------------------------------------------------------------------------

def _add_interaction_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    # Thermal need x gas (EDA: beats individual features)
    df["fr_thermal_need_x_gas"] = df["fr_thermal_need_pos"] * _safe_col(df, "fr_gas")
    df["uk_thermal_need_x_gas"] = df["uk_thermal_need_pos"] * _safe_col(df, "uk_gas")

    # Baseload gap x gas
    df["fr_baseload_gap_x_gas"] = df["fr_baseload_gap_pos"] * _safe_col(df, "fr_gas")

    # UK wind gap x gas
    uk_wind_gap = (df["uk_load_f"] - _safe_col(df, "uk_wind_f")).clip(lower=0)
    df["uk_wind_gap_x_gas"] = uk_wind_gap * _safe_col(df, "uk_gas")

    # Residual x spark
    df["fr_residual_x_spark"] = df["fr_residual_load"] * df["fr_spark_spread"]
    df["uk_residual_x_spark"] = df["uk_residual_load"] * df["uk_spark_spread"]

    # Wind x hour
    df["fr_wind_x_hour"] = _safe_col(df, "fr_wind_f") * df["hour_sin"]
    df["uk_wind_x_hour"] = _safe_col(df, "uk_wind_f") * df["hour_sin"]

    # Solar x load
    df["fr_solar_x_load"] = _safe_col(df, "fr_solar_f") * df["fr_load_f"]

    # Nuclear x gas
    df["fr_nuclear_x_gas"] = _safe_col(df, "fr_nuclear_avcap_f") * _safe_col(df, "fr_gas")

    # Wind x gas
    df["uk_wind_x_gas"] = _safe_col(df, "uk_wind_f") * _safe_col(df, "uk_gas")

    # Congestion x spark difference
    df["fr_congestion_x_spark_diff"] = df["fr_uk_utilization"] * (
        df["fr_spark_spread"] - df["uk_spark_spread"]
    )

    # Thermal need x nuclear change
    df["fr_thermal_need_x_nuclear_change"] = (
        df["fr_thermal_need"] * df["fr_nuclear_change_24h"].abs()
    )

    # Stress index: thermal need × (1 - renewable penetration) — v8 addition
    df["fr_stress_index"] = df["fr_thermal_need_pos"] * (1 - df["fr_renewable_pen"].clip(0, 1))
    df["uk_stress_index"] = df["uk_thermal_need_pos"] * (1 - df["uk_renewable_pen"].clip(0, 1))

    return df


# ---------------------------------------------------------------------------
# Cat 12 — Regional / Neighbor features (11)
# ---------------------------------------------------------------------------

def _add_regional_features(df: pd.DataFrame) -> pd.DataFrame:
    # Continental aggregates
    df["continental_load_total"] = _safe_sum(
        df, ["fr_load_f", "de_load_f", "be_load_f", "nl_load_f"]
    )
    df["continental_wind_total"] = _safe_sum(
        df, ["fr_wind_f", "de_wind_f", "be_wind_f", "nl_wind_f"]
    )
    df["continental_solar_total"] = _safe_sum(
        df, ["fr_solar_f", "de_solar_f", "be_solar_f", "nl_solar_f"]
    )
    df["continental_residual_load"] = (
        df["continental_load_total"]
        - df["continental_wind_total"]
        - df["continental_solar_total"]
    )

    # Nordic wind
    df["nordic_wind_total"] = _safe_sum(df, ["dk1_wind_f", "dk2_wind_f"])

    # Iberian
    df["iberian_load"] = _safe_col(df, "es_load_f")
    df["iberian_wind"] = _safe_col(df, "es_wind_f")
    df["iberian_solar"] = _safe_col(df, "es_solar_f")

    # DE load change
    df["de_load_change_24h"] = df["de_load_f"] - df["de_load_f"].shift(24)

    # Benelux combined
    df["be_nl_combined_load"] = _safe_sum(df, ["be_load_f", "nl_load_f"])

    # Alpine + N.Italy
    df["at_ch_combined_load"] = _safe_sum(df, ["at_load_f", "itn_load_f"])

    return df


# ---------------------------------------------------------------------------
# Cat 13 — River temperature features (12)
# ---------------------------------------------------------------------------

def _add_river_temperature_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    fr_thresh = cfg.get("river_hot_threshold_fr", 25.0)
    de_thresh = cfg.get("river_hot_threshold_de", 23.0)

    rhone = _safe_col(df, "fr_river_temp_rhone_lyon_f")
    rhine = _safe_col(df, "fr_river_temp_rhine_rheinfelden_f")
    danube_don = _safe_col(df, "de_river_temp_danube_donauworth_f")
    danube_ing = _safe_col(df, "de_river_temp_danube_ingolstadt_f")

    # Binary thresholds
    df["fr_rhone_hot"] = (rhone > fr_thresh).astype(np.int8)
    df["fr_rhine_hot"] = (rhine > fr_thresh).astype(np.int8)
    df["de_danube_hot_donauworth"] = (danube_don > de_thresh).astype(np.int8)
    df["de_danube_hot_ingolstadt"] = (danube_ing > de_thresh).astype(np.int8)

    # Any hot
    df["fr_any_river_hot"] = ((df["fr_rhone_hot"] == 1) | (df["fr_rhine_hot"] == 1)).astype(np.int8)
    df["de_any_river_hot"] = (
        (df["de_danube_hot_donauworth"] == 1) | (df["de_danube_hot_ingolstadt"] == 1)
    ).astype(np.int8)

    # Continuous excess
    df["fr_rhone_temp_excess"] = (rhone - fr_thresh).clip(lower=0)
    df["fr_rhine_temp_excess"] = (rhine - fr_thresh).clip(lower=0)

    # Max river temp
    df["fr_max_river_temp"] = pd.concat([rhone, rhine], axis=1).max(axis=1)
    df["de_max_river_temp"] = pd.concat([danube_don, danube_ing], axis=1).max(axis=1)

    # Temp change 24h
    df["fr_river_temp_change_24h"] = rhone - rhone.shift(24)

    # Interaction: hot river AND low nuclear
    df["fr_hot_river_x_nuclear_low"] = df["fr_any_river_hot"] * df["fr_nuclear_low"]

    return df


# ---------------------------------------------------------------------------
# Cat 14 — Non-linear transforms (11)
# ---------------------------------------------------------------------------

def _add_nonlinear_transforms(df: pd.DataFrame) -> pd.DataFrame:
    # Squared residual loads
    df["fr_residual_load_squared"] = df["fr_residual_load"] ** 2
    df["uk_residual_load_squared"] = df["uk_residual_load"] ** 2

    # Super-linear thermal need
    df["fr_thermal_need_cubed_pos"] = df["fr_thermal_need_pos"] ** 1.5

    # Wind penetration squared
    df["uk_wind_pen_squared"] = df["uk_wind_pen"] ** 2

    # Log spark spread
    df["fr_spark_spread_log"] = np.log1p(df["fr_spark_spread"].clip(lower=0))
    df["uk_spark_spread_log"] = np.log1p(df["uk_spark_spread"].clip(lower=0))

    # Log lagged price (sign-preserving)
    fr_la = _safe_col(df, "fr_spot_la")
    uk_la = _safe_col(df, "uk_spot_la")
    df["fr_spot_la_log"] = np.sign(fr_la) * np.log1p(fr_la.abs())
    df["uk_spot_la_log"] = np.sign(uk_la) * np.log1p(uk_la.abs())

    # Sqrt gas
    df["fr_gas_sqrt"] = np.sqrt(_safe_col(df, "fr_gas").clip(lower=0))

    # Clipped wind (physical capacity ceiling)
    df["fr_wind_f_clipped"] = _safe_col(df, "fr_wind_f").clip(upper=22000)
    df["uk_wind_f_clipped"] = _safe_col(df, "uk_wind_f").clip(upper=20000)

    return df


# ---------------------------------------------------------------------------
# Cat 15 — Rolling statistics (24)
# ---------------------------------------------------------------------------

def _add_rolling_statistics(df: pd.DataFrame) -> pd.DataFrame:
    fr_la = _safe_col(df, "fr_spot_la")
    uk_la = _safe_col(df, "uk_spot_la")

    # Price rolling 24h
    df["fr_spot_la_roll_24h_mean"] = fr_la.rolling(24, min_periods=1).mean()
    df["fr_spot_la_roll_24h_std"] = fr_la.rolling(24, min_periods=1).std()
    df["uk_spot_la_roll_24h_mean"] = uk_la.rolling(24, min_periods=1).mean()
    df["uk_spot_la_roll_24h_std"] = uk_la.rolling(24, min_periods=1).std()

    # Price rolling 168h (weekly)
    df["fr_spot_la_roll_168h_mean"] = fr_la.rolling(168, min_periods=1).mean()
    df["fr_spot_la_roll_168h_std"] = fr_la.rolling(168, min_periods=1).std()
    df["uk_spot_la_roll_168h_mean"] = uk_la.rolling(168, min_periods=1).mean()
    df["uk_spot_la_roll_168h_std"] = uk_la.rolling(168, min_periods=1).std()

    # Price rolling 24h min/max/range
    df["fr_spot_la_roll_24h_min"] = fr_la.rolling(24, min_periods=1).min()
    df["fr_spot_la_roll_24h_max"] = fr_la.rolling(24, min_periods=1).max()
    df["uk_spot_la_roll_24h_min"] = uk_la.rolling(24, min_periods=1).min()
    df["uk_spot_la_roll_24h_max"] = uk_la.rolling(24, min_periods=1).max()
    df["fr_spot_la_roll_24h_range"] = df["fr_spot_la_roll_24h_max"] - df["fr_spot_la_roll_24h_min"]
    df["uk_spot_la_roll_24h_range"] = df["uk_spot_la_roll_24h_max"] - df["uk_spot_la_roll_24h_min"]

    # Load/wind rolling
    df["uk_wind_roll_24h_mean"] = _safe_col(df, "uk_wind_f").rolling(24, min_periods=1).mean()
    df["fr_load_roll_168h_mean"] = df["fr_load_f"].rolling(168, min_periods=1).mean()
    df["uk_load_roll_168h_mean"] = df["uk_load_f"].rolling(168, min_periods=1).mean()

    # Gas/emission rolling
    df["fr_gas_roll_168h_mean"] = _safe_col(df, "fr_gas").rolling(168, min_periods=1).mean()
    df["uk_gas_roll_168h_mean"] = _safe_col(df, "uk_gas").rolling(168, min_periods=1).mean()
    df["eu_emission_roll_168h_mean"] = _safe_col(df, "eu_emission").rolling(168, min_periods=1).mean()

    # Price rolling 336h (bi-weekly) — v8 addition
    df["fr_spot_la_roll_336h_mean"] = fr_la.rolling(336, min_periods=48).mean()
    df["fr_spot_la_roll_336h_std"] = fr_la.rolling(336, min_periods=48).std()
    df["uk_spot_la_roll_336h_mean"] = uk_la.rolling(336, min_periods=48).mean()
    df["uk_spot_la_roll_336h_std"] = uk_la.rolling(336, min_periods=48).std()

    # Deviations from rolling mean
    df["fr_spot_la_deviation_24h"] = fr_la - df["fr_spot_la_roll_24h_mean"]
    df["uk_spot_la_deviation_24h"] = uk_la - df["uk_spot_la_roll_24h_mean"]
    df["fr_spot_la_deviation_168h"] = fr_la - df["fr_spot_la_roll_168h_mean"]
    df["uk_spot_la_deviation_168h"] = uk_la - df["uk_spot_la_roll_168h_mean"]

    # Load surprise — deviation from 7-day mean (demand surprise) — v8 addition
    for prefix in ["fr", "uk"]:
        load_col = f"{prefix}_load_f"
        if load_col in df.columns:
            load_7d = df[load_col].rolling(168, min_periods=24).mean()
            df[f"{prefix}_load_surprise"] = (df[load_col] - load_7d) / load_7d.clip(lower=1)

    return df


# ---------------------------------------------------------------------------
# Cat 16 — Momentum / Trend features (8)
# ---------------------------------------------------------------------------

def _add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    fr_la = _safe_col(df, "fr_spot_la")
    uk_la = _safe_col(df, "uk_spot_la")

    # Price acceleration (second derivative)
    fr_mom = df.get("fr_spot_change_24h_la", fr_la - fr_la.shift(24))
    uk_mom = df.get("uk_spot_change_24h_la", uk_la - uk_la.shift(24))
    df["fr_price_acceleration"] = fr_mom - fr_mom.shift(24)
    df["uk_price_acceleration"] = uk_mom - uk_mom.shift(24)

    # Gas momentum
    df["fr_gas_momentum_24h"] = _safe_col(df, "fr_gas") - _safe_col(df, "fr_gas").shift(24)
    df["uk_gas_momentum_24h"] = _safe_col(df, "uk_gas") - _safe_col(df, "uk_gas").shift(24)

    # Nuclear trend (short vs medium term)
    nuke = _safe_col(df, "fr_nuclear_avcap_f")
    df["fr_nuclear_trend_3d"] = (
        nuke.rolling(72, min_periods=1).mean()
        - nuke.rolling(168, min_periods=1).mean()
    )

    # Exponentially weighted means
    df["fr_spot_la_ewm_24h"] = fr_la.ewm(span=24, min_periods=1).mean()
    df["uk_spot_la_ewm_24h"] = uk_la.ewm(span=24, min_periods=1).mean()

    # Spread momentum
    spread = df.get("spread_fr_uk_la", fr_la - uk_la)
    df["spread_fr_uk_momentum"] = spread - spread.shift(24)

    return df


# ---------------------------------------------------------------------------
# Cat 17 — Advanced Supply/Demand (scarcity, security margin, residual v2)
# ---------------------------------------------------------------------------

def _add_advanced_supply_demand(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    # Residual load v2: subtract hydro run-of-river (must-run)
    df["fr_residual_load_v2"] = (
        df["fr_load_f"]
        - _safe_col(df, "fr_wind_f")
        - _safe_col(df, "fr_solar_f")
        - _safe_col(df, "fr_hydro_ror_f")
    )

    # Security margin: (nuclear + gas_cap) - residual_load  [MW headroom]
    fr_dispatchable = _safe_col(df, "fr_nuclear_avcap_f") + _safe_col(df, "fr_gas_avcap_f")
    uk_dispatchable = _safe_col(df, "uk_nuclear_avcap_f") + _safe_col(df, "uk_gas_avcap_f")
    df["fr_security_margin"] = fr_dispatchable - df["fr_residual_load"]
    df["uk_security_margin"] = uk_dispatchable - df["uk_residual_load"]

    # Scarcity ratio: residual / (nuclear + gas_cap) — CRITICAL for convexity
    scarcity_crit = cfg.get("scarcity_critical_threshold", 0.85)
    scarcity_ext = cfg.get("scarcity_extreme_threshold", 0.95)

    df["fr_scarcity_ratio"] = df["fr_residual_load"] / fr_dispatchable.clip(lower=1)
    df["uk_scarcity_ratio"] = df["uk_residual_load"] / uk_dispatchable.clip(lower=1)

    df["fr_scarcity_critical"] = (df["fr_scarcity_ratio"] > scarcity_crit).astype(np.int8)
    df["uk_scarcity_critical"] = (df["uk_scarcity_ratio"] > scarcity_crit).astype(np.int8)
    df["fr_scarcity_extreme"] = (df["fr_scarcity_ratio"] > scarcity_ext).astype(np.int8)
    df["uk_scarcity_extreme"] = (df["uk_scarcity_ratio"] > scarcity_ext).astype(np.int8)

    return df


# ---------------------------------------------------------------------------
# Cat 18 — Load & Residual ramps 1h/3h
# ---------------------------------------------------------------------------

def _add_load_residual_ramps(df: pd.DataFrame) -> pd.DataFrame:
    # Load ramps
    df["fr_load_ramp_1h"] = df["fr_load_f"].diff(1)
    df["fr_load_ramp_3h"] = df["fr_load_f"].diff(3)
    df["uk_load_ramp_1h"] = df["uk_load_f"].diff(1)
    df["uk_load_ramp_3h"] = df["uk_load_f"].diff(3)

    # Residual load ramps (instantaneous pressure on thermal fleet)
    df["fr_residual_ramp_1h"] = df["fr_residual_load"].diff(1)
    df["fr_residual_ramp_3h"] = df["fr_residual_load"].diff(3)
    df["uk_residual_ramp_1h"] = df["uk_residual_load"].diff(1)
    df["uk_residual_ramp_3h"] = df["uk_residual_load"].diff(3)

    return df


# ---------------------------------------------------------------------------
# Cat 19 — Multi-efficiency spark spreads (OCGT/CCGT)
# ---------------------------------------------------------------------------

def _add_multi_efficiency_spark(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    em = cfg.get("emission_factor", 0.37)
    ocgt_eff = cfg.get("ocgt_efficiency", 0.40)
    ccgt_eff = cfg.get("ccgt_efficiency", 0.55)

    # OCGT: old, inefficient peakers — higher marginal cost
    df["fr_spark_ocgt"] = _safe_col(df, "fr_gas") / ocgt_eff + _safe_col(df, "eu_emission") * em
    df["uk_spark_ocgt"] = _safe_col(df, "uk_gas") / ocgt_eff + _safe_col(df, "uk_emission") * em

    # CCGT: modern combined-cycle — lower marginal cost
    df["fr_spark_ccgt"] = _safe_col(df, "fr_gas") / ccgt_eff + _safe_col(df, "eu_emission") * em
    df["uk_spark_ccgt"] = _safe_col(df, "uk_gas") / ccgt_eff + _safe_col(df, "uk_emission") * em

    return df


# ---------------------------------------------------------------------------
# Cat 20 — Advanced interconnection (flow/ATC ratio, unused capacity)
# ---------------------------------------------------------------------------

def _add_advanced_interconnection(df: pd.DataFrame) -> pd.DataFrame:
    # Flow over ATC: actual cable pressure (lagged flow / forecasted ATC)
    fr_to_uk_flows = ["flow_fr-uk-1_la", "flow_fr-uk-2_la", "flow_fr-uk-3_la"]
    fr_uk_flow_total = _safe_sum(df, fr_to_uk_flows)
    df["fr_uk_flow_over_atc"] = fr_uk_flow_total / df["fr_uk_atc_total"].clip(lower=1)

    # Unused capacity per interconnector (MW still available)
    df["fr_uk_unused_capacity"] = df["fr_uk_atc_total"] - fr_uk_flow_total.clip(lower=0)
    df["be_uk_unused_capacity"] = (
        _safe_col(df, "atc_be-uk_f") - _safe_col(df, "flow_be-uk_la").clip(lower=0)
    )
    df["nl_uk_unused_capacity"] = (
        _safe_col(df, "atc_nl-uk_f") - _safe_col(df, "flow_nl-uk_la").clip(lower=0)
    )

    # Total unused capacity to UK
    df["total_unused_capacity_to_uk"] = (
        df["fr_uk_unused_capacity"]
        + df["be_uk_unused_capacity"]
        + df["nl_uk_unused_capacity"]
    )

    return df


# ---------------------------------------------------------------------------
# Cat 21 — Z-Scores & anomalies (14-day rolling z-scores)
# ---------------------------------------------------------------------------

def _add_zscore_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    window = 336  # 14 days * 24 hours

    # Residual load z-scores — panic detector
    for prefix, col in [("fr", "fr_residual_load"), ("uk", "uk_residual_load")]:
        mean_14d = df[col].rolling(window, min_periods=1).mean()
        std_14d = df[col].rolling(window, min_periods=1).std()
        df[f"{prefix}_residual_zscore_14d"] = (df[col] - mean_14d) / std_14d.clip(lower=1)

    # Load z-scores
    for prefix, col in [("fr", "fr_load_f"), ("uk", "uk_load_f")]:
        mean_14d = df[col].rolling(window, min_periods=1).mean()
        std_14d = df[col].rolling(window, min_periods=1).std()
        df[f"{prefix}_load_zscore_14d"] = (df[col] - mean_14d) / std_14d.clip(lower=1)

    # Wind z-scores
    for prefix, raw_col in [("fr", "fr_wind_f"), ("uk", "uk_wind_f")]:
        wind = _safe_col(df, raw_col)
        mean_14d = wind.rolling(window, min_periods=1).mean()
        std_14d = wind.rolling(window, min_periods=1).std()
        df[f"{prefix}_wind_zscore_14d"] = (wind - mean_14d) / std_14d.clip(lower=1)

    # Lag reliability ratio: spot_la / spot_lag_168h — how stale is the weekly lag?
    df["fr_lag_reliability_ratio"] = (
        _safe_col(df, "fr_spot_la") / df["fr_spot_lag_168h"].clip(lower=0.1)
    )
    df["uk_lag_reliability_ratio"] = (
        _safe_col(df, "uk_spot_la") / df["uk_spot_lag_168h"].clip(lower=0.1)
    )

    return df


# ---------------------------------------------------------------------------
# Cat 22 — Stochastic / SDE signals (jump count, vol ratio, mean reversion)
# ---------------------------------------------------------------------------

def _add_stochastic_signals(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    jump_thresh = cfg.get("jump_threshold", 50.0)

    fr_la = _safe_col(df, "fr_spot_la")
    uk_la = _safe_col(df, "uk_spot_la")

    # Price changes (absolute)
    fr_abs_change = fr_la.diff(1).abs()
    uk_abs_change = uk_la.diff(1).abs()

    # Jump indicators
    fr_is_jump = (fr_abs_change > jump_thresh).astype(float)
    uk_is_jump = (uk_abs_change > jump_thresh).astype(float)

    # Jump count: rolling sum of jumps over 24h and 48h
    df["fr_jump_count_24h"] = fr_is_jump.rolling(24, min_periods=1).sum()
    df["uk_jump_count_24h"] = uk_is_jump.rolling(24, min_periods=1).sum()
    df["fr_jump_count_48h"] = fr_is_jump.rolling(48, min_periods=1).sum()
    df["uk_jump_count_48h"] = uk_is_jump.rolling(48, min_periods=1).sum()

    # Jump magnitude: mean absolute change conditional on being a jump
    # Approximation: (sum of abs changes when > threshold) / (count of jumps)
    fr_jump_val = fr_abs_change.where(fr_abs_change > jump_thresh, 0.0)
    uk_jump_val = uk_abs_change.where(uk_abs_change > jump_thresh, 0.0)
    df["fr_jump_magnitude_24h"] = (
        fr_jump_val.rolling(24, min_periods=1).sum()
        / df["fr_jump_count_24h"].clip(lower=1)
    )
    df["uk_jump_magnitude_24h"] = (
        uk_jump_val.rolling(24, min_periods=1).sum()
        / df["uk_jump_count_24h"].clip(lower=1)
    )

    # Vol ratio: short-term vol / long-term vol (stress vs calm)
    fr_std_24h = df["fr_spot_la_roll_24h_std"]
    fr_std_168h = df["fr_spot_la_roll_168h_std"]
    uk_std_24h = df["uk_spot_la_roll_24h_std"]
    uk_std_168h = df["uk_spot_la_roll_168h_std"]

    df["fr_vol_ratio"] = fr_std_24h / fr_std_168h.clip(lower=0.1)
    df["uk_vol_ratio"] = uk_std_24h / uk_std_168h.clip(lower=0.1)

    # Mean reversion strength: deviation_168h / std_168h (Ornstein-Uhlenbeck signal)
    df["fr_mean_reversion_strength"] = (
        df["fr_spot_la_deviation_168h"] / fr_std_168h.clip(lower=0.1)
    )
    df["uk_mean_reversion_strength"] = (
        df["uk_spot_la_deviation_168h"] / uk_std_168h.clip(lower=0.1)
    )

    return df


# ---------------------------------------------------------------------------
# Cat 23 — Improved transforms (asinh replaces sign*log1p)
# ---------------------------------------------------------------------------

def _add_improved_transforms(df: pd.DataFrame) -> pd.DataFrame:
    # asinh: handles negatives natively, smooth, similar to log for large values
    # Replaces sign*log1p which has discontinuity at 0
    df["fr_asinh_spot_la"] = np.arcsinh(_safe_col(df, "fr_spot_la"))
    df["uk_asinh_spot_la"] = np.arcsinh(_safe_col(df, "uk_spot_la"))
    df["fr_asinh_spark"] = np.arcsinh(df["fr_spark_spread"])
    df["uk_asinh_spark"] = np.arcsinh(df["uk_spark_spread"])

    return df


# ---------------------------------------------------------------------------
# Cat 24 — Nuclear shortfall (installed capacity - available capacity)
# ---------------------------------------------------------------------------

def _add_nuclear_shortfall(df: pd.DataFrame) -> pd.DataFrame:
    """Gap between max observed nuclear and current availability (MW missing)."""
    for prefix in ("fr", "uk"):
        col = f"{prefix}_nuclear_avcap_f"
        nuke = _safe_col(df, col)
        # Expanding max: the highest availability seen so far in the data
        nuke_max = nuke.expanding(min_periods=1).max()
        df[f"{prefix}_nuclear_shortfall"] = nuke_max - nuke

    return df


# ---------------------------------------------------------------------------
# Cat 25 — ATC/NTC ratios per individual cable
# ---------------------------------------------------------------------------

def _add_atc_ratios_per_cable(df: pd.DataFrame) -> pd.DataFrame:
    """Per-cable ATC/NTC ratio. Near 0 = congested, 1 = free capacity."""
    cables = [
        ("atc_fr-uk-1_f", "ntc_fr-uk-1_f"),
        ("atc_fr-uk-2_f", "ntc_fr-uk-2_f"),
        ("atc_fr-uk-3_f", "ntc_fr-uk-3_f"),
        ("atc_uk-fr-1_f", "ntc_uk-fr-1_f"),
        ("atc_uk-fr-2_f", "ntc_uk-fr-2_f"),
        ("atc_uk-fr-3_f", "ntc_uk-fr-3_f"),
    ]
    for atc_col, ntc_col in cables:
        atc = _safe_col(df, atc_col)
        ntc = _safe_col(df, ntc_col).clip(lower=1)
        # Name: atc_fr-uk-1_f -> atc_fr-uk-1_f_ratio
        df[f"{atc_col}_ratio"] = atc / ntc

    return df


# ---------------------------------------------------------------------------
# Cat 26 — Market-specific features (SDAC continental + N2EX dependency)
# ---------------------------------------------------------------------------

def _add_market_specific_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Features grounded in SDAC/EUPHEMIA (FR) and N2EX (UK) market mechanisms.

    Validated by both empirical correlation analysis and market theory:
    - FR is a continental price-taker via SDAC → needs European merit order signals
    - UK is an island dependent on imports → needs dependency/self-sufficiency signals
    """
    eff = cfg.get("gas_efficiency", 0.50)
    em = cfg.get("emission_factor", 0.37)

    # ------------------------------------------------------------------
    # A. Gas spread NBP vs TTF (r=-0.74 FR, r=-0.62 UK)
    #    When UK gas diverges from continental TTF, both markets react.
    #    Captures cross-market fuel cost differential.
    # ------------------------------------------------------------------
    df["gas_spread_uk_eu"] = _safe_col(df, "uk_gas") - _safe_col(df, "nl_gas")

    # ------------------------------------------------------------------
    # B. Continental thermal floor — European consensus marginal cost
    #    avg(DE, FR, NL thermal floors) — r=0.902 vs fr_spot,
    #    beats any individual thermal floor (DE=0.898, FR=0.888).
    # ------------------------------------------------------------------
    de_floor = _safe_col(df, "de_gas") / eff + _safe_col(df, "eu_emission") * em
    fr_floor = df["fr_spark_spread"]  # already = gas/eff + emission*em
    nl_floor = df["nl_spark_spread"]
    df["continent_thermal_floor"] = (de_floor + fr_floor + nl_floor) / 3

    # ------------------------------------------------------------------
    # C. UK import ratio — dependency metric (r=-0.42 UK)
    #    What fraction of UK load is met by imports?
    #    High import ratio + low vol = stable (cheap continental power).
    #    Low import ratio = self-reliant → gas sets price.
    # ------------------------------------------------------------------
    df["uk_import_ratio"] = df["total_net_import_uk_la"] / df["uk_load_f"].clip(lower=1)

    # ------------------------------------------------------------------
    # D. FR export ratio — pression relative (r=-0.62 FR)
    #    FR net exports as fraction of FR load.
    #    When FR exports heavily → surplus → low prices.
    # ------------------------------------------------------------------
    df["fr_export_ratio"] = df["fr_uk_net_flow_la"] / df["fr_load_f"].clip(lower=1)

    # ------------------------------------------------------------------
    # E. DE wind thresholds — cliff effects on FR price
    #    Empirically validated: CLIFF at 30% (-27 EUR), big drop at 50%.
    #    DE wind pushes continental merit order down → FR price drops.
    # ------------------------------------------------------------------
    de_wind_pen = _safe_col(df, "de_wind_f") / df["de_load_f"].clip(lower=1)
    df["de_wind_high"] = (de_wind_pen > 0.30).astype(np.int8)
    df["de_wind_very_high"] = (de_wind_pen > 0.50).astype(np.int8)

    # ------------------------------------------------------------------
    # F. UK wind share of flexible demand (from N2EX merit order theory)
    #    Wind / (load - nuclear - biomass) — what % of *dispatchable* demand
    #    is covered by wind? More accurate than wind/total_load because
    #    baseload (nuclear+biomass) runs regardless.
    # ------------------------------------------------------------------
    uk_baseload = (
        _safe_col(df, "uk_nuclear_avcap_f") + _safe_col(df, "uk_biomass_avcap_f")
    )
    uk_flexible_demand = (df["uk_load_f"] - uk_baseload).clip(lower=1)
    df["uk_wind_share_flexible"] = _safe_col(df, "uk_wind_f") / uk_flexible_demand

    # ------------------------------------------------------------------
    # G. FR-DE decoupling indicator (from SDAC/EUPHEMIA theory)
    #    When |spread| > 10 EUR → interconnectors congested → FR prices
    #    set by local fundamentals, not continental merit order.
    # ------------------------------------------------------------------
    fr_de_spread_abs = (
        _safe_col(df, "fr_spot_la") - _safe_col(df, "de_spot_la")
    ).abs()
    df["fr_de_decoupled"] = (fr_de_spread_abs > 10.0).astype(np.int8)

    # ------------------------------------------------------------------
    # H. Scarcity-weighted marginal cost (from merit order theory)
    #    Interpolates between CCGT (efficient, low scarcity) and
    #    OCGT (peaker, high scarcity) based on system tightness.
    # ------------------------------------------------------------------
    fr_scarcity = df["fr_scarcity_ratio"].clip(0, 1)
    uk_scarcity = df["uk_scarcity_ratio"].clip(0, 1)
    fr_ocgt_w = ((fr_scarcity - 0.5) / 0.4).clip(0, 1)
    uk_ocgt_w = ((uk_scarcity - 0.5) / 0.4).clip(0, 1)

    df["fr_merit_order_cost"] = (
        (1 - fr_ocgt_w) * df["fr_spark_ccgt"] + fr_ocgt_w * df["fr_spark_ocgt"]
    )
    df["uk_merit_order_cost"] = (
        (1 - uk_ocgt_w) * df["uk_spark_ccgt"] + uk_ocgt_w * df["uk_spark_ocgt"]
    )

    # ------------------------------------------------------------------
    # I. Intraday price anchors h3/h19 (Ziel & Weron 2018, EPF literature)
    #    h3 = overnight trough (baseload price), h19 = evening peak.
    #    Amplitude = steepness of merit order that day.
    # ------------------------------------------------------------------
    fr_la = _safe_col(df, "fr_spot_la")
    uk_la = _safe_col(df, "uk_spot_la")
    h = df["hour"]

    df["fr_spot_la_h3"] = fr_la.where(h == 3).ffill()
    df["uk_spot_la_h3"] = uk_la.where(h == 3).ffill()
    df["fr_spot_la_h19"] = fr_la.where(h == 19).ffill()
    df["uk_spot_la_h19"] = uk_la.where(h == 19).ffill()
    df["fr_intraday_amplitude"] = df["fr_spot_la_h19"] - df["fr_spot_la_h3"]
    df["uk_intraday_amplitude"] = df["uk_spot_la_h19"] - df["uk_spot_la_h3"]

    # ------------------------------------------------------------------
    # J. Dark doldrums — winter evening + no wind + no solar
    #    Physical condition causing worst price spikes in both markets.
    # ------------------------------------------------------------------
    is_winter = df["month"].isin([11, 12, 1, 2]).astype(float)
    is_evening = df["is_evening_peak"].astype(float)
    df["dark_doldrums_fr"] = (
        is_winter * is_evening
        * (1 - _safe_col(df, "fr_wind_pen"))
        * (1 - _safe_col(df, "fr_solar_pen"))
    )
    df["dark_doldrums_uk"] = (
        is_winter * is_evening * (1 - _safe_col(df, "uk_wind_pen"))
    )

    return df


# ---------------------------------------------------------------------------
# Cat 27 — Advanced price formation signals
# ---------------------------------------------------------------------------

def _add_price_formation_signals(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Empirically validated features with high partial correlations.

    Each feature was tested for INCREMENTAL value beyond existing features:
    - nuclear_shortfall × gas: partial r=0.768 FR / 0.763 UK (vs shortfall alone)
    - implied_renewable_surplus: partial r=0.547 UK (vs uk_spot_la)
    - cheapest_import_price: partial r=0.433 UK (vs fr_spot_la)
    - net_capacity_cost: partial r=-0.278 FR (vs spark_spread)
    """
    eff = cfg.get("gas_efficiency", 0.50)
    em = cfg.get("emission_factor", 0.37)

    # ------------------------------------------------------------------
    # A. Nuclear shortfall × gas price interaction
    #    Shortfall alone (Cat 24) has r=-0.48 FR. But when gas is expensive,
    #    each MW of nuclear offline costs MORE. The interaction captures this
    #    convexity: partial r=0.768 after controlling for shortfall alone.
    # ------------------------------------------------------------------
    for prefix in ("fr", "uk"):
        shortfall = df[f"{prefix}_nuclear_shortfall"]
        gas = _safe_col(df, "nl_gas") if prefix == "fr" else _safe_col(df, "uk_gas")
        df[f"{prefix}_nuke_shortfall_x_gas"] = shortfall * gas

    # ------------------------------------------------------------------
    # B. Implied renewable surplus (spark_spread − spot_la)
    #    Measures how much renewables + imports pushed yesterday's price
    #    below gas marginal cost. Partial r=0.547 UK beyond uk_spot_la.
    #    Positive = renewables abundant, Negative = scarcity.
    # ------------------------------------------------------------------
    for prefix in ("fr", "uk"):
        gas_col = "nl_gas" if prefix == "fr" else "uk_gas"
        spark = _safe_col(df, gas_col) / eff + _safe_col(df, "eu_emission") * em
        spot_la = _safe_col(df, f"{prefix}_spot_la")
        df[f"{prefix}_implied_re_surplus"] = spark - spot_la

    # ------------------------------------------------------------------
    # C. Multi-source UK import pricing
    #    UK imports from FR, BE, NL, DK1. The cheapest available import
    #    is the effective price floor for UK. r=0.895 raw, partial r=0.433
    #    after controlling for fr_spot_la. Source switches dynamically:
    #    FR 32%, NL 32%, BE 26%, DK1 10%.
    # ------------------------------------------------------------------
    fr_uk_cost = _safe_mean(df, ["cost_fr-uk-1_la", "cost_fr-uk-2_la", "cost_fr-uk-3_la"])
    be_uk_cost = _safe_col(df, "cost_be-uk_la")
    nl_uk_cost = _safe_col(df, "cost_nl-uk_la")
    dk1_uk_cost = _safe_col(df, "cost_dk1-uk_la")

    import_fr = _safe_col(df, "fr_spot_la") + fr_uk_cost
    import_be = _safe_col(df, "be_spot_la") + be_uk_cost
    import_nl = _safe_col(df, "nl_spot_la") + nl_uk_cost
    import_dk1 = _safe_col(df, "dk1_spot_la") + dk1_uk_cost

    import_stack = pd.concat(
        [import_fr, import_be, import_nl, import_dk1], axis=1
    )
    df["uk_cheapest_import"] = import_stack.min(axis=1)
    df["uk_import_price_range"] = import_stack.max(axis=1) - import_stack.min(axis=1)

    # ------------------------------------------------------------------
    # D. Net capacity auction cost (FR→UK − UK→FR)
    #    Market-derived forward signal: traders price in expected spread.
    #    r=-0.593 FR, partial r=-0.278 beyond spark_spread.
    #    Positive = market expects UK > FR (FR is cheap).
    # ------------------------------------------------------------------
    fr_to_uk_cost = _safe_mean(df, ["cost_fr-uk-1_la", "cost_fr-uk-2_la", "cost_fr-uk-3_la"])
    uk_to_fr_cost = _safe_mean(df, ["cost_uk-fr-1_la", "cost_uk-fr-2_la", "cost_uk-fr-3_la"])
    df["net_capacity_cost_fr_uk"] = fr_to_uk_cost - uk_to_fr_cost

    # ------------------------------------------------------------------
    # E. UK fossil/import need
    #    How much UK load MUST come from gas or imports after subtracting
    #    all zero-marginal-cost generation. r=0.393 UK.
    # ------------------------------------------------------------------
    uk_zero_mc = (
        _safe_col(df, "uk_nuclear_avcap_f")
        + _safe_col(df, "uk_biomass_avcap_f")
        + _safe_col(df, "uk_wind_f")
        + _safe_col(df, "uk_solar_f")
    )
    df["uk_fossil_or_import_need"] = (df["uk_load_f"] - uk_zero_mc).clip(lower=0)

    return df


# ---------------------------------------------------------------------------
# Cat 28 — FR continent territory features
# ---------------------------------------------------------------------------

def _add_fr_continent_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """FR is a SDAC continental price-taker. These features capture the
    European merit order and supply/demand balance that drives FR prices.

    Empirically validated partial correlations (controlling for spark_spread):
    - continent_thermal_need: 0.534
    - continent_zero_mc_pen: -0.550
    - continent_re_pen: -0.483
    - continent_weighted_price: 0.609
    """
    eff = cfg.get("gas_efficiency", 0.50)
    em = cfg.get("emission_factor", 0.37)

    # ------------------------------------------------------------------
    # A. Continental thermal need = residual_load − nuclear
    #    The demand that GAS must serve across Europe.
    #    partial r=0.534 beyond spark_spread, 0.332 beyond spot_la.
    # ------------------------------------------------------------------
    continent_nuke = (
        _safe_col(df, "fr_nuclear_avcap_f")
        + _safe_col(df, "de_nuclear_avcap_f")
        + _safe_col(df, "be_nuclear_avcap_f")
    )
    df["continent_nuclear_total"] = continent_nuke
    df["continent_thermal_need"] = df["continental_residual_load"] - continent_nuke

    # ------------------------------------------------------------------
    # B. Continental zero-MC penetration
    #    (wind + solar + nuclear + hydro) / load — how much of European
    #    demand is served by free generation?
    #    partial r=-0.550 beyond spark, -0.340 beyond spot_la.
    # ------------------------------------------------------------------
    fr_hydro = _safe_col(df, "fr_hydro_ror_f") + _safe_col(df, "fr_hydro_res_f")
    alpine_hydro = (
        _safe_col(df, "ch_hydro_ror_f") + _safe_col(df, "ch_hydro_res_f")
        + _safe_col(df, "at_hydro_ror_f") + _safe_col(df, "at_hydro_res_f")
    )
    continent_zero_mc = (
        df["continental_wind_total"] + df["continental_solar_total"]
        + continent_nuke + fr_hydro + alpine_hydro
    )
    df["continent_zero_mc_pen"] = (
        continent_zero_mc / df["continental_load_total"].clip(lower=1)
    )

    # ------------------------------------------------------------------
    # C. Continental renewable penetration (wind + solar only)
    #    partial r=-0.483 beyond spark, -0.241 beyond spot_la.
    # ------------------------------------------------------------------
    df["continent_re_pen"] = (
        (df["continental_wind_total"] + df["continental_solar_total"])
        / df["continental_load_total"].clip(lower=1)
    )

    # ------------------------------------------------------------------
    # D. Continental load-weighted price
    #    Markets with higher load contribute more to the European price.
    #    partial r=0.609 beyond spark.
    # ------------------------------------------------------------------
    spot_cols = ["fr_spot_la", "de_spot_la", "nl_spot_la", "be_spot_la",
                 "ch_spot_la", "es_spot_la"]
    load_cols = ["fr_load_f", "de_load_f", "nl_load_f", "be_load_f",
                 "ch_load_f", "es_load_f"]
    weighted_sum = sum(
        _safe_col(df, s) * _safe_col(df, l)
        for s, l in zip(spot_cols, load_cols)
    )
    total_load = sum(_safe_col(df, l) for l in load_cols)
    df["continent_weighted_price"] = weighted_sum / total_load.clip(lower=1)

    # ------------------------------------------------------------------
    # E. Carbon-to-gas ratio
    #    High ratio = carbon cost dominates fuel mix → different price regime.
    #    partial r=-0.226 beyond spot_la.
    # ------------------------------------------------------------------
    df["carbon_to_gas_ratio"] = (
        _safe_col(df, "eu_emission") / _safe_col(df, "nl_gas").clip(lower=1)
    )

    # ------------------------------------------------------------------
    # F. FR-ES spread (Iberian border dynamics)
    #    Spain has different merit order (solar-heavy, less gas).
    #    r=0.873 raw, partial r=0.311 beyond spark.
    # ------------------------------------------------------------------
    df["spread_fr_es"] = _safe_col(df, "fr_spot_la") - _safe_col(df, "es_spot_la")

    # ------------------------------------------------------------------
    # G. European scarcity ratio — THE best incremental signal found
    #    (euro load − all zero-MC) / euro gas capacity
    #    partial r=0.618 beyond spark, 0.414 beyond spot_la.
    # ------------------------------------------------------------------
    es_hydro = _safe_col(df, "es_hydro_ror_f")
    euro_zero_mc = continent_zero_mc + es_hydro
    euro_load = (
        df["continental_load_total"]
        + _safe_col(df, "es_load_f")
        + _safe_col(df, "at_load_f")
        + _safe_col(df, "ch_load_f")
    )
    euro_deficit = euro_load - euro_zero_mc
    euro_gas_cap = (
        _safe_col(df, "fr_gas_avcap_f")
        + _safe_col(df, "de_gas_avcap_f")
        + _safe_col(df, "uk_gas_avcap_f")
    )
    df["euro_scarcity_ratio"] = euro_deficit / euro_gas_cap.clip(lower=1)
    df["euro_adequacy_deficit"] = euro_deficit

    # ------------------------------------------------------------------
    # H. Wind tier-1 penetration (DE+BE = direct FBMC neighbors)
    #    Nearest wind has most price impact. partial r=-0.453 beyond spark.
    # ------------------------------------------------------------------
    wind_tier1 = _safe_col(df, "de_wind_f") + _safe_col(df, "be_wind_f")
    df["wind_tier1_pen"] = wind_tier1 / df["continental_load_total"].clip(lower=1)

    # ------------------------------------------------------------------
    # I. Continental wind / nuclear ratio
    #    Wind substitutes nuclear in the merit order.
    #    partial r=-0.437 beyond spark, -0.207 beyond spot_la.
    # ------------------------------------------------------------------
    df["continent_wind_nuke_ratio"] = (
        df["continental_wind_total"] / continent_nuke.clip(lower=1)
    )

    # ------------------------------------------------------------------
    # J. Spanish fundamentals (different gas market from TTF)
    #    es_thermal_floor: partial r=0.499 beyond spot_la (best in this cat!)
    #    es_load_f: partial r=0.358 beyond spark
    #    es_hydro: partial r=-0.282 beyond spark
    # ------------------------------------------------------------------
    df["es_thermal_floor"] = (
        _safe_col(df, "es_gas") / eff
        + _safe_col(df, "eu_emission") * em
    )
    df["es_residual_load"] = (
        _safe_col(df, "es_load_f")
        - _safe_col(df, "es_wind_f")
        - _safe_col(df, "es_solar_f")
        - es_hydro
    )

    # ------------------------------------------------------------------
    # K. DE river temperature — cooling constraint flag
    #    >20°C triggers thermal plant derating across DE.
    #    partial r=0.334 beyond spark.
    # ------------------------------------------------------------------
    de_river_avg = (
        _safe_col(df, "de_river_temp_danube_donauworth_f")
        + _safe_col(df, "de_river_temp_danube_ingolstadt_f")
    ) / 2
    df["de_river_high"] = (de_river_avg > 20).astype(np.int8)

    # ------------------------------------------------------------------
    # L. Wind-nuclear deviation gap
    #    When wind deviates UP from its weekly norm AND nuclear deviates
    #    DOWN simultaneously → strong price relief signal.
    #    partial r=-0.307 beyond spark, -0.239 beyond spot_la.
    # ------------------------------------------------------------------
    wind_norm = df["continental_wind_total"].rolling(168, min_periods=24).mean()
    nuke_norm = continent_nuke.rolling(168, min_periods=24).mean()
    df["wind_nuke_deviation_gap"] = (
        df["continental_wind_total"] / wind_norm.clip(lower=1)
        - continent_nuke / nuke_norm.clip(lower=1)
    )

    return df


# ---------------------------------------------------------------------------
# Cat 29 — UK island territory features
# ---------------------------------------------------------------------------

def _add_uk_island_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """UK is an isolated island market (GB power market / N2EX).
    Price is set by domestic gas fleet + wind displacement + import dependency.

    Empirically validated partial correlations (controlling for spark_spread):
    - uk_gas_utilization: 0.513
    - uk_capacity_margin: -0.517
    - uk_gas_cost_per_mw: 0.494 (0.663 vs spot_la!)
    - uk_self_sufficiency: -0.509
    """

    uk_nuke = _safe_col(df, "uk_nuclear_avcap_f")
    uk_wind = _safe_col(df, "uk_wind_f")
    uk_solar = _safe_col(df, "uk_solar_f")
    uk_biomass = _safe_col(df, "uk_biomass_avcap_f")
    uk_gas_cap = _safe_col(df, "uk_gas_avcap_f")
    uk_load = df["uk_load_f"]
    uk_gas = _safe_col(df, "uk_gas")

    # Must-run baseload (always on regardless of price)
    uk_mustrun = uk_nuke + uk_biomass

    # Flexible demand = what wind/gas/imports must cover
    uk_flex_need = (uk_load - uk_mustrun).clip(lower=0)

    # Gas generation proxy = flexible demand not covered by renewables
    uk_gas_gen = (uk_flex_need - uk_wind - uk_solar).clip(lower=0)

    # ------------------------------------------------------------------
    # A. UK gas utilization — how stressed is the gas fleet?
    #    partial r=0.513 beyond spark, 0.337 beyond spot_la.
    #    High = expensive peakers running, Low = only efficient CCGTs.
    # ------------------------------------------------------------------
    df["uk_gas_utilization"] = uk_gas_gen / uk_gas_cap.clip(lower=1)

    # ------------------------------------------------------------------
    # B. UK gas headroom — MW of gas capacity still available
    #    partial r=-0.507 beyond spark, -0.326 beyond spot_la.
    #    Low headroom = close to capacity limit = price spikes.
    # ------------------------------------------------------------------
    df["uk_gas_headroom"] = uk_gas_cap - uk_gas_gen

    # ------------------------------------------------------------------
    # C. UK capacity margin — total domestic capacity minus load
    #    partial r=-0.517 beyond spark, -0.336 beyond spot_la.
    #    Negative = must import or face blackout.
    # ------------------------------------------------------------------
    uk_total_domestic = uk_nuke + uk_biomass + uk_gas_cap + uk_wind + uk_solar
    df["uk_capacity_margin"] = uk_total_domestic - uk_load

    # ------------------------------------------------------------------
    # D. UK gas cost per MW of demand — weighted gas contribution
    #    partial r=0.663 beyond spot_la! Strongest incremental feature.
    #    Captures both gas price AND how much gas is needed per unit demand.
    # ------------------------------------------------------------------
    df["uk_gas_cost_per_mw"] = uk_gas * uk_gas_gen / uk_load.clip(lower=1)

    # ------------------------------------------------------------------
    # E. UK self-sufficiency ratio
    #    Can UK domestic generation meet its own load?
    #    partial r=-0.509 beyond spark, -0.307 beyond spot_la.
    # ------------------------------------------------------------------
    df["uk_self_sufficiency"] = uk_total_domestic / uk_load.clip(lower=1)

    # ------------------------------------------------------------------
    # F. UK load as % of weekly peak — demand pressure indicator
    #    partial r=0.427 beyond spark, 0.179 beyond spot_la.
    # ------------------------------------------------------------------
    uk_load_roll_max = uk_load.rolling(168, min_periods=24).max()
    df["uk_load_pct_weekly_peak"] = uk_load / uk_load_roll_max.clip(lower=1)

    return df


# ---------------------------------------------------------------------------
# Cat 30 — Regime & structural break features
# ---------------------------------------------------------------------------

def _add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Structural breaks identified by market research (Jul 2022 — Feb 2025).

    The train period spans two fundamentally different regimes:
    - Crisis (Jul 2022 — ~Mar 2023): 32 nuclear reactors offline, gas is marginal,
      FR is net importer, prices 300+ EUR.
    - Normal (Apr 2023+): Nuclear recovered (362 TWh in 2024), gas barely runs
      (17 TWh), FR is net exporter, prices ~47 EUR.

    Key regulatory events:
    - FBMC Core go-live: 8 Jun 2022 (better cross-border capacity utilization)
    - Iberian exception: 15 Jun 2022 — 31 Dec 2023 (gas cap in ES/PT)
    - EU infra-marginal revenue cap: 1 Jul 2022 — 30 Jun 2023
    """
    dt = df["datetime_CET"]

    # ------------------------------------------------------------------
    # A. Iberian exception flag (Jun 15, 2022 — Dec 31, 2023)
    #    Spain capped gas cost for power at 40-50 EUR/MWh.
    #    → Spanish exports to FR increased 80%, downward pressure on FR price.
    # ------------------------------------------------------------------
    df["iberian_exception"] = (
        (dt >= "2022-06-15") & (dt < "2024-01-01")
    ).astype(np.int8)

    # ------------------------------------------------------------------
    # B. Gas-on-margin proxy
    #    When residual load (after nuclear + renewables) > 0, gas/imports must
    #    run → gas price is the marginal cost setter.
    #    When residual ≤ 0, nuclear + RE cover everything → gas irrelevant.
    # ------------------------------------------------------------------
    fr_nuke = _safe_col(df, "fr_nuclear_avcap_f")
    fr_wind = _safe_col(df, "fr_wind_f")
    fr_solar = _safe_col(df, "fr_solar_f")
    fr_hydro = _safe_col(df, "fr_hydro_ror_f") + _safe_col(df, "fr_hydro_res_f")
    fr_load = df["fr_load_f"]

    fr_zero_mc_gen = fr_nuke + fr_wind + fr_solar + fr_hydro
    fr_thermal_gap = fr_load - fr_zero_mc_gen

    # Continuous: how much thermal/import generation is needed (MW)
    df["fr_thermal_gap"] = fr_thermal_gap
    # Binary: is gas likely marginal?
    df["fr_gas_on_margin"] = (fr_thermal_gap > 0).astype(np.int8)
    # Interaction: gas price ONLY matters when gas is marginal
    df["fr_gas_price_if_marginal"] = (
        _safe_col(df, "nl_gas") * df["fr_gas_on_margin"]
    )

    # Same for UK
    uk_nuke = _safe_col(df, "uk_nuclear_avcap_f")
    uk_bio = _safe_col(df, "uk_biomass_avcap_f")
    uk_wind = _safe_col(df, "uk_wind_f")
    uk_solar = _safe_col(df, "uk_solar_f")
    uk_load = df["uk_load_f"]
    uk_zero_mc = uk_nuke + uk_bio + uk_wind + uk_solar
    uk_thermal_gap = uk_load - uk_zero_mc

    df["uk_thermal_gap"] = uk_thermal_gap
    df["uk_gas_on_margin"] = (uk_thermal_gap > 0).astype(np.int8)
    df["uk_gas_price_if_marginal"] = (
        _safe_col(df, "uk_gas") * df["uk_gas_on_margin"]
    )

    # ------------------------------------------------------------------
    # C. Negative price risk indicator
    #    When zero-MC generation > load → oversupply → negative prices.
    #    Doubled in 2024 vs 2023. Especially spring/summer weekends.
    # ------------------------------------------------------------------
    fr_oversupply = fr_zero_mc_gen - fr_load
    df["fr_oversupply_mw"] = fr_oversupply.clip(lower=0)
    df["fr_negative_price_risk"] = (fr_oversupply > 0).astype(np.int8)

    uk_oversupply = uk_zero_mc - uk_load
    df["uk_oversupply_mw"] = uk_oversupply.clip(lower=0)
    df["uk_negative_price_risk"] = (uk_oversupply > 0).astype(np.int8)

    # ------------------------------------------------------------------
    # D. Nuclear availability ratio (% of max fleet capacity)
    #    Continuous signal capturing the crisis→recovery arc.
    #    FR theoretical max: ~61.4 GW, but we use expanding max as proxy.
    # ------------------------------------------------------------------
    fr_nuke_max = fr_nuke.expanding(min_periods=1).max()
    df["fr_nuclear_avail_ratio"] = fr_nuke / fr_nuke_max.clip(lower=1)

    uk_nuke_max = uk_nuke.expanding(min_periods=1).max()
    df["uk_nuclear_avail_ratio"] = uk_nuke / uk_nuke_max.clip(lower=1)

    # ------------------------------------------------------------------
    # E. Gas sensitivity regime
    #    How much does the gas price explain the spot price right now?
    #    Rolling correlation gas vs spot over 168h window.
    #    High = gas-driven regime, Low = nuclear/RE-driven regime.
    # ------------------------------------------------------------------
    gas = _safe_col(df, "nl_gas")
    fr_la = _safe_col(df, "fr_spot_la")
    df["fr_gas_spot_rolling_corr"] = (
        gas.rolling(168, min_periods=48).corr(fr_la)
    )

    # ------------------------------------------------------------------
    # F. Gas price regime: relative gas price vs trailing 1-year average
    #    Captures the crisis↔normal transition without lookahead.
    #    Crisis (2022-mid-2023): gas ~200+ EUR/MWh → this ratio > 2
    #    Post-crisis (2024+):    gas ~35-50 EUR/MWh → this ratio < 0.5
    #    XGBoost can learn "when ratio < 0.7, renewables dominate price
    #    formation; when ratio > 1.5, gas marginal cost sets the price."
    # ------------------------------------------------------------------
    fr_gas  = _safe_col(df, "fr_gas").ffill()
    uk_gas  = _safe_col(df, "uk_gas").ffill()
    nl_gas  = gas.ffill()

    # Trailing 1-year rolling mean (8760h = 365 days)
    fr_gas_1y  = fr_gas.rolling(8760, min_periods=168).mean()
    uk_gas_1y  = uk_gas.rolling(8760, min_periods=168).mean()
    nl_gas_1y  = nl_gas.rolling(8760, min_periods=168).mean()

    # Relative gas price (>1 = expensive vs history, <1 = cheap vs history)
    df["fr_gas_vs_1y_avg"]  = fr_gas  / fr_gas_1y.clip(lower=0.1)
    df["uk_gas_vs_1y_avg"]  = uk_gas  / uk_gas_1y.clip(lower=0.1)
    df["nl_gas_vs_1y_avg"]  = nl_gas  / nl_gas_1y.clip(lower=0.1)

    # Binary crisis regime flag (gas > 2x 1-year average)
    df["fr_gas_crisis_regime"] = (df["fr_gas_vs_1y_avg"] > 2.0).astype(np.int8)
    df["uk_gas_crisis_regime"] = (df["uk_gas_vs_1y_avg"] > 2.0).astype(np.int8)

    # Thermal gap interaction: importance of gas depends on its relative price
    df["fr_thermal_gap_x_gas_regime"] = (
        fr_thermal_gap * df["fr_gas_vs_1y_avg"].clip(upper=5)
    )
    df["uk_thermal_gap_x_gas_regime"] = (
        uk_thermal_gap * df["uk_gas_vs_1y_avg"].clip(upper=5)
    )

    return df


# ---------------------------------------------------------------------------
# Cat 32 — Advanced price proxies (opportunity cost, barrier, load-price, hydro)
# ---------------------------------------------------------------------------

def _add_advanced_price_proxies(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Advanced price formation proxies for non-stationarity correction.

    Addresses the systematic over-prediction bias (+20 EUR) caused by
    the gas-based merit_order_cost ignoring that nuclear/renewables set
    the FR price ~90% of the time in normal regime (2024+).

    A. Dynamic marginal: nuclear vs gas weighted by scarcity
    B. Opportunity cost: min(internal, import) — SDAC coupling
    C. Scarcity barrier: exponential bridge for convex merit order
    D. Load-price signal: recent load→price mapping (regime detector)
    E. Hydro opportunity cost: value of water = peak spark spread
    F. Basis v2: spot_la - opportunity_cost (for basis modeling)
    """
    eff = cfg.get("gas_efficiency", 0.50)
    nuclear_mc = cfg.get("nuclear_marginal_cost", 12.0)

    # ------------------------------------------------------------------
    # A. Dynamic marginal cost — nuclear vs gas weighted by scarcity
    #    When system is relaxed (scarcity < 0.3): marginal = nuclear (~12 EUR)
    #    When system is tight (scarcity > 0.8): marginal = gas spark spread
    #    Smooth interpolation in between.
    # ------------------------------------------------------------------
    fr_scarcity = df["fr_scarcity_ratio"].clip(0, 1)
    uk_scarcity = df["uk_scarcity_ratio"].clip(0, 1)

    fr_gas_w = ((fr_scarcity - 0.3) / 0.5).clip(0, 1)
    uk_gas_w = ((uk_scarcity - 0.3) / 0.5).clip(0, 1)

    df["fr_dynamic_marginal"] = (
        fr_gas_w * df["fr_merit_order_cost"] + (1 - fr_gas_w) * nuclear_mc
    )
    df["uk_dynamic_marginal"] = (
        uk_gas_w * df["uk_merit_order_cost"] + (1 - uk_gas_w) * nuclear_mc
    )

    # ------------------------------------------------------------------
    # B. Opportunity cost — SDAC coupling regime
    #    FR price = min(internal_marginal, cheapest_import)
    #    In EUPHEMIA, the French price is capped by import from neighbors
    #    when France is in surplus or neighbors have cheaper generation.
    #    Transport cost approximated at 2 EUR/MWh (typical congestion rent).
    # ------------------------------------------------------------------
    transport_cost = cfg.get("transport_cost_approx", 2.0)

    # Cheapest continental import price (lagged — same timing as spot_la)
    neighbor_spots = pd.concat([
        _safe_col(df, "de_spot_la"),
        _safe_col(df, "be_spot_la"),
        _safe_col(df, "ch_spot_la"),
        _safe_col(df, "es_spot_la"),
    ], axis=1)
    fr_import_price = neighbor_spots.min(axis=1) + transport_cost
    df["fr_import_price"] = fr_import_price

    df["fr_opportunity_cost"] = pd.concat([
        df["fr_dynamic_marginal"], fr_import_price
    ], axis=1).min(axis=1)

    # UK: imports from continent via cables
    uk_import_spots = pd.concat([
        _safe_col(df, "fr_spot_la"),
        _safe_col(df, "be_spot_la"),
        _safe_col(df, "nl_spot_la"),
    ], axis=1)
    # UK transport cost higher (subsea cables)
    uk_import_price = uk_import_spots.min(axis=1) + transport_cost * 2
    df["uk_import_floor"] = uk_import_price

    df["uk_opportunity_cost"] = pd.concat([
        df["uk_dynamic_marginal"], uk_import_price
    ], axis=1).min(axis=1)

    # ------------------------------------------------------------------
    # C. Scarcity exponential barrier — convex merit order
    #    merit_barrier = spark * (1 / (1 - scarcity)^p)
    #    When scarcity → 1, price explodes (peaker activation).
    #    Clip at 0.98 to avoid division by zero.
    # ------------------------------------------------------------------
    p = cfg.get("scarcity_barrier_power", 1.5)

    fr_s_clipped = fr_scarcity.clip(upper=0.98)
    uk_s_clipped = uk_scarcity.clip(upper=0.98)

    df["fr_scarcity_barrier"] = (
        df["fr_spark_spread"] * (1.0 / (1.0 - fr_s_clipped) ** p)
    )
    df["uk_scarcity_barrier"] = (
        df["uk_spark_spread"] * (1.0 / (1.0 - uk_s_clipped) ** p)
    )

    # ------------------------------------------------------------------
    # D. Load-price signal — recent load→price mapping (regime detector)
    #    "For the current load level, what did the market pay recently?"
    #    Approximated as: recent_avg_price * (current_load / recent_avg_load)
    #    This adapts to regime shifts: if prices dropped 50% for the same
    #    load level, this feature reflects it within 7 days.
    # ------------------------------------------------------------------
    fr_la = _safe_col(df, "fr_spot_la")
    uk_la = _safe_col(df, "uk_spot_la")
    fr_load = df["fr_load_f"]
    uk_load = df["uk_load_f"]
    fr_resid = df["fr_residual_load"]
    uk_resid = df["uk_residual_load"]

    # Price per MW of residual load (dynamic merit order slope)
    fr_price_168h = fr_la.rolling(168, min_periods=24).mean()
    fr_resid_168h = fr_resid.rolling(168, min_periods=24).mean()
    uk_price_168h = uk_la.rolling(168, min_periods=24).mean()
    uk_resid_168h = uk_resid.rolling(168, min_periods=24).mean()

    df["fr_price_per_mw_7d"] = fr_price_168h / fr_resid_168h.clip(lower=1)
    df["uk_price_per_mw_7d"] = uk_price_168h / uk_resid_168h.clip(lower=1)

    # Recent load-price signal: "at current load, expect this price"
    df["fr_load_price_signal_7d"] = df["fr_price_per_mw_7d"] * fr_resid
    df["uk_load_price_signal_7d"] = df["uk_price_per_mw_7d"] * uk_resid

    # Same for raw load (not just residual)
    fr_load_168h = fr_load.rolling(168, min_periods=24).mean()
    df["fr_load_price_signal_load"] = fr_price_168h * (fr_load / fr_load_168h.clip(lower=1))

    # ------------------------------------------------------------------
    # E. Hydro opportunity cost — value of water
    #    Hydro reservoir operators sell at the peak spark spread of the week,
    #    not at marginal cost (0 EUR). Rolling max spark over 168h = the
    #    price floor that hydro sets during peaks.
    # ------------------------------------------------------------------
    df["fr_hydro_opp_cost"] = df["fr_spark_spread"].rolling(168, min_periods=24).max()
    df["uk_hydro_opp_cost"] = df["uk_spark_spread"].rolling(168, min_periods=24).max()

    # ------------------------------------------------------------------
    # F. Basis v2 — spot_la - opportunity_cost
    #    More stationary than spot - merit_order_cost because opportunity_cost
    #    accounts for nuclear marginal cost AND import caps.
    # ------------------------------------------------------------------
    df["fr_basis_v2"] = fr_la - df["fr_opportunity_cost"]
    df["uk_basis_v2"] = uk_la - df["uk_opportunity_cost"]

    # Basis v2 lags (for basis modeling target)
    df["fr_basis_v2_lag_48h"] = df["fr_basis_v2"].shift(24)
    df["uk_basis_v2_lag_48h"] = df["uk_basis_v2"].shift(24)
    df["fr_basis_v2_roll_24h_mean"] = df["fr_basis_v2"].rolling(24, min_periods=1).mean()
    df["uk_basis_v2_roll_24h_mean"] = df["uk_basis_v2"].rolling(24, min_periods=1).mean()

    return df.copy()  # defragment after 290+ column assignments


# ---------------------------------------------------------------------------
# Cat 31 -- Daily renewable surplus / deficit features (game-changer)
#
# Error analysis showed that V15 fails catastrophically on solar-surplus days
# (e.g. 2024-06-26: entire day of negative prices, model predicted +50-75).
# The model has no signal for "today is a renewable-surplus day across all 24h".
# All _f forecast columns are D-1 published → no lookahead.
# ---------------------------------------------------------------------------

def _add_daily_renewable_surplus_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute daily aggregated renewable vs demand features.

    Groups hourly _f forecast columns by calendar date and broadcasts the
    daily aggregate back to every hour of that day.  All inputs are day-ahead
    forecasts published before the auction (no look-ahead).
    """
    dt = pd.to_datetime(df["datetime_CET"])
    date_key = dt.dt.date.astype(str)

    fr_solar  = _safe_col(df, "fr_solar_f")
    fr_wind   = _safe_col(df, "fr_wind_f")
    fr_load   = df["fr_load_f"].clip(lower=1)
    fr_nuke   = _safe_col(df, "fr_nuclear_avcap_f")
    fr_resid  = fr_load - fr_solar - fr_wind          # residual load: <0 => renewable surplus

    uk_wind   = _safe_col(df, "uk_wind_f")
    uk_load   = _safe_col(df, "uk_load_f").clip(lower=1)
    uk_resid  = uk_load - uk_wind

    tmp = pd.DataFrame({
        "dk": date_key,
        "fr_solar": fr_solar, "fr_wind": fr_wind,
        "fr_load": fr_load,   "fr_resid": fr_resid, "fr_nuke": fr_nuke,
        "uk_wind": uk_wind,   "uk_load": uk_load,   "uk_resid": uk_resid,
    })

    # ── France daily features ─────────────────────────────────────────────────
    # Renewable coverage ratio: >1 means more renewable energy than demand
    daily_fr_solar_sum  = tmp.groupby("dk")["fr_solar"].transform("sum")
    daily_fr_wind_sum   = tmp.groupby("dk")["fr_wind"].transform("sum")
    daily_fr_load_sum   = tmp.groupby("dk")["fr_load"].transform("sum")
    daily_fr_load_min   = tmp.groupby("dk")["fr_load"].transform("min").clip(lower=1)
    daily_fr_solar_peak = tmp.groupby("dk")["fr_solar"].transform("max")
    daily_fr_wind_mean  = tmp.groupby("dk")["fr_wind"].transform("mean")
    daily_fr_nuke_mean  = tmp.groupby("dk")["fr_nuke"].transform("mean")

    # Residual load statistics
    daily_fr_resid_min  = tmp.groupby("dk")["fr_resid"].transform("min")
    daily_fr_resid_mean = tmp.groupby("dk")["fr_resid"].transform("mean")
    daily_fr_neg_hours  = tmp.groupby("dk")["fr_resid"].transform(lambda x: (x < 0).sum().astype(float))
    daily_fr_surplus    = tmp.groupby("dk")["fr_resid"].transform(lambda x: (-x).clip(lower=0).sum())

    df["fr_daily_re_coverage"]       = (daily_fr_solar_sum + daily_fr_wind_sum) / daily_fr_load_sum
    df["fr_daily_residual_load_min"] = daily_fr_resid_min
    df["fr_daily_residual_load_mean"]= daily_fr_resid_mean
    df["fr_daily_solar_peak"]        = daily_fr_solar_peak
    df["fr_daily_wind_mean"]         = daily_fr_wind_mean
    df["fr_daily_nuclear_mean"]      = daily_fr_nuke_mean
    # Solar peak vs minimum load: high = midday solar glut risk
    df["fr_solar_peak_vs_load_min"]  = daily_fr_solar_peak / daily_fr_load_min
    # Count and surplus removed — zero importance in XGBoost (redundant with
    # fr_daily_residual_load_min and fr_daily_re_coverage)
    # Nuclear mean relative to load: low = more gas/imports needed
    df["fr_nuclear_load_ratio_daily"]= daily_fr_nuke_mean / daily_fr_load_sum.clip(lower=1) * 24

    # ── France nuclear heat stress (river temp x nuclear capacity) ────────────
    # Curtailment threshold is 25 C, but stress builds from ~22 C.
    # When river temp rises, nuclear output must be cut -> price spikes.
    river_temp = _safe_col(df, "fr_max_river_temp")
    heat_stress = (river_temp - 22.0).clip(lower=0)                          # 0 below 22 C
    df["fr_nuclear_heat_stress"]     = heat_stress * (daily_fr_nuke_mean / 63000.0)
    df["fr_heat_stress_resid_load"]  = heat_stress * daily_fr_resid_mean / 10000.0

    # ── UK daily features ─────────────────────────────────────────────────────
    daily_uk_wind_sum   = tmp.groupby("dk")["uk_wind"].transform("sum")
    daily_uk_load_sum   = tmp.groupby("dk")["uk_load"].transform("sum")
    daily_uk_resid_min  = tmp.groupby("dk")["uk_resid"].transform("min")
    daily_uk_resid_mean = tmp.groupby("dk")["uk_resid"].transform("mean")
    df["uk_daily_wind_penetration"]  = daily_uk_wind_sum / daily_uk_load_sum
    df["uk_daily_residual_load_min"] = daily_uk_resid_min
    df["uk_daily_residual_load_mean"]= daily_uk_resid_mean
    # uk_negative_resid_hours removed — zero importance (redundant with uk_daily_residual_load_min)

    # ── Continent-wide daily surplus (cross-border price suppression) ─────────
    de_wind  = _safe_col(df, "de_wind_f")
    de_solar = _safe_col(df, "de_solar_f")
    de_load  = _safe_col(df, "de_load_f").clip(lower=1)
    tmp["cont_surplus"] = (fr_solar + fr_wind - fr_load +
                           de_wind + de_solar - de_load)
    daily_cont_surplus = tmp.groupby("dk")["cont_surplus"].transform("mean")
    df["fr_cont_daily_surplus_mean"] = daily_cont_surplus

    return df
