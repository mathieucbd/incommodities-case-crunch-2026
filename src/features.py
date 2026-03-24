"""Feature engineering pipeline for InCommodities Case Crunch 2026.

Builds ~290 derived features across 30 categories on top of the raw columns.
Stateless: identical behaviour on train and test, no target access.
CatBoost handles NaN natively — leading NaN from shift/rolling are left as-is.

Categories
----------
 1  Calendar / cyclical time features
 2  Holiday features (FR/UK/DE/BE/NL + bridge days + distance)
 3  Fundamental market (spark spreads, residual load, supply-demand)
 4  Renewable penetration
 5  Renewable dynamics (ramps, 24h changes, volatility)
 6  Nuclear features (changes, thresholds, rolling deviation)
 7  Hydro features (alpine, reservoir share, baseload surplus)
 8  Interconnector features (utilization, net flows, congestion)
 9  Price features (multi-horizon lags, spreads, continental avg)
10  Interaction features (thermal_need × gas, wind × hour …)
11  Regional / neighbour features (continental aggregates)
12  River temperature features
13  Non-linear transforms (squared, cubed, log-spark, clipped wind)
14  Rolling statistics (24h / 168h mean/std/min/max)
15  Momentum / trend features (acceleration, EWM, spread momentum)
16  Advanced supply/demand (scarcity critical/extreme, residual v2)
17  Load & residual ramps (1h / 3h)
18  Multi-efficiency spark spreads (OCGT / CCGT)
19  Advanced interconnection (flow/ATC ratio, unused capacity)
20  Z-scores & anomalies (residual, load, wind — 14-day windows)
21  Stochastic / SDE signals (jump count, vol ratio, mean reversion)
22  Improved transforms (asinh for spot and spark)
23  Nuclear shortfall (expanding-max gap — avoids look-ahead bias)
24  ATC / NTC ratios per individual cable
25  Market-specific features (SDAC / N2EX — empirically validated)
26  Advanced price formation signals (partial-r validated)
27  FR continent territory (merit order, zero-MC pen, weighted price)
28  UK island territory (gas utilization, capacity margin, self-sufficiency)
29  Regime & structural break features (iberian exception, gas-on-margin)
30  Advanced price proxies (dynamic marginal, opportunity cost, scarcity barrier)
"""

from __future__ import annotations

import logging
from pathlib import Path

import holidays as holidays_lib
import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load feature-engineering config once at import time
# ---------------------------------------------------------------------------

_cfg_path = Path(__file__).resolve().parent.parent / "config.yaml"
with open(_cfg_path, "r") as _f:
    _RAW_CFG = yaml.safe_load(_f)
_CFG = _RAW_CFG.get("feature_engineering", {})

# Convenience constants
_GAS_EFF = _CFG.get("gas_efficiency", 0.50)
_EM_FACTOR = _CFG.get("emission_factor", 0.37)
_OCGT_EFF = _CFG.get("ocgt_efficiency", 0.40)
_CCGT_EFF = _CFG.get("ccgt_efficiency", 0.55)
_NUCLEAR_MC = _CFG.get("nuclear_marginal_cost", 12.0)
_TRANSPORT_COST = _CFG.get("transport_cost_approx", 2.0)
_RIVER_HOT_FR = _CFG.get("river_hot_threshold_fr", 25.0)
_RIVER_HOT_DE = _CFG.get("river_hot_threshold_de", 23.0)
_NUKE_LOW = _CFG.get("nuclear_low_threshold", 35000)
_NUKE_VLOW = _CFG.get("nuclear_very_low_threshold", 25000)
_SCARCITY_CRIT = _CFG.get("scarcity_critical_threshold", 0.85)
_SCARCITY_EXT = _CFG.get("scarcity_extreme_threshold", 0.95)
_JUMP_THRESH = _CFG.get("jump_threshold", 50.0)
_SCAR_BARRIER_P = _CFG.get("scarcity_barrier_power", 1.5)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_col(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    """Return column if present, else a constant series (avoids KeyError)."""
    if col in df.columns:
        return df[col]
    return pd.Series(default, index=df.index)


def _safe_sum(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Sum of existing columns; returns 0 series if none exist."""
    existing = [c for c in cols if c in df.columns]
    if not existing:
        return pd.Series(0.0, index=df.index)
    return df[existing].sum(axis=1)


def _safe_mean(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Mean of existing columns; returns NaN series if none exist."""
    existing = [c for c in cols if c in df.columns]
    if not existing:
        return pd.Series(np.nan, index=df.index)
    return df[existing].mean(axis=1)


def _dt(df: pd.DataFrame) -> pd.DatetimeIndex:
    """Return a DatetimeIndex from the DataFrame index."""
    return pd.DatetimeIndex(df.index)


# ---------------------------------------------------------------------------
# Cat 1 — Calendar / cyclical time features
# ---------------------------------------------------------------------------

def engineer_calendar_effects(df: pd.DataFrame) -> pd.DataFrame:
    """Raw integers + cyclical sin/cos encodings + binary period flags."""
    out = df.copy()
    dt = _dt(out)
    h = dt.hour
    dow = dt.dayofweek
    m = dt.month
    doy = dt.dayofyear

    # Raw integers
    out["hour"] = h
    out["day_of_week"] = dow
    out["month"] = m
    out["day_of_year"] = doy
    out["week_of_year"] = dt.isocalendar().week.astype(int).values
    out["quarter"] = dt.quarter

    # Cyclical encodings
    out["hour_sin"] = np.sin(2 * np.pi * h / 24)
    out["hour_cos"] = np.cos(2 * np.pi * h / 24)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    out["month_sin"] = np.sin(2 * np.pi * m / 12)
    out["month_cos"] = np.cos(2 * np.pi * m / 12)
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    # Binary period indicators
    out["is_weekend"] = (dow >= 5).astype(np.int8)
    out["is_business_hour"] = ((dow < 5) & (h >= 8) & (h <= 19)).astype(np.int8)
    out["is_morning_ramp"] = ((h >= 6) & (h <= 9)).astype(np.int8)
    out["is_evening_peak"] = ((h >= 17) & (h <= 20)).astype(np.int8)
    out["is_night"] = ((h >= 23) | (h <= 5)).astype(np.int8)
    out["is_solar_hours"] = ((h >= 10) & (h <= 16)).astype(np.int8)

    # Composite
    out["hour_x_dow"] = h * 7 + dow

    return out


# ---------------------------------------------------------------------------
# Cat 2 — Holiday features
# ---------------------------------------------------------------------------

def engineer_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    """FR/UK/DE/BE/NL holiday flags, bridge days, and distance to next holiday."""
    out = df.copy()
    dt = _dt(out)
    dates = dt.date
    years = dt.year
    yr_range = range(int(years.min()), int(years.max()) + 2)

    hol_fr = holidays_lib.France(years=yr_range)
    hol_uk = holidays_lib.UnitedKingdom(years=yr_range)
    hol_de = holidays_lib.Germany(years=yr_range)
    hol_be = holidays_lib.Belgium(years=yr_range)
    hol_nl = holidays_lib.Netherlands(years=yr_range)

    date_series = pd.Series(dates, index=out.index)

    out["is_fr_holiday"] = date_series.map(lambda d: int(d in hol_fr)).astype(np.int8)
    out["is_uk_holiday"] = date_series.map(lambda d: int(d in hol_uk)).astype(np.int8)
    out["is_holiday_de"] = date_series.map(lambda d: int(d in hol_de)).astype(np.int8)
    out["is_holiday_be"] = date_series.map(lambda d: int(d in hol_be)).astype(np.int8)
    out["is_holiday_nl"] = date_series.map(lambda d: int(d in hol_nl)).astype(np.int8)

    # Bridge day: Mon after holiday Sunday OR Fri before holiday Saturday
    dow = dt.dayofweek
    date_ts = pd.to_datetime(dates)
    prev_day = (date_ts - pd.Timedelta(days=1)).date
    next_day = (date_ts + pd.Timedelta(days=1)).date
    prev_s = pd.Series(prev_day, index=out.index)
    next_s = pd.Series(next_day, index=out.index)

    out["is_bridge_day_fr"] = (
        ((dow == 0) & prev_s.map(lambda d: d in hol_fr))
        | ((dow == 4) & next_s.map(lambda d: d in hol_fr))
    ).astype(np.int8)

    out["is_holiday_or_weekend_fr"] = (
        (out["is_fr_holiday"] == 1) | (out["is_weekend"] == 1)
    ).astype(np.int8)
    out["is_holiday_or_weekend_uk"] = (
        (out["is_uk_holiday"] == 1) | (out["is_weekend"] == 1)
    ).astype(np.int8)

    # Distance to next / since last FR holiday (capped at 30 days)
    fr_hol_dates = sorted(hol_fr.keys())
    fr_hol_arr = np.array([np.datetime64(d) for d in fr_hol_dates])
    date_vals = date_ts.values.astype("datetime64[D]")

    days_to_next = np.full(len(date_vals), 30, dtype=np.int32)
    days_since_last = np.full(len(date_vals), 30, dtype=np.int32)

    for i, d in enumerate(date_vals):
        future = fr_hol_arr[fr_hol_arr >= d]
        if len(future) > 0:
            days_to_next[i] = min(int((future[0] - d) / np.timedelta64(1, "D")), 30)
        past = fr_hol_arr[fr_hol_arr <= d]
        if len(past) > 0:
            days_since_last[i] = min(int((d - past[-1]) / np.timedelta64(1, "D")), 30)

    out["days_to_next_holiday_fr"] = days_to_next
    out["days_since_last_holiday_fr"] = days_since_last

    return out


# ---------------------------------------------------------------------------
# Cat 3 — Fundamental market features
# ---------------------------------------------------------------------------

def engineer_fundamental_market(df: pd.DataFrame) -> pd.DataFrame:
    """Residual loads, spark spreads, baseload gaps, supply-demand ratios."""
    out = df.copy()

    # Residual loads
    out["fr_residual_load"] = (
        out["fr_load_f"]
        - _safe_col(out, "fr_solar_f")
        - _safe_col(out, "fr_wind_f")
    )
    out["uk_residual_load"] = (
        out["uk_load_f"]
        - _safe_col(out, "uk_solar_f")
        - _safe_col(out, "uk_wind_f")
    )
    out["de_residual_load"] = (
        out["de_load_f"]
        - _safe_col(out, "de_solar_f")
        - _safe_col(out, "de_wind_f")
    )

    # Thermal need = residual − nuclear
    out["fr_thermal_need"] = (
        out["fr_residual_load"] - _safe_col(out, "fr_nuclear_avcap_f")
    )
    out["uk_thermal_need"] = (
        out["uk_residual_load"] - _safe_col(out, "uk_nuclear_avcap_f")
    )
    out["fr_thermal_need_pos"] = out["fr_thermal_need"].clip(lower=0)
    out["uk_thermal_need_pos"] = out["uk_thermal_need"].clip(lower=0)

    # Spark spreads (baseline CCGT efficiency)
    out["fr_spark_spread"] = (
        _safe_col(out, "fr_gas") / _GAS_EFF
        + _safe_col(out, "eu_emission") * _EM_FACTOR
    )
    out["uk_spark_spread"] = (
        _safe_col(out, "uk_gas") / _GAS_EFF
        + _safe_col(out, "uk_emission") * _EM_FACTOR
    )
    out["de_spark_spread"] = (
        _safe_col(out, "de_gas") / _GAS_EFF
        + _safe_col(out, "eu_emission") * _EM_FACTOR
    )
    out["nl_spark_spread"] = (
        _safe_col(out, "nl_gas") / _GAS_EFF
        + _safe_col(out, "eu_emission") * _EM_FACTOR
    )

    # Hydro total
    out["fr_hydro_total"] = (
        _safe_col(out, "fr_hydro_res_f") + _safe_col(out, "fr_hydro_ror_f")
    )

    # Baseload gap = load − nuclear − hydro
    out["fr_baseload_gap"] = (
        out["fr_load_f"]
        - _safe_col(out, "fr_nuclear_avcap_f")
        - out["fr_hydro_total"]
    )
    out["uk_baseload_gap"] = (
        out["uk_load_f"]
        - _safe_col(out, "uk_nuclear_avcap_f")
        - _safe_col(out, "uk_biomass_avcap_f")
    )
    out["fr_baseload_gap_pos"] = out["fr_baseload_gap"].clip(lower=0)

    # Spot minus spark (lagged)
    out["fr_spot_minus_spark"] = (
        _safe_col(out, "fr_spot_la") - out["fr_spark_spread"]
    )
    out["uk_spot_minus_spark"] = (
        _safe_col(out, "uk_spot_la") - out["uk_spark_spread"]
    )

    # Gas margin
    out["fr_gas_margin"] = (
        _safe_col(out, "fr_gas_avcap_f") - out["fr_thermal_need_pos"]
    )
    out["uk_gas_margin"] = (
        _safe_col(out, "uk_gas_avcap_f") - out["uk_thermal_need_pos"]
    )

    # Total dispatchable capacity
    out["fr_total_dispatchable"] = (
        _safe_col(out, "fr_nuclear_avcap_f")
        + _safe_col(out, "fr_gas_avcap_f")
        + out["fr_hydro_total"]
    )
    out["uk_total_dispatchable"] = (
        _safe_col(out, "uk_nuclear_avcap_f")
        + _safe_col(out, "uk_gas_avcap_f")
        + _safe_col(out, "uk_biomass_avcap_f")
    )

    # Supply-demand ratio
    out["fr_supply_demand_ratio"] = (
        out["fr_total_dispatchable"] / out["fr_load_f"].clip(lower=1)
    )
    out["uk_supply_demand_ratio"] = (
        (out["uk_total_dispatchable"] + _safe_col(out, "uk_wind_f"))
        / out["uk_load_f"].clip(lower=1)
    )

    return out


# ---------------------------------------------------------------------------
# Cat 4 — Renewable penetration
# ---------------------------------------------------------------------------

def engineer_renewable_penetration(df: pd.DataFrame) -> pd.DataFrame:
    """Individual + total renewable penetration for FR/UK/DE + continental."""
    out = df.copy()

    for zone in ("fr", "uk", "de"):
        load = out[f"{zone}_load_f"].clip(lower=1)
        wind = _safe_col(out, f"{zone}_wind_f")
        solar = _safe_col(out, f"{zone}_solar_f")
        out[f"{zone}_wind_pen"] = wind / load
        out[f"{zone}_solar_pen"] = solar / load
        out[f"{zone}_renewable_pen"] = (wind + solar) / load

    # Continental wind penetration
    cont_wind = _safe_sum(out, ["fr_wind_f", "de_wind_f", "be_wind_f", "nl_wind_f"])
    cont_load = _safe_sum(out, ["fr_load_f", "de_load_f", "be_load_f", "nl_load_f"])
    out["continental_wind_pen"] = cont_wind / cont_load.clip(lower=1)

    # UK wind thresholds
    out["uk_wind_high"] = (
        out["uk_wind_pen"] > _CFG.get("uk_wind_high_threshold", 0.50)
    ).astype(np.int8)
    out["uk_wind_very_high"] = (
        out["uk_wind_pen"] > _CFG.get("uk_wind_very_high_threshold", 0.65)
    ).astype(np.int8)

    return out


# ---------------------------------------------------------------------------
# Cat 5 — Renewable dynamics
# ---------------------------------------------------------------------------

def engineer_renewable_dynamics(df: pd.DataFrame) -> pd.DataFrame:
    """Ramps, 24h changes, and 24h rolling volatility for wind/solar/load."""
    out = df.copy()

    out["uk_wind_ramp_1h"] = _safe_col(out, "uk_wind_f").diff(1)
    out["uk_wind_ramp_3h"] = _safe_col(out, "uk_wind_f").diff(3)
    out["uk_wind_ramp_6h"] = _safe_col(out, "uk_wind_f").diff(6)
    out["fr_wind_ramp_3h"] = _safe_col(out, "fr_wind_f").diff(3)
    out["fr_solar_ramp_3h"] = _safe_col(out, "fr_solar_f").diff(3)
    out["de_wind_ramp_3h"] = _safe_col(out, "de_wind_f").diff(3)

    for col, name in [
        ("fr_wind_f", "fr_wind_change_24h"),
        ("uk_wind_f", "uk_wind_change_24h"),
        ("fr_solar_f", "fr_solar_change_24h"),
        ("fr_load_f", "fr_load_change_24h"),
        ("uk_load_f", "uk_load_change_24h"),
    ]:
        s = _safe_col(out, col)
        out[name] = s - s.shift(24)

    out["fr_residual_change_24h"] = (
        out["fr_residual_load"] - out["fr_residual_load"].shift(24)
    )
    out["uk_residual_change_24h"] = (
        out["uk_residual_load"] - out["uk_residual_load"].shift(24)
    )

    out["fr_wind_volatility_24h"] = (
        _safe_col(out, "fr_wind_f").rolling(24, min_periods=1).std()
    )
    out["uk_wind_volatility_24h"] = (
        _safe_col(out, "uk_wind_f").rolling(24, min_periods=1).std()
    )

    return out


# ---------------------------------------------------------------------------
# Cat 6 — Nuclear features
# ---------------------------------------------------------------------------

def engineer_nuclear_features(df: pd.DataFrame) -> pd.DataFrame:
    """Changes, threshold flags, pct of load, rolling deviation, ramp magnitude."""
    out = df.copy()
    nuke_fr = _safe_col(out, "fr_nuclear_avcap_f")
    nuke_uk = _safe_col(out, "uk_nuclear_avcap_f")

    out["fr_nuclear_change_24h"] = nuke_fr - nuke_fr.shift(24)
    out["fr_nuclear_change_48h"] = nuke_fr - nuke_fr.shift(48)
    out["fr_nuclear_change_168h"] = nuke_fr - nuke_fr.shift(168)
    out["uk_nuclear_change_24h"] = nuke_uk - nuke_uk.shift(24)

    out["fr_nuclear_low"] = (nuke_fr < _NUKE_LOW).astype(np.int8)
    out["fr_nuclear_very_low"] = (nuke_fr < _NUKE_VLOW).astype(np.int8)

    out["fr_nuclear_pct_of_load"] = nuke_fr / out["fr_load_f"].clip(lower=1)
    out["uk_nuclear_pct_of_load"] = nuke_uk / out["uk_load_f"].clip(lower=1)

    out["fr_nuclear_rolling_7d_mean"] = nuke_fr.rolling(168, min_periods=1).mean()
    out["fr_nuclear_deviation_from_7d"] = (
        nuke_fr - out["fr_nuclear_rolling_7d_mean"]
    )
    out["fr_nuclear_ramp_magnitude"] = out["fr_nuclear_change_24h"].abs()

    return out


# ---------------------------------------------------------------------------
# Cat 7 — Hydro features
# ---------------------------------------------------------------------------

def engineer_hydro_features(df: pd.DataFrame) -> pd.DataFrame:
    """Alpine hydro, reservoir share, nuclear+hydro combined, baseload surplus."""
    out = df.copy()

    out["ch_hydro_total"] = (
        _safe_col(out, "ch_hydro_res_f") + _safe_col(out, "ch_hydro_ror_f")
    )
    out["at_hydro_total"] = (
        _safe_col(out, "at_hydro_res_f") + _safe_col(out, "at_hydro_ror_f")
    )
    out["alpine_hydro_total"] = out["ch_hydro_total"] + out["at_hydro_total"]

    out["fr_hydro_change_24h"] = (
        out["fr_hydro_total"] - out["fr_hydro_total"].shift(24)
    )
    out["alpine_hydro_change_168h"] = (
        out["alpine_hydro_total"] - out["alpine_hydro_total"].shift(168)
    )

    hydro_safe = out["fr_hydro_total"].clip(lower=1)
    out["fr_hydro_res_share"] = _safe_col(out, "fr_hydro_res_f") / hydro_safe

    out["fr_nuclear_plus_hydro"] = (
        _safe_col(out, "fr_nuclear_avcap_f") + out["fr_hydro_total"]
    )
    out["fr_baseload_surplus"] = out["fr_nuclear_plus_hydro"] - out["fr_load_f"]

    return out


# ---------------------------------------------------------------------------
# Cat 8 — Interconnector features
# ---------------------------------------------------------------------------

def engineer_interconnectors(df: pd.DataFrame) -> pd.DataFrame:
    """ATC/NTC totals, utilization rates, net flows, cost spreads, congestion."""
    out = df.copy()

    fr_uk_atc = ["atc_fr-uk-1_f", "atc_fr-uk-2_f", "atc_fr-uk-3_f"]
    uk_fr_atc = ["atc_uk-fr-1_f", "atc_uk-fr-2_f", "atc_uk-fr-3_f"]
    fr_uk_ntc = ["ntc_fr-uk-1_f", "ntc_fr-uk-2_f", "ntc_fr-uk-3_f"]
    uk_fr_ntc = ["ntc_uk-fr-1_f", "ntc_uk-fr-2_f", "ntc_uk-fr-3_f"]

    out["fr_uk_atc_total"] = _safe_sum(out, fr_uk_atc)
    out["uk_fr_atc_total"] = _safe_sum(out, uk_fr_atc)
    out["fr_uk_ntc_total"] = _safe_sum(out, fr_uk_ntc)
    out["uk_fr_ntc_total"] = _safe_sum(out, uk_fr_ntc)

    all_to_uk = ["atc_be-uk_f", "atc_dk1-uk_f"] + fr_uk_atc + ["atc_nl-uk_f"]
    all_from_uk = ["atc_uk-be_f", "atc_uk-dk1_f"] + uk_fr_atc + ["atc_uk-nl_f"]
    out["all_to_uk_atc"] = _safe_sum(out, all_to_uk)
    out["all_from_uk_atc"] = _safe_sum(out, all_from_uk)

    # Utilization = 1 − (ATC / NTC)
    out["fr_uk_utilization"] = 1 - (
        out["fr_uk_atc_total"] / out["fr_uk_ntc_total"].clip(lower=1)
    )
    out["uk_fr_utilization"] = 1 - (
        out["uk_fr_atc_total"] / out["uk_fr_ntc_total"].clip(lower=1)
    )
    out["be_uk_utilization"] = 1 - (
        _safe_col(out, "atc_be-uk_f") / _safe_col(out, "ntc_be-uk_f").clip(lower=1)
    )
    out["nl_uk_utilization"] = 1 - (
        _safe_col(out, "atc_nl-uk_f") / _safe_col(out, "ntc_nl-uk_f").clip(lower=1)
    )
    out["max_utilization_to_uk"] = out[
        ["fr_uk_utilization", "be_uk_utilization", "nl_uk_utilization"]
    ].max(axis=1)

    # Net flows (lagged)
    fr_to_uk_flows = ["flow_fr-uk-1_la", "flow_fr-uk-2_la", "flow_fr-uk-3_la"]
    uk_to_fr_flows = ["flow_uk-fr-1_la", "flow_uk-fr-2_la", "flow_uk-fr-3_la"]
    out["fr_uk_net_flow_la"] = (
        _safe_sum(out, fr_to_uk_flows) - _safe_sum(out, uk_to_fr_flows)
    )
    out["be_uk_net_flow_la"] = (
        _safe_col(out, "flow_be-uk_la") - _safe_col(out, "flow_uk-be_la")
    )
    out["nl_uk_net_flow_la"] = (
        _safe_col(out, "flow_nl-uk_la") - _safe_col(out, "flow_uk-nl_la")
    )
    out["dk1_uk_net_flow_la"] = (
        _safe_col(out, "flow_dk1-uk_la") - _safe_col(out, "flow_uk-dk1_la")
    )
    out["total_net_import_uk_la"] = (
        out["fr_uk_net_flow_la"]
        + out["be_uk_net_flow_la"]
        + out["nl_uk_net_flow_la"]
        + out["dk1_uk_net_flow_la"]
    )

    # Cost spreads (lagged)
    out["fr_uk_avg_cost_la"] = _safe_mean(
        out, ["cost_fr-uk-1_la", "cost_fr-uk-2_la", "cost_fr-uk-3_la"]
    )
    out["uk_fr_avg_cost_la"] = _safe_mean(
        out, ["cost_uk-fr-1_la", "cost_uk-fr-2_la", "cost_uk-fr-3_la"]
    )
    out["fr_uk_cost_spread_la"] = out["fr_uk_avg_cost_la"] - out["uk_fr_avg_cost_la"]

    # Congestion indicators
    out["fr_uk_congested"] = (out["fr_uk_utilization"] > 0.9).astype(np.int8)
    out["uk_fr_congested"] = (out["uk_fr_utilization"] > 0.9).astype(np.int8)
    out["any_direction_congested"] = (
        (out["fr_uk_congested"] == 1) | (out["uk_fr_congested"] == 1)
    ).astype(np.int8)

    # Binary online mask for columns with large NaN blocks (>1000 missing)
    for col in out.columns:
        if ("atc_" in col or "cost_" in col) and out[col].isna().sum() > 1000:
            out[f"is_{col}_online"] = out[col].notna().astype(int)

    out["fr_uk_atc_change_24h"] = (
        out["fr_uk_atc_total"] - out["fr_uk_atc_total"].shift(24)
    )

    return out


# ---------------------------------------------------------------------------
# Cat 9 — Price features (lags, spreads, continental average)
# ---------------------------------------------------------------------------

def engineer_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Multi-horizon lags, cross-zone spreads, continental average."""
    out = df.copy()
    fr_la = _safe_col(out, "fr_spot_la")
    uk_la = _safe_col(out, "uk_spot_la")

    # Multi-horizon lags (la is already T-24, so shift adds extra days)
    out["fr_spot_lag_48h"] = fr_la.shift(24)
    out["fr_spot_lag_168h"] = fr_la.shift(144)
    out["uk_spot_lag_48h"] = uk_la.shift(24)
    out["uk_spot_lag_168h"] = uk_la.shift(144)

    # Neighbour lags
    out["de_spot_lag_48h"] = _safe_col(out, "de_spot_la").shift(24)
    out["be_spot_lag_48h"] = _safe_col(out, "be_spot_la").shift(24)
    out["nl_spot_lag_48h"] = _safe_col(out, "nl_spot_la").shift(24)

    # Cross-zone spreads
    out["spread_fr_uk_la"] = fr_la - uk_la
    out["spread_fr_de_la"] = fr_la - _safe_col(out, "de_spot_la")
    out["spread_uk_nl_la"] = uk_la - _safe_col(out, "nl_spot_la")
    out["spread_uk_be_la"] = uk_la - _safe_col(out, "be_spot_la")
    out["spread_de_fr_abs_la"] = (fr_la - _safe_col(out, "de_spot_la")).abs()

    # Continental average
    out["continental_avg_spot_la"] = _safe_mean(
        out, ["de_spot_la", "be_spot_la", "nl_spot_la", "fr_spot_la"]
    )
    out["fr_vs_continental_la"] = fr_la - out["continental_avg_spot_la"]
    out["uk_vs_continental_la"] = uk_la - out["continental_avg_spot_la"]

    # 24h and 168h price changes
    out["fr_spot_change_24h_la"] = fr_la - fr_la.shift(24)
    out["uk_spot_change_24h_la"] = uk_la - uk_la.shift(24)
    out["fr_spot_change_168h_la"] = fr_la - fr_la.shift(144)
    out["uk_spot_change_168h_la"] = uk_la - uk_la.shift(144)

    return out


# ---------------------------------------------------------------------------
# Cat 10 — Interaction features
# ---------------------------------------------------------------------------

def engineer_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Multiplicative interactions validated by EDA."""
    out = df.copy()

    out["fr_thermal_need_x_gas"] = (
        out["fr_thermal_need_pos"] * _safe_col(out, "fr_gas")
    )
    out["uk_thermal_need_x_gas"] = (
        out["uk_thermal_need_pos"] * _safe_col(out, "uk_gas")
    )
    out["fr_baseload_gap_x_gas"] = (
        out["fr_baseload_gap_pos"] * _safe_col(out, "fr_gas")
    )

    uk_wind_gap = (out["uk_load_f"] - _safe_col(out, "uk_wind_f")).clip(lower=0)
    out["uk_wind_gap_x_gas"] = uk_wind_gap * _safe_col(out, "uk_gas")

    out["fr_residual_x_spark"] = out["fr_residual_load"] * out["fr_spark_spread"]
    out["uk_residual_x_spark"] = out["uk_residual_load"] * out["uk_spark_spread"]

    out["fr_wind_x_hour"] = _safe_col(out, "fr_wind_f") * out["hour_sin"]
    out["uk_wind_x_hour"] = _safe_col(out, "uk_wind_f") * out["hour_sin"]

    out["fr_solar_x_load"] = _safe_col(out, "fr_solar_f") * out["fr_load_f"]
    out["fr_nuclear_x_gas"] = (
        _safe_col(out, "fr_nuclear_avcap_f") * _safe_col(out, "fr_gas")
    )
    out["uk_wind_x_gas"] = _safe_col(out, "uk_wind_f") * _safe_col(out, "uk_gas")

    out["fr_congestion_x_spark_diff"] = out["fr_uk_utilization"] * (
        out["fr_spark_spread"] - out["uk_spark_spread"]
    )
    out["fr_thermal_need_x_nuclear_change"] = (
        out["fr_thermal_need"] * out["fr_nuclear_change_24h"].abs()
    )

    return out


# ---------------------------------------------------------------------------
# Cat 11 — Regional / neighbour features
# ---------------------------------------------------------------------------

def engineer_regional_features(df: pd.DataFrame) -> pd.DataFrame:
    """Continental aggregates, Nordic wind, Iberian, Benelux, Alpine."""
    out = df.copy()

    out["continental_load_total"] = _safe_sum(
        out, ["fr_load_f", "de_load_f", "be_load_f", "nl_load_f"]
    )
    out["continental_wind_total"] = _safe_sum(
        out, ["fr_wind_f", "de_wind_f", "be_wind_f", "nl_wind_f"]
    )
    out["continental_solar_total"] = _safe_sum(
        out, ["fr_solar_f", "de_solar_f", "be_solar_f", "nl_solar_f"]
    )
    out["continental_residual_load"] = (
        out["continental_load_total"]
        - out["continental_wind_total"]
        - out["continental_solar_total"]
    )

    out["nordic_wind_total"] = _safe_sum(out, ["dk1_wind_f", "dk2_wind_f"])
    out["iberian_load"] = _safe_col(out, "es_load_f")
    out["iberian_wind"] = _safe_col(out, "es_wind_f")
    out["iberian_solar"] = _safe_col(out, "es_solar_f")

    out["de_load_change_24h"] = out["de_load_f"] - out["de_load_f"].shift(24)
    out["be_nl_combined_load"] = _safe_sum(out, ["be_load_f", "nl_load_f"])
    out["at_ch_combined_load"] = _safe_sum(out, ["at_load_f", "itn_load_f"])

    return out


# ---------------------------------------------------------------------------
# Cat 12 — River temperature features
# ---------------------------------------------------------------------------

def engineer_river_temperature(df: pd.DataFrame) -> pd.DataFrame:
    """Binary threshold flags, excess temperatures, interaction with nuclear."""
    out = df.copy()
    rhone = _safe_col(out, "fr_river_temp_rhone_lyon_f")
    rhine = _safe_col(out, "fr_river_temp_rhine_rheinfelden_f")
    danube_don = _safe_col(out, "de_river_temp_danube_donauworth_f")
    danube_ing = _safe_col(out, "de_river_temp_danube_ingolstadt_f")

    out["fr_rhone_hot"] = (rhone > _RIVER_HOT_FR).astype(np.int8)
    out["fr_rhine_hot"] = (rhine > _RIVER_HOT_FR).astype(np.int8)
    out["de_danube_hot_donauworth"] = (danube_don > _RIVER_HOT_DE).astype(np.int8)
    out["de_danube_hot_ingolstadt"] = (danube_ing > _RIVER_HOT_DE).astype(np.int8)

    out["fr_any_river_hot"] = (
        (out["fr_rhone_hot"] == 1) | (out["fr_rhine_hot"] == 1)
    ).astype(np.int8)
    out["de_any_river_hot"] = (
        (out["de_danube_hot_donauworth"] == 1)
        | (out["de_danube_hot_ingolstadt"] == 1)
    ).astype(np.int8)

    out["fr_rhone_temp_excess"] = (rhone - _RIVER_HOT_FR).clip(lower=0)
    out["fr_rhine_temp_excess"] = (rhine - _RIVER_HOT_FR).clip(lower=0)

    out["fr_max_river_temp"] = pd.concat([rhone, rhine], axis=1).max(axis=1)
    out["de_max_river_temp"] = pd.concat([danube_don, danube_ing], axis=1).max(axis=1)

    out["fr_river_temp_change_24h"] = rhone - rhone.shift(24)
    out["fr_hot_river_x_nuclear_low"] = (
        out["fr_any_river_hot"] * out["fr_nuclear_low"]
    )

    return out


# ---------------------------------------------------------------------------
# Cat 13 — Non-linear transforms
# ---------------------------------------------------------------------------

def engineer_nonlinear_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """Squared loads, cubed thermal need, log spark, sqrt gas, clipped wind."""
    out = df.copy()

    out["fr_residual_load_squared"] = out["fr_residual_load"] ** 2
    out["uk_residual_load_squared"] = out["uk_residual_load"] ** 2
    out["fr_thermal_need_cubed_pos"] = out["fr_thermal_need_pos"] ** 1.5

    out["uk_wind_pen_squared"] = out["uk_wind_pen"] ** 2

    out["fr_spark_spread_log"] = np.log1p(out["fr_spark_spread"].clip(lower=0))
    out["uk_spark_spread_log"] = np.log1p(out["uk_spark_spread"].clip(lower=0))

    fr_la = _safe_col(out, "fr_spot_la")
    uk_la = _safe_col(out, "uk_spot_la")
    out["fr_spot_la_log"] = np.sign(fr_la) * np.log1p(fr_la.abs())
    out["uk_spot_la_log"] = np.sign(uk_la) * np.log1p(uk_la.abs())

    out["fr_gas_sqrt"] = np.sqrt(_safe_col(out, "fr_gas").clip(lower=0))

    out["fr_wind_f_clipped"] = _safe_col(out, "fr_wind_f").clip(upper=22000)
    out["uk_wind_f_clipped"] = _safe_col(out, "uk_wind_f").clip(upper=20000)

    return out


# ---------------------------------------------------------------------------
# Cat 14 — Rolling statistics
# ---------------------------------------------------------------------------

def engineer_rolling_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """24h and 168h rolling mean/std/min/max for spot, load, gas/emission."""
    out = df.copy()
    fr_la = _safe_col(out, "fr_spot_la")
    uk_la = _safe_col(out, "uk_spot_la")

    for prefix, s in [("fr_spot_la", fr_la), ("uk_spot_la", uk_la)]:
        for win, label in [(24, "24h"), (168, "168h")]:
            out[f"{prefix}_roll_{label}_mean"] = s.rolling(win, min_periods=1).mean()
            out[f"{prefix}_roll_{label}_std"] = s.rolling(win, min_periods=1).std()
        out[f"{prefix}_roll_24h_min"] = s.rolling(24, min_periods=1).min()
        out[f"{prefix}_roll_24h_max"] = s.rolling(24, min_periods=1).max()
        out[f"{prefix}_roll_24h_range"] = (
            out[f"{prefix}_roll_24h_max"] - out[f"{prefix}_roll_24h_min"]
        )
        out[f"{prefix}_deviation_24h"] = s - out[f"{prefix}_roll_24h_mean"]
        out[f"{prefix}_deviation_168h"] = s - out[f"{prefix}_roll_168h_mean"]

    out["uk_wind_roll_24h_mean"] = (
        _safe_col(out, "uk_wind_f").rolling(24, min_periods=1).mean()
    )
    out["fr_load_roll_168h_mean"] = out["fr_load_f"].rolling(168, min_periods=1).mean()
    out["uk_load_roll_168h_mean"] = out["uk_load_f"].rolling(168, min_periods=1).mean()

    out["fr_gas_roll_168h_mean"] = (
        _safe_col(out, "fr_gas").rolling(168, min_periods=1).mean()
    )
    out["uk_gas_roll_168h_mean"] = (
        _safe_col(out, "uk_gas").rolling(168, min_periods=1).mean()
    )
    out["eu_emission_roll_168h_mean"] = (
        _safe_col(out, "eu_emission").rolling(168, min_periods=1).mean()
    )

    return out


# ---------------------------------------------------------------------------
# Cat 15 — Momentum / trend features
# ---------------------------------------------------------------------------

def engineer_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Price acceleration, gas momentum, nuclear trend, EWM, spread momentum."""
    out = df.copy()
    fr_la = _safe_col(out, "fr_spot_la")
    uk_la = _safe_col(out, "uk_spot_la")

    fr_mom = out.get("fr_spot_change_24h_la", fr_la - fr_la.shift(24))
    uk_mom = out.get("uk_spot_change_24h_la", uk_la - uk_la.shift(24))
    out["fr_price_acceleration"] = fr_mom - fr_mom.shift(24)
    out["uk_price_acceleration"] = uk_mom - uk_mom.shift(24)

    out["fr_gas_momentum_24h"] = (
        _safe_col(out, "fr_gas") - _safe_col(out, "fr_gas").shift(24)
    )
    out["uk_gas_momentum_24h"] = (
        _safe_col(out, "uk_gas") - _safe_col(out, "uk_gas").shift(24)
    )

    nuke = _safe_col(out, "fr_nuclear_avcap_f")
    out["fr_nuclear_trend_3d"] = (
        nuke.rolling(72, min_periods=1).mean()
        - nuke.rolling(168, min_periods=1).mean()
    )

    out["fr_spot_la_ewm_24h"] = fr_la.ewm(span=24, min_periods=1).mean()
    out["uk_spot_la_ewm_24h"] = uk_la.ewm(span=24, min_periods=1).mean()

    spread = out.get("spread_fr_uk_la", fr_la - uk_la)
    out["spread_fr_uk_momentum"] = spread - spread.shift(24)

    return out


# ---------------------------------------------------------------------------
# Cat 16 — Advanced supply/demand
# ---------------------------------------------------------------------------

def engineer_advanced_supply_demand(df: pd.DataFrame) -> pd.DataFrame:
    """Residual v2 (minus run-of-river), security margin, scarcity critical/extreme."""
    out = df.copy()

    out["fr_residual_load_v2"] = (
        out["fr_load_f"]
        - _safe_col(out, "fr_wind_f")
        - _safe_col(out, "fr_solar_f")
        - _safe_col(out, "fr_hydro_ror_f")
    )

    fr_disp = _safe_col(out, "fr_nuclear_avcap_f") + _safe_col(out, "fr_gas_avcap_f")
    uk_disp = _safe_col(out, "uk_nuclear_avcap_f") + _safe_col(out, "uk_gas_avcap_f")
    out["fr_security_margin"] = fr_disp - out["fr_residual_load"]
    out["uk_security_margin"] = uk_disp - out["uk_residual_load"]

    out["fr_scarcity_ratio"] = out["fr_residual_load"] / fr_disp.clip(lower=1)
    out["uk_scarcity_ratio"] = out["uk_residual_load"] / uk_disp.clip(lower=1)

    out["fr_scarcity_critical"] = (out["fr_scarcity_ratio"] > _SCARCITY_CRIT).astype(np.int8)
    out["uk_scarcity_critical"] = (out["uk_scarcity_ratio"] > _SCARCITY_CRIT).astype(np.int8)
    out["fr_scarcity_extreme"] = (out["fr_scarcity_ratio"] > _SCARCITY_EXT).astype(np.int8)
    out["uk_scarcity_extreme"] = (out["uk_scarcity_ratio"] > _SCARCITY_EXT).astype(np.int8)

    return out


# ---------------------------------------------------------------------------
# Cat 17 — Load & residual ramps
# ---------------------------------------------------------------------------

def engineer_residual_ramps(df: pd.DataFrame) -> pd.DataFrame:
    """1h and 3h ramps for load and residual load (duck-curve pressure)."""
    out = df.copy()

    out["fr_load_ramp_1h"] = out["fr_load_f"].diff(1)
    out["fr_load_ramp_3h"] = out["fr_load_f"].diff(3)
    out["uk_load_ramp_1h"] = out["uk_load_f"].diff(1)
    out["uk_load_ramp_3h"] = out["uk_load_f"].diff(3)

    out["fr_residual_ramp_1h"] = out["fr_residual_load"].diff(1)
    out["fr_residual_ramp_3h"] = out["fr_residual_load"].diff(3)
    out["uk_residual_ramp_1h"] = out["uk_residual_load"].diff(1)
    out["uk_residual_ramp_3h"] = out["uk_residual_load"].diff(3)

    return out


# ---------------------------------------------------------------------------
# Cat 18 — Multi-efficiency spark spreads (OCGT / CCGT)
# ---------------------------------------------------------------------------

def engineer_multi_efficiency_spark(df: pd.DataFrame) -> pd.DataFrame:
    """OCGT (peaker) and CCGT (efficient) marginal costs for FR and UK."""
    out = df.copy()

    out["fr_spark_ocgt"] = (
        _safe_col(out, "fr_gas") / _OCGT_EFF
        + _safe_col(out, "eu_emission") * _EM_FACTOR
    )
    out["uk_spark_ocgt"] = (
        _safe_col(out, "uk_gas") / _OCGT_EFF
        + _safe_col(out, "uk_emission") * _EM_FACTOR
    )
    out["fr_spark_ccgt"] = (
        _safe_col(out, "fr_gas") / _CCGT_EFF
        + _safe_col(out, "eu_emission") * _EM_FACTOR
    )
    out["uk_spark_ccgt"] = (
        _safe_col(out, "uk_gas") / _CCGT_EFF
        + _safe_col(out, "uk_emission") * _EM_FACTOR
    )

    return out


# ---------------------------------------------------------------------------
# Cat 19 — Advanced interconnection
# ---------------------------------------------------------------------------

def engineer_advanced_interconnection(df: pd.DataFrame) -> pd.DataFrame:
    """Flow-over-ATC ratio, unused capacity per cable, total unused to UK."""
    out = df.copy()

    fr_to_uk_flows = ["flow_fr-uk-1_la", "flow_fr-uk-2_la", "flow_fr-uk-3_la"]
    fr_uk_flow = _safe_sum(out, fr_to_uk_flows)

    out["fr_uk_flow_over_atc"] = fr_uk_flow / out["fr_uk_atc_total"].clip(lower=1)
    out["fr_uk_unused_capacity"] = out["fr_uk_atc_total"] - fr_uk_flow.clip(lower=0)
    out["be_uk_unused_capacity"] = (
        _safe_col(out, "atc_be-uk_f")
        - _safe_col(out, "flow_be-uk_la").clip(lower=0)
    )
    out["nl_uk_unused_capacity"] = (
        _safe_col(out, "atc_nl-uk_f")
        - _safe_col(out, "flow_nl-uk_la").clip(lower=0)
    )
    out["total_unused_capacity_to_uk"] = (
        out["fr_uk_unused_capacity"]
        + out["be_uk_unused_capacity"]
        + out["nl_uk_unused_capacity"]
    )

    return out


# ---------------------------------------------------------------------------
# Cat 20 — Z-scores & anomalies (14-day rolling)
# ---------------------------------------------------------------------------

def engineer_regime_zscores(df: pd.DataFrame) -> pd.DataFrame:
    """14-day rolling z-scores for residual load, load, wind; lag reliability."""
    out = df.copy()
    window = 336  # 14 days × 24 h

    for col in ["fr_residual_load", "uk_residual_load"]:
        prefix = col.split("_")[0]
        m = out[col].rolling(window, min_periods=1).mean()
        s = out[col].rolling(window, min_periods=1).std()
        out[f"{prefix}_residual_zscore_14d"] = (out[col] - m) / s.clip(lower=1)

    for col in ["fr_load_f", "uk_load_f"]:
        prefix = col.split("_")[0]
        m = out[col].rolling(window, min_periods=1).mean()
        s = out[col].rolling(window, min_periods=1).std()
        out[f"{prefix}_load_zscore_14d"] = (out[col] - m) / s.clip(lower=1)

    for raw_col in ["fr_wind_f", "uk_wind_f"]:
        prefix = raw_col.split("_")[0]
        wind = _safe_col(out, raw_col)
        m = wind.rolling(window, min_periods=1).mean()
        s = wind.rolling(window, min_periods=1).std()
        out[f"{prefix}_wind_zscore_14d"] = (wind - m) / s.clip(lower=1)

    # Lag reliability: how stale is the 168h lag?
    out["fr_lag_reliability_ratio"] = (
        _safe_col(out, "fr_spot_la") / out["fr_spot_lag_168h"].clip(lower=0.1)
    )
    out["uk_lag_reliability_ratio"] = (
        _safe_col(out, "uk_spot_la") / out["uk_spot_lag_168h"].clip(lower=0.1)
    )

    return out


# ---------------------------------------------------------------------------
# Cat 21 — Stochastic / SDE signals
# ---------------------------------------------------------------------------

def engineer_stochastic_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Jump count/magnitude, vol ratio (stress vs calm), mean reversion strength."""
    out = df.copy()
    fr_la = _safe_col(out, "fr_spot_la")
    uk_la = _safe_col(out, "uk_spot_la")

    fr_abs = fr_la.diff(1).abs()
    uk_abs = uk_la.diff(1).abs()

    fr_jump = (fr_abs > _JUMP_THRESH).astype(float)
    uk_jump = (uk_abs > _JUMP_THRESH).astype(float)

    out["fr_jump_count_24h"] = fr_jump.rolling(24, min_periods=1).sum()
    out["uk_jump_count_24h"] = uk_jump.rolling(24, min_periods=1).sum()
    out["fr_jump_count_48h"] = fr_jump.rolling(48, min_periods=1).sum()
    out["uk_jump_count_48h"] = uk_jump.rolling(48, min_periods=1).sum()

    fr_jval = fr_abs.where(fr_abs > _JUMP_THRESH, 0.0)
    uk_jval = uk_abs.where(uk_abs > _JUMP_THRESH, 0.0)
    out["fr_jump_magnitude_24h"] = (
        fr_jval.rolling(24, min_periods=1).sum()
        / out["fr_jump_count_24h"].clip(lower=1)
    )
    out["uk_jump_magnitude_24h"] = (
        uk_jval.rolling(24, min_periods=1).sum()
        / out["uk_jump_count_24h"].clip(lower=1)
    )

    fr_std24 = out["fr_spot_la_roll_24h_std"]
    fr_std168 = out["fr_spot_la_roll_168h_std"]
    uk_std24 = out["uk_spot_la_roll_24h_std"]
    uk_std168 = out["uk_spot_la_roll_168h_std"]

    out["fr_vol_ratio"] = fr_std24 / fr_std168.clip(lower=0.1)
    out["uk_vol_ratio"] = uk_std24 / uk_std168.clip(lower=0.1)

    out["fr_mean_reversion_strength"] = (
        out["fr_spot_la_deviation_168h"] / fr_std168.clip(lower=0.1)
    )
    out["uk_mean_reversion_strength"] = (
        out["uk_spot_la_deviation_168h"] / uk_std168.clip(lower=0.1)
    )

    return out


# ---------------------------------------------------------------------------
# Cat 22 — Improved transforms (asinh)
# ---------------------------------------------------------------------------

def engineer_asinh_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """arcsinh for spot_la and spark spreads — handles negatives smoothly."""
    out = df.copy()
    out["fr_asinh_spot_la"] = np.arcsinh(_safe_col(out, "fr_spot_la"))
    out["uk_asinh_spot_la"] = np.arcsinh(_safe_col(out, "uk_spot_la"))
    out["fr_asinh_spark"] = np.arcsinh(out["fr_spark_spread"])
    out["uk_asinh_spark"] = np.arcsinh(out["uk_spark_spread"])

    # Legacy names kept for backward compatibility
    for col in ["fr_spot_la", "uk_spot_la", "ch_spot_la"]:
        if col in out.columns:
            out[f"{col}_asinh"] = np.arcsinh(out[col])

    return out


# ---------------------------------------------------------------------------
# Cat 23 — Nuclear shortfall (expanding max — no look-ahead bias)
# ---------------------------------------------------------------------------

_NUCLEAR_FLEET_MAX = {
    "fr": 61400.0,  # MW — FR 56-reactor fleet installed capacity
    "uk": 5800.0,   # MW — UK nuclear fleet installed capacity
}


def engineer_nuclear_shortfall(df: pd.DataFrame) -> pd.DataFrame:
    """Gap between installed fleet capacity and current available capacity.

    Uses hardcoded physical fleet maxima instead of expanding().max() so that
    the feature is identical whether the model sees 2 years or 30 days of
    context (the finals live-inference format provides only 30 days).
    """
    out = df.copy()
    for prefix in ("fr", "uk"):
        col = f"{prefix}_nuclear_avcap_f"
        nuke = _safe_col(out, col)
        out[f"{prefix}_nuclear_shortfall"] = _NUCLEAR_FLEET_MAX[prefix] - nuke
    return out


# ---------------------------------------------------------------------------
# Cat 24 — ATC / NTC ratios per individual cable
# ---------------------------------------------------------------------------

def engineer_atc_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Per-cable ATC/NTC ratio: near 0 = congested, near 1 = free capacity."""
    out = df.copy()
    cables = [
        ("atc_fr-uk-1_f", "ntc_fr-uk-1_f"),
        ("atc_fr-uk-2_f", "ntc_fr-uk-2_f"),
        ("atc_fr-uk-3_f", "ntc_fr-uk-3_f"),
        ("atc_uk-fr-1_f", "ntc_uk-fr-1_f"),
        ("atc_uk-fr-2_f", "ntc_uk-fr-2_f"),
        ("atc_uk-fr-3_f", "ntc_uk-fr-3_f"),
    ]
    for atc_col, ntc_col in cables:
        atc = _safe_col(out, atc_col)
        ntc = _safe_col(out, ntc_col).clip(lower=1)
        out[f"{atc_col}_ratio"] = atc / ntc
    return out


# ---------------------------------------------------------------------------
# Cat 25 — Market-specific features (SDAC / N2EX)
# ---------------------------------------------------------------------------

def engineer_market_specific_features(df: pd.DataFrame) -> pd.DataFrame:
    """SDAC/EUPHEMIA (FR) and N2EX (UK) market mechanism features."""
    out = df.copy()

    # Gas spread NBP vs TTF
    out["gas_spread_uk_eu"] = _safe_col(out, "uk_gas") - _safe_col(out, "nl_gas")

    # Continental thermal floor (avg DE/FR/NL)
    de_floor = (
        _safe_col(out, "de_gas") / _GAS_EFF
        + _safe_col(out, "eu_emission") * _EM_FACTOR
    )
    out["continent_thermal_floor"] = (
        de_floor + out["fr_spark_spread"] + out["nl_spark_spread"]
    ) / 3

    # UK import ratio
    out["uk_import_ratio"] = (
        out["total_net_import_uk_la"] / out["uk_load_f"].clip(lower=1)
    )

    # FR export ratio
    out["fr_export_ratio"] = (
        out["fr_uk_net_flow_la"] / out["fr_load_f"].clip(lower=1)
    )

    # DE wind thresholds
    de_wind_pen = _safe_col(out, "de_wind_f") / out["de_load_f"].clip(lower=1)
    out["de_wind_high"] = (de_wind_pen > 0.30).astype(np.int8)
    out["de_wind_very_high"] = (de_wind_pen > 0.50).astype(np.int8)

    # UK wind share of flexible demand (N2EX merit order)
    uk_baseload = (
        _safe_col(out, "uk_nuclear_avcap_f") + _safe_col(out, "uk_biomass_avcap_f")
    )
    uk_flex_demand = (out["uk_load_f"] - uk_baseload).clip(lower=1)
    out["uk_wind_share_flexible"] = _safe_col(out, "uk_wind_f") / uk_flex_demand

    # FR-DE decoupling indicator (|spread| > 10 EUR → local fundamentals set price)
    fr_de_abs = (
        _safe_col(out, "fr_spot_la") - _safe_col(out, "de_spot_la")
    ).abs()
    out["fr_de_decoupled"] = (fr_de_abs > 10.0).astype(np.int8)

    # Scarcity-weighted merit order cost
    fr_s = out["fr_scarcity_ratio"].clip(0, 1)
    uk_s = out["uk_scarcity_ratio"].clip(0, 1)
    fr_ocgt_w = ((fr_s - 0.5) / 0.4).clip(0, 1)
    uk_ocgt_w = ((uk_s - 0.5) / 0.4).clip(0, 1)
    out["fr_merit_order_cost"] = (
        (1 - fr_ocgt_w) * out["fr_spark_ccgt"] + fr_ocgt_w * out["fr_spark_ocgt"]
    )
    out["uk_merit_order_cost"] = (
        (1 - uk_ocgt_w) * out["uk_spark_ccgt"] + uk_ocgt_w * out["uk_spark_ocgt"]
    )

    # Intraday price anchors h3 / h19 (Ziel & Weron 2018)
    fr_la = _safe_col(out, "fr_spot_la")
    uk_la = _safe_col(out, "uk_spot_la")
    h = out["hour"]
    out["fr_spot_la_h3"] = fr_la.where(h == 3).ffill()
    out["uk_spot_la_h3"] = uk_la.where(h == 3).ffill()
    out["fr_spot_la_h19"] = fr_la.where(h == 19).ffill()
    out["uk_spot_la_h19"] = uk_la.where(h == 19).ffill()
    out["fr_intraday_amplitude"] = out["fr_spot_la_h19"] - out["fr_spot_la_h3"]
    out["uk_intraday_amplitude"] = out["uk_spot_la_h19"] - out["uk_spot_la_h3"]

    # Dark doldrums: winter evening + no wind + no solar
    is_winter = out["month"].isin([11, 12, 1, 2]).astype(float)
    is_eve = out["is_evening_peak"].astype(float)
    out["dark_doldrums_fr"] = (
        is_winter * is_eve
        * (1 - _safe_col(out, "fr_wind_pen"))
        * (1 - _safe_col(out, "fr_solar_pen"))
    )
    out["dark_doldrums_uk"] = (
        is_winter * is_eve * (1 - _safe_col(out, "uk_wind_pen"))
    )

    return out


# ---------------------------------------------------------------------------
# Cat 26 — Advanced price formation signals
# ---------------------------------------------------------------------------

def engineer_price_formation_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Empirically validated features: nuke×gas, implied RE surplus, import pricing."""
    out = df.copy()

    # Nuclear shortfall × gas interaction
    for prefix in ("fr", "uk"):
        shortfall = out[f"{prefix}_nuclear_shortfall"]
        gas = (
            _safe_col(out, "nl_gas") if prefix == "fr"
            else _safe_col(out, "uk_gas")
        )
        out[f"{prefix}_nuke_shortfall_x_gas"] = shortfall * gas

    # Implied renewable surplus = spark − spot_la
    for prefix in ("fr", "uk"):
        gas_col = "nl_gas" if prefix == "fr" else "uk_gas"
        spark = (
            _safe_col(out, gas_col) / _GAS_EFF
            + _safe_col(out, "eu_emission") * _EM_FACTOR
        )
        out[f"{prefix}_implied_re_surplus"] = (
            spark - _safe_col(out, f"{prefix}_spot_la")
        )

    # UK cheapest import price
    fr_uk_cost = _safe_mean(
        out, ["cost_fr-uk-1_la", "cost_fr-uk-2_la", "cost_fr-uk-3_la"]
    )
    import_fr = _safe_col(out, "fr_spot_la") + fr_uk_cost
    import_be = _safe_col(out, "be_spot_la") + _safe_col(out, "cost_be-uk_la")
    import_nl = _safe_col(out, "nl_spot_la") + _safe_col(out, "cost_nl-uk_la")
    import_dk1 = _safe_col(out, "dk1_spot_la") + _safe_col(out, "cost_dk1-uk_la")
    import_stack = pd.concat([import_fr, import_be, import_nl, import_dk1], axis=1)
    out["uk_cheapest_import"] = import_stack.min(axis=1)
    out["uk_import_price_range"] = import_stack.max(axis=1) - import_stack.min(axis=1)

    # Net capacity auction cost
    fr_uk_c = _safe_mean(out, ["cost_fr-uk-1_la", "cost_fr-uk-2_la", "cost_fr-uk-3_la"])
    uk_fr_c = _safe_mean(out, ["cost_uk-fr-1_la", "cost_uk-fr-2_la", "cost_uk-fr-3_la"])
    out["net_capacity_cost_fr_uk"] = fr_uk_c - uk_fr_c

    # UK fossil/import need
    uk_zero_mc = (
        _safe_col(out, "uk_nuclear_avcap_f")
        + _safe_col(out, "uk_biomass_avcap_f")
        + _safe_col(out, "uk_wind_f")
        + _safe_col(out, "uk_solar_f")
    )
    out["uk_fossil_or_import_need"] = (out["uk_load_f"] - uk_zero_mc).clip(lower=0)

    return out


# ---------------------------------------------------------------------------
# Cat 27 — FR continent territory features
# ---------------------------------------------------------------------------

def engineer_fr_continent_features(df: pd.DataFrame) -> pd.DataFrame:
    """Continental merit order, zero-MC penetration, weighted price, euro scarcity."""
    out = df.copy()

    continent_nuke = (
        _safe_col(out, "fr_nuclear_avcap_f")
        + _safe_col(out, "de_nuclear_avcap_f")
        + _safe_col(out, "be_nuclear_avcap_f")
    )
    out["continent_nuclear_total"] = continent_nuke
    out["continent_thermal_need"] = (
        out["continental_residual_load"] - continent_nuke
    )

    fr_hydro = (
        _safe_col(out, "fr_hydro_ror_f") + _safe_col(out, "fr_hydro_res_f")
    )
    alpine_hydro = (
        _safe_col(out, "ch_hydro_ror_f") + _safe_col(out, "ch_hydro_res_f")
        + _safe_col(out, "at_hydro_ror_f") + _safe_col(out, "at_hydro_res_f")
    )
    continent_zero_mc = (
        out["continental_wind_total"] + out["continental_solar_total"]
        + continent_nuke + fr_hydro + alpine_hydro
    )
    out["continent_zero_mc_pen"] = (
        continent_zero_mc / out["continental_load_total"].clip(lower=1)
    )
    out["continent_re_pen"] = (
        (out["continental_wind_total"] + out["continental_solar_total"])
        / out["continental_load_total"].clip(lower=1)
    )

    # Load-weighted continental price
    spot_cols = ["fr_spot_la", "de_spot_la", "nl_spot_la", "be_spot_la",
                 "ch_spot_la", "es_spot_la"]
    load_cols = ["fr_load_f", "de_load_f", "nl_load_f", "be_load_f",
                 "ch_load_f", "es_load_f"]
    weighted_sum = sum(
        _safe_col(out, s) * _safe_col(out, lo)
        for s, lo in zip(spot_cols, load_cols)
    )
    total_load = sum(_safe_col(out, lo) for lo in load_cols)
    out["continent_weighted_price"] = weighted_sum / total_load.clip(lower=1)

    out["carbon_to_gas_ratio"] = (
        _safe_col(out, "eu_emission") / _safe_col(out, "nl_gas").clip(lower=1)
    )
    out["spread_fr_es"] = (
        _safe_col(out, "fr_spot_la") - _safe_col(out, "es_spot_la")
    )

    # European scarcity ratio
    es_hydro = _safe_col(out, "es_hydro_ror_f")
    euro_zero_mc = continent_zero_mc + es_hydro
    euro_load = (
        out["continental_load_total"]
        + _safe_col(out, "es_load_f")
        + _safe_col(out, "at_load_f")
        + _safe_col(out, "ch_load_f")
    )
    euro_deficit = euro_load - euro_zero_mc
    euro_gas_cap = (
        _safe_col(out, "fr_gas_avcap_f")
        + _safe_col(out, "de_gas_avcap_f")
        + _safe_col(out, "uk_gas_avcap_f")
    )
    out["euro_scarcity_ratio"] = euro_deficit / euro_gas_cap.clip(lower=1)
    out["euro_adequacy_deficit"] = euro_deficit

    # Wind tier-1 (DE+BE — direct FBMC neighbours)
    wind_t1 = _safe_col(out, "de_wind_f") + _safe_col(out, "be_wind_f")
    out["wind_tier1_pen"] = wind_t1 / out["continental_load_total"].clip(lower=1)

    out["continent_wind_nuke_ratio"] = (
        out["continental_wind_total"] / continent_nuke.clip(lower=1)
    )

    # Spanish fundamentals
    out["es_thermal_floor"] = (
        _safe_col(out, "es_gas") / _GAS_EFF
        + _safe_col(out, "eu_emission") * _EM_FACTOR
    )
    out["es_residual_load"] = (
        _safe_col(out, "es_load_f")
        - _safe_col(out, "es_wind_f")
        - _safe_col(out, "es_solar_f")
        - es_hydro
    )

    de_river_avg = (
        _safe_col(out, "de_river_temp_danube_donauworth_f")
        + _safe_col(out, "de_river_temp_danube_ingolstadt_f")
    ) / 2
    out["de_river_high"] = (de_river_avg > 20).astype(np.int8)

    wind_norm = out["continental_wind_total"].rolling(168, min_periods=24).mean()
    nuke_norm = continent_nuke.rolling(168, min_periods=24).mean()
    out["wind_nuke_deviation_gap"] = (
        out["continental_wind_total"] / wind_norm.clip(lower=1)
        - continent_nuke / nuke_norm.clip(lower=1)
    )

    return out


# ---------------------------------------------------------------------------
# Cat 28 — UK island territory features
# ---------------------------------------------------------------------------

def engineer_uk_island_features(df: pd.DataFrame) -> pd.DataFrame:
    """Gas utilization, capacity margin, gas cost per MW, self-sufficiency."""
    out = df.copy()

    uk_nuke = _safe_col(out, "uk_nuclear_avcap_f")
    uk_wind = _safe_col(out, "uk_wind_f")
    uk_solar = _safe_col(out, "uk_solar_f")
    uk_biomass = _safe_col(out, "uk_biomass_avcap_f")
    uk_gas_cap = _safe_col(out, "uk_gas_avcap_f")
    uk_load = out["uk_load_f"]
    uk_gas = _safe_col(out, "uk_gas")

    uk_mustrun = uk_nuke + uk_biomass
    uk_flex_need = (uk_load - uk_mustrun).clip(lower=0)
    uk_gas_gen = (uk_flex_need - uk_wind - uk_solar).clip(lower=0)

    out["uk_gas_utilization"] = uk_gas_gen / uk_gas_cap.clip(lower=1)
    out["uk_gas_headroom"] = uk_gas_cap - uk_gas_gen

    uk_total_domestic = uk_nuke + uk_biomass + uk_gas_cap + uk_wind + uk_solar
    out["uk_capacity_margin"] = uk_total_domestic - uk_load

    out["uk_gas_cost_per_mw"] = uk_gas * uk_gas_gen / uk_load.clip(lower=1)
    out["uk_self_sufficiency"] = uk_total_domestic / uk_load.clip(lower=1)

    uk_load_roll_max = uk_load.rolling(168, min_periods=24).max()
    out["uk_load_pct_weekly_peak"] = uk_load / uk_load_roll_max.clip(lower=1)

    return out


# ---------------------------------------------------------------------------
# Cat 29 — Regime & structural break features
# ---------------------------------------------------------------------------

def engineer_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Iberian exception flag, gas-on-margin binary, oversupply/negative-price risk."""
    out = df.copy()
    dt_index = _dt(out)

    # Iberian exception: Jun 15 2022 – Dec 31 2023 (gas cap in ES/PT)
    out["iberian_exception"] = (
        (dt_index >= "2022-06-15") & (dt_index < "2024-01-01")
    ).astype(np.int8)

    # FR gas-on-margin
    fr_nuke = _safe_col(out, "fr_nuclear_avcap_f")
    fr_wind = _safe_col(out, "fr_wind_f")
    fr_solar = _safe_col(out, "fr_solar_f")
    fr_hydro = (
        _safe_col(out, "fr_hydro_ror_f") + _safe_col(out, "fr_hydro_res_f")
    )
    fr_zero_mc = fr_nuke + fr_wind + fr_solar + fr_hydro
    fr_thermal_gap = out["fr_load_f"] - fr_zero_mc
    out["fr_thermal_gap"] = fr_thermal_gap
    out["fr_gas_on_margin"] = (fr_thermal_gap > 0).astype(np.int8)
    out["fr_gas_price_if_marginal"] = (
        _safe_col(out, "nl_gas") * out["fr_gas_on_margin"]
    )

    # UK gas-on-margin
    uk_nuke = _safe_col(out, "uk_nuclear_avcap_f")
    uk_bio = _safe_col(out, "uk_biomass_avcap_f")
    uk_wind = _safe_col(out, "uk_wind_f")
    uk_solar = _safe_col(out, "uk_solar_f")
    uk_zero_mc = uk_nuke + uk_bio + uk_wind + uk_solar
    uk_thermal_gap = out["uk_load_f"] - uk_zero_mc
    out["uk_thermal_gap"] = uk_thermal_gap
    out["uk_gas_on_margin"] = (uk_thermal_gap > 0).astype(np.int8)
    out["uk_gas_price_if_marginal"] = (
        _safe_col(out, "uk_gas") * out["uk_gas_on_margin"]
    )

    # Oversupply / negative price risk
    fr_oversupply = fr_zero_mc - out["fr_load_f"]
    out["fr_oversupply_mw"] = fr_oversupply.clip(lower=0)
    out["fr_negative_price_risk"] = (fr_oversupply > 0).astype(np.int8)

    uk_oversupply = uk_zero_mc - out["uk_load_f"]
    out["uk_oversupply_mw"] = uk_oversupply.clip(lower=0)
    out["uk_negative_price_risk"] = (uk_oversupply > 0).astype(np.int8)

    # Nuclear availability ratio (crisis → recovery arc)
    # Use hardcoded fleet maxima — expanding().max() breaks with 30-day finals context
    out["fr_nuclear_avail_ratio"] = fr_nuke / max(_NUCLEAR_FLEET_MAX["fr"], 1)
    out["uk_nuclear_avail_ratio"] = uk_nuke / max(_NUCLEAR_FLEET_MAX["uk"], 1)

    # Rolling gas-spot correlation (gas-driven vs nuclear/RE-driven regime)
    gas = _safe_col(out, "nl_gas")
    fr_la = _safe_col(out, "fr_spot_la")
    out["fr_gas_spot_rolling_corr"] = (
        gas.rolling(168, min_periods=48).corr(fr_la)
    )

    return out


# ---------------------------------------------------------------------------
# Cat 30 — Advanced price proxies
# ---------------------------------------------------------------------------

def engineer_advanced_price_proxies(df: pd.DataFrame) -> pd.DataFrame:
    """Dynamic marginal cost, opportunity cost, scarcity barrier, load-price signal."""
    out = df.copy()

    fr_s = out["fr_scarcity_ratio"].clip(0, 1)
    uk_s = out["uk_scarcity_ratio"].clip(0, 1)
    fr_gas_w = ((fr_s - 0.3) / 0.5).clip(0, 1)
    uk_gas_w = ((uk_s - 0.3) / 0.5).clip(0, 1)

    out["fr_dynamic_marginal"] = (
        fr_gas_w * out["fr_merit_order_cost"] + (1 - fr_gas_w) * _NUCLEAR_MC
    )
    out["uk_dynamic_marginal"] = (
        uk_gas_w * out["uk_merit_order_cost"] + (1 - uk_gas_w) * _NUCLEAR_MC
    )

    # Opportunity cost (FR = min of internal vs import)
    neighbor_spots = pd.concat(
        [
            _safe_col(out, "de_spot_la"),
            _safe_col(out, "be_spot_la"),
            _safe_col(out, "ch_spot_la"),
            _safe_col(out, "es_spot_la"),
        ],
        axis=1,
    )
    fr_import_price = neighbor_spots.min(axis=1) + _TRANSPORT_COST
    out["fr_import_price"] = fr_import_price
    out["fr_opportunity_cost"] = pd.concat(
        [out["fr_dynamic_marginal"], fr_import_price], axis=1
    ).min(axis=1)

    uk_import_spots = pd.concat(
        [
            _safe_col(out, "fr_spot_la"),
            _safe_col(out, "be_spot_la"),
            _safe_col(out, "nl_spot_la"),
        ],
        axis=1,
    )
    uk_import_price = uk_import_spots.min(axis=1) + _TRANSPORT_COST * 2
    out["uk_import_floor"] = uk_import_price
    out["uk_opportunity_cost"] = pd.concat(
        [out["uk_dynamic_marginal"], uk_import_price], axis=1
    ).min(axis=1)

    # Scarcity exponential barrier
    fr_s_clip = fr_s.clip(upper=0.98)
    uk_s_clip = uk_s.clip(upper=0.98)
    out["fr_scarcity_barrier"] = (
        out["fr_spark_spread"] * (1.0 / (1.0 - fr_s_clip) ** _SCAR_BARRIER_P)
    )
    out["uk_scarcity_barrier"] = (
        out["uk_spark_spread"] * (1.0 / (1.0 - uk_s_clip) ** _SCAR_BARRIER_P)
    )

    # Load-price signal (7d rolling: "at this load level, expect this price")
    fr_la = _safe_col(out, "fr_spot_la")
    uk_la = _safe_col(out, "uk_spot_la")
    fr_resid = out["fr_residual_load"]
    uk_resid = out["uk_residual_load"]
    fr_load = out["fr_load_f"]

    fr_p168 = fr_la.rolling(168, min_periods=24).mean()
    fr_r168 = fr_resid.rolling(168, min_periods=24).mean()
    uk_p168 = uk_la.rolling(168, min_periods=24).mean()
    uk_r168 = uk_resid.rolling(168, min_periods=24).mean()

    out["fr_price_per_mw_7d"] = fr_p168 / fr_r168.clip(lower=1)
    out["uk_price_per_mw_7d"] = uk_p168 / uk_r168.clip(lower=1)
    out["fr_load_price_signal_7d"] = out["fr_price_per_mw_7d"] * fr_resid
    out["uk_load_price_signal_7d"] = out["uk_price_per_mw_7d"] * uk_resid

    fr_l168 = fr_load.rolling(168, min_periods=24).mean()
    out["fr_load_price_signal_load"] = fr_p168 * (fr_load / fr_l168.clip(lower=1))

    # Hydro opportunity cost = rolling max spark over 168h
    out["fr_hydro_opp_cost"] = (
        out["fr_spark_spread"].rolling(168, min_periods=24).max()
    )
    out["uk_hydro_opp_cost"] = (
        out["uk_spark_spread"].rolling(168, min_periods=24).max()
    )

    # Basis v2 = spot_la − opportunity_cost
    out["fr_basis_v2"] = fr_la - out["fr_opportunity_cost"]
    out["uk_basis_v2"] = uk_la - out["uk_opportunity_cost"]
    out["fr_basis_v2_lag_48h"] = out["fr_basis_v2"].shift(24)
    out["uk_basis_v2_lag_48h"] = out["uk_basis_v2"].shift(24)
    out["fr_basis_v2_roll_24h_mean"] = out["fr_basis_v2"].rolling(24, min_periods=1).mean()
    out["uk_basis_v2_roll_24h_mean"] = out["uk_basis_v2"].rolling(24, min_periods=1).mean()

    return out


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def apply_full_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all 30 feature categories sequentially.

    Designed for a DataFrame with a DatetimeCET index (no datetime_CET column).
    All transformations are stateless: no target leakage.
    """
    logger.info("Applying full feature engineering pipeline...")

    # Tier 1 — time and calendar (no other features needed)
    df = engineer_calendar_effects(df)
    df = engineer_holiday_features(df)

    # Tier 2 — fundamental physical features (depend on raw columns only)
    df = engineer_fundamental_market(df)
    df = engineer_renewable_penetration(df)
    df = engineer_renewable_dynamics(df)
    df = engineer_nuclear_features(df)
    df = engineer_hydro_features(df)
    df = engineer_interconnectors(df)
    df = engineer_regional_features(df)
    df = engineer_river_temperature(df)

    # Tier 3 — price-based features (depend on spot_la columns)
    df = engineer_price_features(df)

    # Tier 4 — rolling stats (needed by stochastic signals)
    df = engineer_rolling_statistics(df)

    # Tier 5 — features that depend on previous categories
    df = engineer_interaction_features(df)
    df = engineer_nonlinear_transforms(df)
    df = engineer_momentum_features(df)
    df = engineer_advanced_supply_demand(df)
    df = engineer_residual_ramps(df)
    df = engineer_multi_efficiency_spark(df)
    df = engineer_advanced_interconnection(df)
    df = engineer_regime_zscores(df)
    df = engineer_stochastic_signals(df)
    df = engineer_asinh_transforms(df)
    df = engineer_nuclear_shortfall(df)
    df = engineer_atc_ratios(df)

    # Tier 6 — high-level derived features (depend on scarcity, shortfall, merit order)
    df = engineer_market_specific_features(df)
    df = engineer_price_formation_signals(df)
    df = engineer_fr_continent_features(df)
    df = engineer_uk_island_features(df)
    df = engineer_regime_features(df)
    df = engineer_advanced_price_proxies(df)

    # Defragment to avoid PerformanceWarning
    df = df.copy()

    logger.info(f"Feature engineering complete. Final shape: {df.shape}")
    return df


# Alias for callers that use the Paul-style signature
def build_features(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """Alias for apply_full_feature_engineering (config param ignored)."""
    return apply_full_feature_engineering(df)
