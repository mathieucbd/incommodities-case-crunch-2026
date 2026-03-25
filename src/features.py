"""Feature block routing for block-specific ensemble."""

from typing import NamedTuple


class FeatureBlocks(NamedTuple):
    """Container for feature blocks."""
    block_a: list[str]  # Autoregressive & Calendar
    block_b: list[str]  # Fundamentals & Scarcity
    block_c: list[str]  # Interconnections


def get_feature_blocks(all_features: list[str]) -> FeatureBlocks:
    """
    Categorize features into three mutually exclusive blocks.

    Block A (Autoregressive & Calendar): Calendar features, holidays, lagged spot prices,
        rolling spot price stats, price momentum, and AR signals.

    Block B (Fundamentals & Scarcity): Raw forecasts (_f), residual load, thermal need,
        nuclear/hydro metrics, spark spreads, merit order cost, scarcity ratios, and
        supply-demand indicators.

    Block C (Interconnections): ATC/NTC capacities, per-cable ratios, interconnector flows,
        costs, utilization, and congestion flags.

    Args:
        all_features: List of all available feature names.

    Returns:
        FeatureBlocks containing three mutually exclusive lists (block_a, block_b, block_c).
    """
    block_a = []
    block_b = []
    block_c = []
    unclassified = []

    # Patterns for Block C (Interconnections) — most specific, check first
    block_c_patterns = {
        "starts_with": ("atc_", "ntc_"),
        "contains": (
            "_atc_", "_ntc_", "_flow_la", "_net_flow_la", "_avg_cost_la",
            "_cost_spread_la", "_uk_utilization", "_fr_utilization", "_uk_congested",
            "any_direction_congested", "_unused_capacity", "flow_over_atc"
        ),
        "exact": (
            "all_to_uk_atc", "all_from_uk_atc", "max_utilization_to_uk",
            "total_net_import_uk_la", "fr_uk_atc_change_24h"
        )
    }

    # Patterns for Block B (Fundamentals & Scarcity) — check after Block C
    block_b_patterns = {
        "ends_with": ("_f",),
        "contains": (
            "_residual_load", "_thermal_need", "_thermal_floor", "_nuclear_shortfall",
            "_spark_spread", "_merit_order_cost", "_scarcity_ratio", "_baseload_gap",
            "_supply_demand_ratio", "_security_margin", "_hydro_total", "_wind_pen",
            "_solar_pen", "_renewable_pen", "_nuclear_pct_of_load", "_nuclear_low",
            "_nuclear_change_", "_nuclear_ramp_", "_nuclear_deviation", "_nuclear_rolling_",
            "_load_change_", "_wind_change_", "_residual_change_", "_wind_ramp_",
            "_solar_ramp_", "_wind_volatility_", "_gas_margin", "_total_dispatchable",
            "_wind_high", "continental_", "nordic_", "iberian_", "_spark_ocgt", "_spark_ccgt",
            "_dynamic_marginal", "_implied_re_surplus", "_scarcity_barrier", "_scarcity_critical",
            "_scarcity_extreme", "_gas_utilization", "_gas_headroom", "_capacity_margin",
            "_gas_cost_per_mw", "_self_sufficiency", "_thermal_gap", "_gas_on_margin",
            "_negative_price_risk", "_fossil_or_import_need", "_wind_x_gas", "_nuclear_x_gas",
            "_residual_x_spark", "_thermal_need_x_gas", "_stress_index", "_load_ramp_",
            "_residual_ramp_", "dark_doldrums", "_zscore_14d", "carbon_to_gas_ratio",
            "euro_scarcity_ratio", "wind_tier1_pen", "_wind_f_clipped", "_gas_sqrt",
            "_asinh_spark", "_residual_load_squared", "_wind_pen_squared", "_spark_spread_log",
            "_nuke_shortfall_x_gas", "iberian_exception", "_oversupply_mw",
            # Hydro & gas columns
            "de_gas", "fr_gas", "uk_gas", "es_gas", "nl_gas", "eu_emission", "uk_emission"
        )
    }

    # Patterns for Block A (Autoregressive & Calendar) — everything else
    block_a_patterns = {
        "exact": (
            "hour", "day_of_week", "month", "doy", "week_of_year", "quarter",
            "is_weekend", "is_business_hour", "is_morning_ramp", "is_evening_peak",
            "is_night", "is_solar_hours"
        ),
        "contains": (
            "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
            "doy_sin", "doy_cos", "hour_x_dow",
            "is_holiday", "is_bridge_day", "days_to_next_holiday", "days_since_last_holiday",
            "_spot_la", "_spot_la_roll_", "_spot_lag_", "_spot_la_ewm_", "_spot_la_deviation_",
            "spread_", "_vs_continental_la", "continental_avg_spot_la",
            "_price_acceleration", "_gas_momentum_", "_spot_la_h", "_mean_reversion_strength",
            "_vol_ratio", "_jump_", "_asinh_spot_la", "_spot_la_log", "_basis_v2",
            "_intraday_amplitude"
        )
    }

    for feat in all_features:
        # Check Block C (Interconnections)
        is_block_c = False

        # Check starts_with patterns
        if any(feat.startswith(p) for p in block_c_patterns["starts_with"]):
            is_block_c = True
        # Check contains patterns
        elif any(p in feat for p in block_c_patterns["contains"]):
            is_block_c = True
        # Check exact patterns
        elif feat in block_c_patterns["exact"]:
            is_block_c = True

        if is_block_c:
            block_c.append(feat)
            continue

        # Check Block B (Fundamentals & Scarcity)
        is_block_b = False

        # Check ends_with patterns
        if any(feat.endswith(p) for p in block_b_patterns["ends_with"]):
            is_block_b = True
        # Check contains patterns
        elif any(p in feat for p in block_b_patterns["contains"]):
            is_block_b = True

        if is_block_b:
            block_b.append(feat)
            continue

        # Check Block A (Autoregressive & Calendar) — default
        is_block_a = False

        # Check exact patterns
        if feat in block_a_patterns["exact"]:
            is_block_a = True
        # Check contains patterns
        elif any(p in feat for p in block_a_patterns["contains"]):
            is_block_a = True

        if is_block_a:
            block_a.append(feat)
        else:
            unclassified.append(feat)

    # Default: any unclassified features → Block A (catch-all for robustness)
    block_a.extend(unclassified)

    return FeatureBlocks(block_a, block_b, block_c)


def describe_blocks(blocks: FeatureBlocks) -> None:
    """Print a compact summary of block sizes and sample features."""
    print(f"  Block A (Calendar/AR)    : {len(blocks.block_a):3d} features")
    if blocks.block_a:
        print(f"    Samples: {', '.join(blocks.block_a[:3])}{'...' if len(blocks.block_a) > 3 else ''}")
    print(f"  Block B (Fundamentals)   : {len(blocks.block_b):3d} features")
    if blocks.block_b:
        print(f"    Samples: {', '.join(blocks.block_b[:3])}{'...' if len(blocks.block_b) > 3 else ''}")
    print(f"  Block C (Interconnections): {len(blocks.block_c):3d} features")
    if blocks.block_c:
        print(f"    Samples: {', '.join(blocks.block_c[:3])}{'...' if len(blocks.block_c) > 3 else ''}")
    print(f"  Total: {sum(len(b) for b in blocks)} features")
