"""
Global constants and data schemas for the Day-Ahead Market (DAM) forecasting pipeline.
"""

# Market Zones
ALL_ZONES = ["AT", "BE", "CH", "CZ", "DE", "DK1", "DK2", "FR", "NL", "NO2", "PL", "SE4"]

# Target Variable
TARGET_COL = "Spot_Price"

# ENTSO-E Generation Groupings (Merit Order Effect)
RENEWABLE_COLS = ["SOLAR", "WIND-OFFSHORE", "WIND-ONSHORE"]
BASELOAD_COLS = ["NUCLEAR", "HYDRO-ROR", "BIOMASS"]
DISPATCHABLE_COLS = ["FOSSIL-GAS", "HARD-COAL", "LIGNITE"]

# Expected Input File Suffixes
FILE_SUFFIXES = {
    "price": "-spot-price.csv",
    "load": "-total-load.csv",
    "generation": "-generation.csv",
    "flows": "-physical-flows-in.csv",
    "weather": "-open-meteo" # Note: Weather has coordinates in the name, needs partial matching
}