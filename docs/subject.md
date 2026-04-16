Overview
Welcome to the InCommodities Case Crunch 2026 - Turning data into a competitive advantage.

This year’s competition challenges you to build a model that can forecast hourly day-ahead electricity spot prices in United Kingdom (UK) and France (FR). You will work with real-world market data and face many of the same complexities our trading teams deal with every day.

All the data and information needed to build your models can be found here on the InCommodities Case Crunch 2026 Kaggle competition. You can use this setup to train you model by submitting your predictions, and having Kaggle evaluate it agains the out of sample test set - on the Leaderboard you can see how your predictions perform compared to the other teams' submissions. This is merely a way for you to see how you are doing, and will not count towards the final evaluation. That will take place on the day of the finals at the InCommodities head quarters, where we will put your models to the test in a live setup. But more on that later.

Good luck - and enjoy the challenge.

Start

5 hours ago
Close
3 days to go
Description
The european power market
The European power market is a tightly coupled, cross-border system where electricity prices are formed hourly based on supply, demand, and transmission constraints. Market integration allows power to flow across countries, linking price formation in one region to conditions in its neighbors.

At InCommodities, the Forward Power team trades electricity futures, spot-linked products, and transmission capacity. Our profitability heavily depends on accurately forecasting spot prices across Europe - not just daily averages, but hour by hour.

Spot prices emerge from the interaction of:

Electricity demand
Renewable generation (wind, solar, hydro)
Thermal generation costs (fuel prices, CO₂ prices, efficiencies)
Plant outages and maintenance
Cross-border transmission constraints
And more
Exploring, analyzing, and understanding these dynamics - and eventually incorporating them into your models - will be central to your success in this challenge.

From fundamentals to prices
Electricity prices are set by the marginal power plant required to meet demand in each hour. This depends on the power supply stack, which orders generation units from lowest to highest marginal cost.

The stack is not static:

Renewable output varies with weather
Thermal units may be unavailable
Hydro availability and strategy change over time
Interconnector constraints alter effective supply
As a result, prices can change sharply from one hour to the next. Capturing this behaviour is where data science and machine learning become powerful tools.

Day-ahead price formation
The prices you will forecast are day-ahead spot prices - the hourly prices determined in auctions held the day before electricity is delivered.

France participates in the Single day-ahead coupling (SDAC), the integrated European day-ahead market operated by EPEX SPOT. Each day, generators and consumers submit buy and sell orders for every hour of the following day. At 12:00 CET, the EUPHEMIA algorithm clears all orders simultaneously across coupled European bidding zones, determining a single market-clearing price per zone per hour. The algorithm accounts for cross-border transmission capacity, allowing power to flow where it is most needed - subject to interconnector limits.

The United Kingdom operates its own day-ahead auction, separate from the European coupled market since Brexit. The UK day-ahead prices in this competition come from the Nord Pool N2EX auction, which clears daily at 09:10 UK time. The auction follows a similar principle - aggregated supply and demand curves are matched to produce a clearing price for each hour - but UK prices are determined independently, in GBP. Cross-border trade between the UK and the continent is facilitated through explicit capacity auctions on the interconnectors, meaning that the link between UK and European prices depends on both the price spread and the available transmission capacity.

In both markets, the clearing price in each hour reflects the cost of the most expensive generation unit needed to meet demand - the marginal price. This is why understanding the supply stack, renewable generation, fuel costs, and interconnector flows is key to building an accurate forecast.

Your challenge
Step into the role of a quantitative analyst at InCommodities
You have joined the InCommodities Forward Power team as a quantitative analyst. Your task is to forecast hourly electricity spot prices for a specified market and horizon using the provided data.

Accurate price forecasts are essential for:

Trading and risk management
Valuation of assets and contracts
Understanding market stress and volatility
This competition mirrors a real forecasting problem we actively work on - simplified, but realistic.

Your assignment
Using the datasets provided, you must:

Build a model that predicts hourly spot prices for the out-of-sample period for the power price areas: UK and FR.
Submit forecasts in the required format found in the sample submission on the Data tab.
Fine-tune your model to be as accurate as possible.
You are free to choose your approach:

Machine learning models
Time-series methods
AI or deep learning approaches
Hybrid or structural-inspired models
Feature engineering grounded in market fundamentals


A note on the test data
To allow you to build models with lagged features - for example, using recent historical prices as inputs - the test dataset includes actual spot prices alongside the feature variables.

We trust you to use this data responsibly. The Kaggle leaderboard is there to help you gauge your model's performance and iterate on your approach. Uploading the actual prices as your forecast would render the leaderboard meaningless - for you and for everyone else. Please use it as the development tool it is intended to be.

How to approach the task
1. Access the data
Navigate to the Data tab and download the available files. These include historical observations, a wide range of potential explanatory variables, and a test set. A more detailed description can also be found on the Data tab.

2. Explore and understand
Analyse the data carefully:

Temporal patterns (daily, weekly, seasonal)
Price spikes and volatility
Relationships between prices and drivers such as demand, renewables, availability, etc.
3. Build a model
Design a forecasting model that reflects both statistical rigour and market intuition. Keep in mind - simplicity is not a weakness if it is well-justified.

4. Train and validate
Train your model on historical data. Validate it properly to ensure robustness and avoid overfitting.

5. Generate forecasts
Apply your trained model to the test set and generate hourly price predictions.

6. Submit and iterate
Upload your predictions via the Submissions tab. You may submit multiple times and refine your approach throughout the competition.

Evaluation
Kaggle leaderboard
Submissions are ranked using Root Mean Squared Error (RMSE) on a hidden subset of the test data. The leaderboard gives you a sense of how your model compares, but be mindful of overfitting to the public score.

Final Evaluation
The final evaluation will take place on the last day of the competition and will put your forecasting models to the test in a live setting at our office. The better your model, the better your chances - so use the time leading up to the final to build the strongest model you can.

Final Notes
Electricity markets are fast, noisy, and unforgiving - but deeply rewarding to understand.

This competition is your opportunity to demonstrate how data science and machine learning can be applied to one of the most challenging real-world forecasting problems.

Good luck.



Dataset Description
The goal of this competition is to forecast hourly day-ahead electricity spot prices for France (fr_spot) and the United Kingdom (uk_spot) over an 8-month out-of-sample period. All prices are denominated in EUR/MWh.

All forecast features in the dataset are based on information available at 09:00 CET on the day before delivery — before the Nord Pool N2EX auction for the UK and the SDAC auction (EPEX SPOT) for France. This reflects the realistic information set a trader would have when creating their day-ahead forecasts.

Permitted data
Participants may only use the provided dataset as their source of input data. No external datasets, APIs, or supplementary data files (e.g., additional weather forecasts, price histories, or generation data from third-party providers) may be incorporated into models. However, participants are encouraged to perform feature engineering on the provided columns, including constructing derived features (e.g., residual load as demand minus renewables generation), extracting temporal features from the datetime column (e.g., hour-of-day, day-of-week, month), and generating calendar-based features such as public holiday indicators for France and the United Kingdom using standard Python libraries (e.g., holidays, workalendar). Participants may also incorporate general domain knowledge about the French and British power markets — such as known characteristics of the energy mix, typical demand patterns, or market structure — as modelling assumptions or hard-coded parameters, provided this does not involve importing additional time-series data or forecasts beyond what is supplied.

Column naming conventions
All feature columns follow a naming convention that encodes both the data source and its temporal availability:

Suffix	Meaning	Description
_f	Forecast	A forecast value published before delivery. Available for the delivery day at 09:00 CET the day before. Hourly resolution.
_la	Lagged actual	An actual (realized) value, lagged by 24 hours. This means each row contains the actual value from the same hour on the previous day. This ensures no lookahead bias — only information known before the auction is used. Hourly resolution.
(no suffix)	Daily	Daily-resolution values (gas prices, emission prices). These appear as a single value at 00:00 CET each day, with NaN for the remaining 23 hours. These columns are sparsely populated — see the note on missing data below.
Files
x_train.csv
Training set features. Each row represents one hour. Covers the period 2022-07-01 00:00 CET to 2024-06-30 23:00 CET (17,544 hours). Contains 111 columns.

Identifier columns (3):
Column	Description
id	Unique row identifier (integer, 0–17,543). Joins to y_train.csv.
datetime_CET	Hour start timestamp in Central European Time (CET/CEST). Format: YYYY-MM-DD HH:MM:SS.
datetime_UTC	Hour start timestamp in UTC. Format: YYYY-MM-DD HH:MM:SS.
The datetime columns mark the beginning of the hour. For example, if datetime_CET is 2022-07-01 00:00 then the row covers the first hour of July 1st 2022.

Hourly forecast features (suffix _f, 67 columns):
Category	Columns	Unit	Description
Load forecasts	at_load_f, be_load_f, de_load_f, dk1_load_f, dk2_load_f, es_load_f, fr_load_f, itn_load_f, nl_load_f, uk_load_f	MW	Forecasted electricity demand for Austria, Belgium, Germany, Denmark (DK1, DK2), Spain, France, Italy North, Netherlands, and UK.
Solar generation forecasts	be_solar_f, de_solar_f, dk1_solar_f, dk2_solar_f, es_solar_f, fr_solar_f, nl_solar_f, uk_solar_f	MW	Forecasted solar PV generation.
Wind generation forecasts	be_wind_f, de_wind_f, dk1_wind_f, dk2_wind_f, es_wind_f, fr_wind_f, nl_wind_f, uk_wind_f	MW	Forecasted wind generation.
Hydro reservoir generation	at_hydro_res_f, ch_hydro_res_f, fr_hydro_res_f	MW	Forecasted power production from hydro reservoir plants for Austria, Switzerland, and France.
Hydro run-of-river generation	at_hydro_ror_f, ch_hydro_ror_f, es_hydro_ror_f, fr_hydro_ror_f	MW	Forecasted power production from hydro run-of-river plants for Austria, Switzerland, Spain, and France.
River temperatures	de_river_temp_danube_donauworth_f, de_river_temp_danube_ingolstadt_f, fr_river_temp_rhine_rheinfelden_f, fr_river_temp_rhone_lyon_f	°C	Forecasted river water temperatures. Relevant for thermal power plant cooling constraints.
Nuclear available capacity	fr_nuclear_avcap_f, uk_nuclear_avcap_f	MW	Forecasted available nuclear generation capacity for France and UK, accounting for planned and unplanned outages (UMM data).
Gas-fired available capacity	de_gas_avcap_f, fr_gas_avcap_f, uk_gas_avcap_f	MW	Forecasted available gas-fired generation capacity for Germany, France, and UK (UMM data).
Biomass available capacity	uk_biomass_avcap_f	MW	Forecasted available biomass generation capacity for UK (UMM data).
Interconnector ATC	atc_be-uk_f, atc_dk1-uk_f, atc_fr-uk-1_f, atc_fr-uk-2_f, atc_fr-uk-3_f, atc_nl-uk_f, atc_uk-be_f, atc_uk-dk1_f, atc_uk-fr-1_f, atc_uk-fr-2_f, atc_uk-fr-3_f, atc_uk-nl_f	MW	Available Transfer Capacity on UK interconnectors. See interconnector reference table below.
Interconnector NTC	ntc_be-uk_f, ntc_dk1-uk_f, ntc_fr-uk-1_f, ntc_fr-uk-2_f, ntc_fr-uk-3_f, ntc_nl-uk_f, ntc_uk-be_f, ntc_uk-dk1_f, ntc_uk-fr-1_f, ntc_uk-fr-2_f, ntc_uk-fr-3_f, ntc_uk-nl_f	MW	Net Transfer Capacity on UK interconnectors.
ATC and NTC explained:

The interconnector capacity columns use two related measures defined by ENTSO-E:

NTC (Net Transfer Capacity) is the maximum power that can be transferred across an interconnection without violating operational security limits. It represents the theoretical ceiling based on system stability and equipment constraints.
ATC (Available Transfer Capacity) is the portion of NTC that remains available for new transfers: ATC = NTC − TRM − AAC, where TRM is a transmission reliability margin (safety buffer) and AAC is already allocated capacity (committed transfers).
In practice, NTC sets the upper bound while ATC reflects what is actually available for day-ahead auction allocation on a given hour.

Lagged actual features (suffix _la, 34 columns):
These are actual realized values from the same hour on the previous day (shifted by 24 hours).

Category	Columns	Unit	Description
Lagged spot prices	at_spot_la, be_spot_la, ch_spot_la, de_spot_la, dk1_spot_la, dk2_spot_la, es_spot_la, fr_spot_la, nl_spot_la, uk_spot_la	EUR/MWh	Day-ahead spot prices from the previous day, for 10 European bidding zones.
Lagged interconnector costs	cost_be-uk_la, cost_dk1-uk_la, cost_fr-uk-1_la, cost_fr-uk-2_la, cost_fr-uk-3_la, cost_nl-uk_la, cost_uk-be_la, cost_uk-dk1_la, cost_uk-fr-1_la, cost_uk-fr-2_la, cost_uk-fr-3_la, cost_uk-nl_la	EUR/MWh	Capacity auction clearing prices on UK interconnectors from the previous day.
Lagged interconnector flows	flow_be-uk_la, flow_dk1-uk_la, flow_fr-uk-1_la, flow_fr-uk-2_la, flow_fr-uk-3_la, flow_nl-uk_la, flow_uk-be_la, flow_uk-dk1_la, flow_uk-fr-1_la, flow_uk-fr-2_la, flow_uk-fr-3_la, flow_uk-nl_la	MW	Day-ahead scheduled commercial power flows on UK interconnectors from the previous day.
Daily features (no suffix, 7 columns):
These are daily-resolution values. Each value appears once per day at the 00:00 CET row; all other hours for that day are NaN. These columns are sparsely populated — see the note on missing data below.

Category	Columns	Unit	Description
Gas prices	de_gas, es_gas, fr_gas, nl_gas, uk_gas	EUR/MWh	Day-ahead natural gas hub prices for Germany (THE), Spain (PVB), France (PEG), Netherlands (TTF), and UK (NBP).
Emission prices	eu_emission, uk_emission	EUR/tCO2	Carbon emission allowance prices — EU ETS (EUA) and UK ETS (UKA) daily VWAP.
y_train.csv
Training set target values. Contains 17,544 rows with columns:

Column	Description
id	Row identifier (joins to x_train.csv).
datetime_CET	Hour start timestamp in CET.
datetime_UTC	Hour start timestamp in UTC.
fr_spot	Hourly France day-ahead spot price (EUR/MWh).
uk_spot	Hourly UK day-ahead spot price (EUR/MWh).
x_test.csv
Test set features with the same 111 columns as x_train.csv. Covers the period 2024-07-01 00:00 CET to 2025-02-28 23:00 CET (5,833 hours). IDs range from 17,544 to 23,376. Your task is to predict fr_spot and uk_spot for each row.

sample_submission.csv
A valid submission file with the correct format. Contains 5,833 rows with columns id, fr_spot, and uk_spot (all zeros as placeholder values). Your submission must match this format exactly.

Interconnector reference
The UK is connected to continental Europe via six submarine interconnectors. The column naming convention {from}-{to} indicates the direction of capacity/flow.

Column suffix	Interconnector	Route	Capacity
be-uk / uk-be	NEMO	Belgium ↔ UK	~1,000 MW
nl-uk / uk-nl	BritNed	Netherlands ↔ UK	~1,000 MW
fr-uk-1 / uk-fr-1	IFA1	France ↔ UK	~2,000 MW
fr-uk-2 / uk-fr-2	IFA2	France ↔ UK	~1,000 MW
fr-uk-3 / uk-fr-3	ElecLink	France ↔ UK (via Channel Tunnel)	~1,000 MW
dk1-uk / uk-dk1	Viking Link	Denmark (DK1) ↔ UK	~1,400 MW
Missing data
Some columns are sparsely populated or become available only partway through the dataset. This is not an error - it reflects the real-world availability of these data sources.

Daily columns (gas, emissions): These columns have values only at the 00:00 CET hour of each day. Even then, they are only available for a subset of dates (~3–5% of the training set). They become more consistently available toward the end of the training period and into the test period.

All hourly forecast columns (_f) and lagged actual columns (_la) have full or near-full coverage across both train and test sets. Models should handle missing values gracefully - either by imputation, by using models that natively support NaN (e.g., tree-based methods), or by excluding sparse columns.

Evaluation
Submissions are evaluated on Root Mean Squared Error (RMSE) averaged across both target columns (fr_spot and uk_spot).

Note: While RMSE is the competition metric, consider experimenting with different loss functions during training (e.g., MAE, Huber, quantile loss). Models trained on a single objective can overfit to that metric's assumptions - using alternative or combined loss functions may produce more robust forecasts, especially given the heavy-tailed nature of electricity spot prices.

Public vs. private leaderboard: The public leaderboard shown during the competition is calculated on only 60% of the test data. The remaining 40% is reserved for the private leaderboard, which is revealed only after the competition ends. The final ranking is determined by the private leaderboard score, calculated on 100% of the test data.

This means repeatedly submitting forecasts to minimize your public leaderboard score is risky - you may be overfitting to the 60% public split rather than improving your model's true generalization. A submission that ranks well on the public leaderboard may perform poorly on the private leaderboard if it has been tuned too aggressively to the public subset. Focus on building a robust model with sound cross-validation rather than chasing the public score.

Disclaimer on data leakage:
The dataset includes lagged actual spot prices (fr_spot_la, uk_spot_la) as features. Since the test set targets (fr_spot, uk_spot) are the same series shifted by 24 hours, it is technically possible to reverse-engineer near-perfect predictions on the Kaggle leaderboard by exploiting this overlap. However, doing so would be pointless - the Kaggle leaderboard has zero influence on the final competition standings. On the day of the finals, participants will run their models live on unseen future data, where this trick will not work. Building a model that genuinely forecasts well - that is what will help you take home the grand price!