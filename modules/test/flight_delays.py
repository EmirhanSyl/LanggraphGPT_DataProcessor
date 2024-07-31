import pandas as pd
import numpy as np
from modules import text_cleaner, missing_value_handler, language_processor, data_type_converter, outlier_handler, data_visualizer, data_type_converter

# Load Data
df = pd.read_csv("../../dataset/flight_delays.csv")

# Get missing values ratio
missing_handler = missing_value_handler.MissingValueHandler()
missing_handler.print_nan_ratios(df)

# Tail num is like a licence plate of a plate. Can be filled with constant "UNKNOWN"
print(pd.api.types.is_string_dtype(df["TAIL_NUM"]))
df = missing_handler.replace_missing_values(df, 'TAIL_NUM', missing_handler.Strategy.CONSTANT, 'UNKNOWN')  # 1.08%

# Dep time of the plane. Column type is float although column has the clock values. Performed median replacement
df = missing_handler.replace_missing_values(df, 'DEP_TIME', missing_handler.Strategy.MEDIAN)  # 3.67%
df = missing_handler.replace_missing_values(df, 'WHEELS_OFF', missing_handler.Strategy.MEDIAN)  # 3.76%
df = missing_handler.replace_missing_values(df, 'WHEELS_ON', missing_handler.Strategy.MEDIAN)  # 3.83%
df = missing_handler.replace_missing_values(df, 'ARR_TIME', missing_handler.Strategy.MEDIAN)  # 3.83%

# Unknown data. applying mean replacement...
df = missing_handler.replace_missing_values(df, 'TAXI_OUT', missing_handler.Strategy.MEAN)  # 3.76%
df = missing_handler.replace_missing_values(df, 'TAXI_IN', missing_handler.Strategy.MEAN)  # 3.83%
df = missing_handler.replace_missing_values(df, 'DEP_DELAY', missing_handler.Strategy.MEAN)  # 3.69%
df = missing_handler.replace_missing_values(df, 'ARR_DELAY', missing_handler.Strategy.MEAN)  # 3.69%
df = missing_handler.replace_missing_values(df, 'ARR_DELAY_NEW', missing_handler.Strategy.MEAN)  # 4.07%

# Missing values on Delay causes column can be filled as categorical -1 and 0
df = missing_handler.replace_missing_values(df, 'CARRIER_DELAY', missing_handler.Strategy.CONSTANT, -1)  # 76.89%
df = missing_handler.replace_missing_values(df, 'WEATHER_DELAY', missing_handler.Strategy.CONSTANT, -1)  # 76.89%
df = missing_handler.replace_missing_values(df, 'NAS_DELAY', missing_handler.Strategy.CONSTANT, -1)  # 76.89%
df = missing_handler.replace_missing_values(df, 'SECURITY_DELAY', missing_handler.Strategy.CONSTANT, -1)  # 76.89%
df = missing_handler.replace_missing_values(df, 'LATE_AIRCRAFT_DELAY', missing_handler.Strategy.CONSTANT, -1)  # 76.89%

# Huge Amount of missing data. Applying remove column operation
df = missing_handler.replace_missing_values(df, 'CANCELLATION_CODE', missing_handler.Strategy.REMOVE_COLUMN)  # 96.21%
df = missing_handler.replace_missing_values(df, 'FIRST_DEP_TIME', missing_handler.Strategy.REMOVE_COLUMN)  # 99.12%
df = missing_handler.replace_missing_values(df, 'TOTAL_ADD_GTIME', missing_handler.Strategy.REMOVE_COLUMN)  # 99.12%
df = missing_handler.replace_missing_values(df, 'DIV_AIRPORT_LANDINGS', missing_handler.Strategy.REMOVE_COLUMN)  # 99.12%
df = missing_handler.replace_missing_values(df, 'DIV_ACTUAL_ELAPSED_TIME', missing_handler.Strategy.REMOVE_COLUMN)  # 99.76%
df = missing_handler.replace_missing_values(df, 'DIV1_AIRPORT', missing_handler.Strategy.REMOVE_COLUMN)  # 99.69%
df = missing_handler.replace_missing_values(df, 'DIV1_WHEELS_ON', missing_handler.Strategy.REMOVE_COLUMN)  # 99.69%
df = missing_handler.replace_missing_values(df, 'DIV1_TOTAL_GTIME', missing_handler.Strategy.REMOVE_COLUMN)  # 99.69%
df = missing_handler.replace_missing_values(df, 'DIV1_WHEELS_OFF', missing_handler.Strategy.REMOVE_COLUMN)  # 99.69%
df = missing_handler.replace_missing_values(df, 'DIV1_TAIL_NUM', missing_handler.Strategy.REMOVE_COLUMN)  # 99.76%
df = missing_handler.replace_missing_values(df, 'DIV2_AIRPORT', missing_handler.Strategy.REMOVE_COLUMN)  # 99.99%
df = missing_handler.replace_missing_values(df, 'DIV2_WHEELS_ON', missing_handler.Strategy.REMOVE_COLUMN)  # 99.99%
df = missing_handler.replace_missing_values(df, 'DIV2_TOTAL_GTIME', missing_handler.Strategy.REMOVE_COLUMN)  # 99.99%
df = missing_handler.replace_missing_values(df, 'DIV2_WHEELS_OFF', missing_handler.Strategy.REMOVE_COLUMN)  # 100.00%
df = missing_handler.replace_missing_values(df, 'DIV2_TAIL_NUM', missing_handler.Strategy.REMOVE_COLUMN)  # 100.00%
df = missing_handler.replace_missing_values(df, 'DIV3_AIRPORT', missing_handler.Strategy.REMOVE_COLUMN)  # 100.00%
df = missing_handler.replace_missing_values(df, 'DIV3_WHEELS_ON', missing_handler.Strategy.REMOVE_COLUMN)  # 100.00%
df = missing_handler.replace_missing_values(df, 'DIV3_TOTAL_GTIME', missing_handler.Strategy.REMOVE_COLUMN)  # 100.00%
df = missing_handler.replace_missing_values(df, 'DIV3_WHEELS_OFF', missing_handler.Strategy.REMOVE_COLUMN)  # 100.00%
df = missing_handler.replace_missing_values(df, 'DIV3_TAIL_NUM', missing_handler.Strategy.REMOVE_COLUMN)  # 100.00%
df = missing_handler.replace_missing_values(df, 'DIV4_AIRPORT', missing_handler.Strategy.REMOVE_COLUMN)  # 100.00%
df = missing_handler.replace_missing_values(df, 'DIV4_WHEELS_ON', missing_handler.Strategy.REMOVE_COLUMN)  # 100.00%
df = missing_handler.replace_missing_values(df, 'DIV4_TOTAL_GTIME', missing_handler.Strategy.REMOVE_COLUMN)  # 100.00%
df = missing_handler.replace_missing_values(df, 'DIV4_WHEELS_OFF', missing_handler.Strategy.REMOVE_COLUMN)  # 100.00%
df = missing_handler.replace_missing_values(df, 'DIV4_TAIL_NUM', missing_handler.Strategy.REMOVE_COLUMN)  # 100.00%
df = missing_handler.replace_missing_values(df, 'DIV5_AIRPORT', missing_handler.Strategy.REMOVE_COLUMN)  # 100.00%
df = missing_handler.replace_missing_values(df, 'DIV5_WHEELS_ON', missing_handler.Strategy.REMOVE_COLUMN)  # 100.00%
df = missing_handler.replace_missing_values(df, 'DIV5_WHEELS_OFF', missing_handler.Strategy.REMOVE_COLUMN)  # 100.00%
df = missing_handler.replace_missing_values(df, 'DIV5_TAIL_NUM', missing_handler.Strategy.REMOVE_COLUMN)  # 100.00

print(f"\n{'_'*60} END OF MISSING VALUE HANDLING PROCESS {'_'*60}\n")
missing_handler.print_nan_ratios(df)
print(df.head())


# _________ OUTLIER HANDLING STARTS ___________
outlier_handler = outlier_handler.OutlierHandler()
df = outlier_handler.log_transform(df, 'DISTANCE')
for column in df.columns:
    if pd.api.types.is_numeric_dtype(df[column]):
        df = outlier_handler.handle_outliers(df, column, outlier_handler.Identifier.AUTO, missing_handler.Strategy.NONE)
missing_handler.print_nan_ratios(df)
print(df.isna().sum())

# Detected Outliers deleted. Now replacement starts...
# Categorical
df = missing_handler.replace_missing_values(df, 'OP_CARRIER_FL_NUM', missing_handler.Strategy.MEDIAN)  # 0.02%
df = missing_handler.replace_missing_values(df, 'ORIGIN_CITY_MARKET_ID', missing_handler.Strategy.MEDIAN)  # 0.47%
df = missing_handler.replace_missing_values(df, 'DEST_CITY_MARKET_ID', missing_handler.Strategy.MEDIAN)  # 0.47%

# Numeric
df = missing_handler.replace_missing_values(df, 'DEP_DELAY', missing_handler.Strategy.MEAN)  # 10.40%
df = missing_handler.replace_missing_values(df, 'TAXI_OUT', missing_handler.Strategy.MEAN)  # 7.24%
df = missing_handler.replace_missing_values(df, 'TAXI_IN', missing_handler.Strategy.MEAN)  # 8.49%
df = missing_handler.replace_missing_values(df, 'ARR_DELAY', missing_handler.Strategy.MEAN)  # 9.71%
df = missing_handler.replace_missing_values(df, 'ARR_DELAY_NEW', missing_handler.Strategy.MEAN)  # 10.97%
df = missing_handler.replace_missing_values(df, 'DISTANCE', missing_handler.Strategy.MEAN)  # 5.56%

# Rare
df = missing_handler.replace_missing_values(df, 'CANCELLED', missing_handler.Strategy.MODE)  # 3.79%
df = missing_handler.replace_missing_values(df, 'DIVERTED', missing_handler.Strategy.MODE)  # 0.28%
df = missing_handler.replace_missing_values(df, 'CARRIER_DELAY', missing_handler.Strategy.MODE)  # 23.11%
df = missing_handler.replace_missing_values(df, 'WEATHER_DELAY', missing_handler.Strategy.MODE)  # 23.11%
df = missing_handler.replace_missing_values(df, 'NAS_DELAY', missing_handler.Strategy.MODE)  # 23.11%
df = missing_handler.replace_missing_values(df, 'SECURITY_DELAY', missing_handler.Strategy.MODE)  # 23.11%
df = missing_handler.replace_missing_values(df, 'LATE_AIRCRAFT_DELAY', missing_handler.Strategy.MODE)  # 23.11%


for column in df.columns:
    if pd.api.types.is_numeric_dtype(df[column]):
        df = outlier_handler.handle_outliers(df, column, outlier_handler.Identifier.AUTO, missing_handler.Strategy.NONE)
missing_handler.print_nan_ratios(df)
print(df.isna().sum())

df = missing_handler.replace_missing_values(df, 'ORIGIN_CITY_MARKET_ID', missing_handler.Strategy.MEDIAN)
df = missing_handler.replace_missing_values(df, 'DEP_DELAY', missing_handler.Strategy.MEAN)
df = missing_handler.replace_missing_values(df, 'TAXI_OUT', missing_handler.Strategy.MEAN)
df = missing_handler.replace_missing_values(df, 'TAXI_IN', missing_handler.Strategy.MEAN)
df = missing_handler.replace_missing_values(df, 'DEP_DELAY', missing_handler.Strategy.MEAN)
df = missing_handler.replace_missing_values(df, 'ARR_DELAY', missing_handler.Strategy.MEAN)
df = missing_handler.replace_missing_values(df, 'ARR_DELAY_NEW', missing_handler.Strategy.MEAN)
df = missing_handler.replace_missing_values(df, 'DISTANCE', missing_handler.Strategy.MEAN)
# __________________________ OUTLIERS REMOVED __________________________

df.to_csv('flight_delays_processed.csv', index=False)
