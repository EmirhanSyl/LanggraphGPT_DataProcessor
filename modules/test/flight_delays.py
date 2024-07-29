import pandas as pd
import numpy as np
from modules import text_cleaner, missing_value_handler, language_processor, data_type_converter, outlier_handler, data_visualizer

# Load Data
df = pd.read_csv("../../dataset/flight_delays.csv")

# Get missing values ratio
missing_handler = missing_value_handler.MissingValueHandler()
missing_handler.print_nan_ratios(df)

# Tail num is like a licence plate of a plate. Can be filled with constant "UNKNOWN"
print(pd.api.types.is_string_dtype(df["TAIL_NUM"]))
df = missing_handler.replace_missing_values(df, 'TAIL_NUM', missing_handler.Strategy.CONSTANT, 'UNKNOWN')  # 1.08%

# Dep time of the plane. Performed median replacement
df = missing_handler.replace_missing_values(df, 'DEP_TIME', missing_handler.Strategy.MEDIAN)  # 3.67%

# Missing values on Delay causes column can be filled as categorical -1 and 0
df = missing_handler.replace_missing_values(df, 'CARRIER_DELAY', missing_handler.Strategy.CONSTANT, -1)  # 76.89%
df = missing_handler.replace_missing_values(df, 'WEATHER_DELAY', missing_handler.Strategy.CONSTANT, -1)  # 76.89%
df = missing_handler.replace_missing_values(df, 'NAS_DELAY', missing_handler.Strategy.CONSTANT, -1)  # 76.89%
df = missing_handler.replace_missing_values(df, 'SECURITY_DELAY', missing_handler.Strategy.CONSTANT, -1)  # 76.89%
df = missing_handler.replace_missing_values(df, 'LATE_AIRCRAFT_DELAY', missing_handler.Strategy.CONSTANT, -1)  # 76.89%

print(df['TAXI_OUT'])