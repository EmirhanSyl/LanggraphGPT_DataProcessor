import pandas as pd
import numpy as np
from modules import data_type_converter, datetime_handler

# Load Data
df = pd.read_csv("flight_delays_processed.csv")

type_converter = data_type_converter.DataTypeConverter()

df['MKT_UNIQUE_CARRIER_ENCODED'] = type_converter.label_encoding(df, 'MKT_UNIQUE_CARRIER')['MKT_UNIQUE_CARRIER']
df['OP_UNIQUE_CARRIER_ENCODED'] = type_converter.label_encoding(df, 'OP_UNIQUE_CARRIER')['OP_UNIQUE_CARRIER']
df['TAIL_NUM_ENCODED'] = type_converter.label_encoding(df, 'TAIL_NUM')['TAIL_NUM']
df['DUP_ENCODED'] = type_converter.label_encoding(df, 'DUP')['DUP']

unique_cities = pd.concat([df['DEST_CITY_NAME'], df['ORIGIN_CITY_NAME']]).unique()
df['ORIGIN_CITY_NAME_ENCODED'] = type_converter.label_encoding(df, 'ORIGIN_CITY_NAME', fit=unique_cities)['ORIGIN_CITY_NAME']
df['DEST_CITY_NAME_ENCODED'] = type_converter.label_encoding(df, 'DEST_CITY_NAME', fit=unique_cities)['DEST_CITY_NAME']

unique_city_abrs = pd.concat([df['ORIGIN_STATE_ABR'], df['DEST_STATE_ABR']]).unique()
df['ORIGIN_STATE_ABR_ENCODED'] = type_converter.label_encoding(df, 'ORIGIN_STATE_ABR', fit=unique_city_abrs)['ORIGIN_STATE_ABR']
df['DEST_STATE_ABR_ENCODED'] = type_converter.label_encoding(df, 'DEST_STATE_ABR', fit=unique_city_abrs)['DEST_STATE_ABR']

print(df['ORIGIN_CITY_NAME_ENCODED'].head())
print(df['DEST_CITY_NAME_ENCODED'].head())

datetime_handler = datetime_handler.DatetimeHandler()
df = datetime_handler.convert_to_datetime(df, 'FL_DATE', format="%m/%d/%Y %I:%M:%S %p")
print(df.head())

df.drop(columns=["MKT_UNIQUE_CARRIER", "OP_UNIQUE_CARRIER", "TAIL_NUM", "DUP", "ORIGIN_CITY_NAME",
                 "DEST_CITY_NAME", "ORIGIN_STATE_ABR", "DEST_STATE_ABR"], inplace=True)

df.to_csv("flight_delays_only_numeric.csv", index=False)
