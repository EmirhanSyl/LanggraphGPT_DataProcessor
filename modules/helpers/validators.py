import pandas as pd
import numpy as np
from functools import wraps

class ColumnTypeValidators:
    @staticmethod
    def is_column_exists(func):
        @wraps(func)
        def wrapper(self, df, column, *args, **kwargs):
            if column not in dataframe.columns:
                raise ValueError(f"Column '{column}' not found in DataFrame.")
            return func(self, df, column, *args, **kwargs)
        return wrapper
    
    @staticmethod
    def string_required(func):
        @wraps(func)
        def wrapper(self, df, column, *args, **kwargs):
            # Check if the column is of string type
            if not pd.api.types.is_string_dtype(df[column]):
                raise ValueError(f"Column '{column}' must be of string type.")
            # Call the original function
            return func(self, df, column, *args, **kwargs)
        return wrapper

    @staticmethod
    def numeric_required(func):
        @wraps(func)
        def wrapper(self, df, column, *args, **kwargs):
            if not pd.api.types.is_numeric_dtype(df[column]):
                raise ValueError(f"Column '{column}' must be of numeric type.")
            return func(self, df, column, *args, **kwargs)
        return wrapper

    @staticmethod
    def datetime_required(func):
        @wraps(func)
        def wrapper(self, df, column, *args, **kwargs):
            if not pd.api.types.is_datetime64_any_dtype(df[column]):
                raise ValueError(f"Column '{column}' must be of datetime type.")
            return func(self, df, column, *args, **kwargs)
        return wrapper