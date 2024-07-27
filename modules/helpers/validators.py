import pandas as pd
import numpy as np
from functools import wraps

class ColumnTypeValidators:
    
    @staticmethod
    def check_column_existance(df, column):
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame.")
        
    @staticmethod
    def is_column_exists(func):
        @wraps(func)
        def wrapper(self, df, column, *args, **kwargs):
            ColumnTypeValidators.check_column_existance(df, column)
            return func(self, df, column, *args, **kwargs)
        return wrapper
    
    @staticmethod
    def string_required(func):
        @wraps(func)
        def wrapper(self, df, column, *args, **kwargs):
            ColumnTypeValidators.check_column_existance(df, column)
            if not pd.api.types.is_string_dtype(df[column]):
                raise ValueError(f"Column '{column}' must be of string type.")
            return func(self, df, column, *args, **kwargs)
        return wrapper

    @staticmethod
    def numeric_required(func):
        @wraps(func)
        def wrapper(self, df, column, *args, **kwargs):
            ColumnTypeValidators.check_column_existance(df, column)
            if not pd.api.types.is_numeric_dtype(df[column]):
                raise ValueError(f"Column '{column}' must be of numeric type.")
            return func(self, df, column, *args, **kwargs)
        return wrapper

    @staticmethod
    def datetime_required(func):
        @wraps(func)
        def wrapper(self, df, column, *args, **kwargs):
            ColumnTypeValidators.check_column_existance(df, column)
            if not pd.api.types.is_datetime64_any_dtype(df[column]):
                raise ValueError(f"Column '{column}' must be of datetime type.")
            return func(self, df, column, *args, **kwargs)
        return wrapper