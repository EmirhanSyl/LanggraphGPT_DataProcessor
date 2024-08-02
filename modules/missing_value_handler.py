import pandas as pd
import numpy as np
from datetime import datetime
from typing import Union
from enum import Enum

from sklearn.impute import KNNImputer

from modules.helpers.validators import ColumnTypeValidators


class MissingValueHandler:
    """Analyse And Handle Missing Values With Veraious Methods"""
    
    class Strategy(Enum):
        MODE = 0
        MEAN = 1
        MEDIAN = 2
        CONSTANT = 3
        REMOVE_ROW = 4
        REMOVE_COLUMN = 5
        FORWARD = 6
        BACKWARD = 7
        KNN_IMPUTE = 8
        NONE = 9
    
    
    def __init__(self) -> None:
        """"""
        pass
    
    
    # -------------- ANALYSE MISSING VALUES --------------
    def calculate_nan_ratios(self, df: pd.DataFrame, none_values: list) -> pd.Series:
        """Calculate none ratio of dataset directly, considering specific values as NaNs."""
        if none_values is not None:
            df = df.replace(none_values, np.nan)
        nan_counts = df.isna().sum()
        total_counts = len(df)
        nan_ratios = nan_counts / total_counts
        return nan_ratios 
    
    def print_nan_ratios(self, df: pd.DataFrame, none_values: list = None):
        def get_status(ratio):
            if ratio > 0.20:
                return 'Critical'
            elif 0.05 <= ratio <= 0.20:
                return 'Acceptable'
            else:
                return 'Good'

        overall_nan_ratio = self.calculate_nan_ratios(df, none_values)

        for column, ratio in overall_nan_ratio.items():
            status = get_status(ratio)
            dtype = df[column].dtype
            if pd.api.types.is_datetime64_any_dtype(df[column]):
                dtype = 'datetime'
            elif pd.api.types.is_string_dtype(df[column]):
                dtype = 'string'
            print(f"'{column}' \nnone value ratio: {ratio:.2%} | Data type: {dtype} | Status: {status}")


    # -------------- HANDLE MISSING VALUES --------------
    def replace_mode(self, dataframe: pd.DataFrame, column: Union[int, str]) -> pd.DataFrame:
        if dataframe[column].mode().empty:
            print(f"There is no mode value for column '{column}. Using Median replacement instead...'")
            self.replace_median(dataframe, column)
        df_copy = dataframe.copy()
        
        mode_value = dataframe[column].mode()[0]
        df_copy[column] = df_copy[column].fillna(mode_value)
        return df_copy
    
    
    @ColumnTypeValidators.numeric_required
    def replace_mean(self, dataframe: pd.DataFrame, column: Union[int, str]) -> pd.DataFrame:
        df_copy = dataframe.copy()
        mean_value = df_copy[column].mean()
        df_copy[column] = df_copy[column].fillna(mean_value)
        return df_copy

    @ColumnTypeValidators.numeric_required
    def replace_median(self, dataframe: pd.DataFrame, column: Union[int, str]) -> pd.DataFrame:
        df_copy = dataframe.copy()
        median_value = df_copy[column].median()
        df_copy[column] = df_copy[column].fillna(median_value)
        
        return df_copy
    
    def replace_constant(self, dataframe: pd.DataFrame, column: Union[int, str], const: Union[int, str, datetime]) -> pd.DataFrame:
       df_copy = dataframe.copy()
       if pd.api.types.is_numeric_dtype(df_copy[column]) and (isinstance(const, int) or isinstance(const, float)):
           const_value = const
       elif pd.api.types.is_string_dtype(df_copy[column]) and isinstance(const, str):
           const_value = const
       elif pd.api.types.is_datetime64_any_dtype(df_copy[column]) and isinstance(const, datetime):
           const_value = const
       else:
            const_value = const
            print(f"Unsupported const type for column '{column}'. This might cause unexpected results")
        
       df_copy[column] = df_copy[column].fillna(const_value)
       return df_copy


    def replace_remove_row(self, dataframe: pd.DataFrame, column: Union[int, str]) -> pd.DataFrame:
        df_copy = dataframe.copy()
        return df_copy.dropna(subset=[column])

    def replace_remove_column(self, dataframe: pd.DataFrame, column: Union[int, str]) -> pd.DataFrame:
        df_copy = dataframe.copy()
        return df_copy.drop(columns=[column])
    
    def replace_forward_backward(self, dataframe: pd.DataFrame, column: Union[int, str], method: str = "ffill") -> pd.DataFrame:
        df_copy = dataframe.copy()
        df_copy[column] = df_copy[column].fillna(method)
        return df_copy

    # ___________________ ADVANCED HANDLING ___________________
    def knn_imputation(self, dataframe: pd.DataFrame, n_neighbors=3):
        knn_imputer = KNNImputer(n_neighbors=3)
        df_imputed = knn_imputer.fit_transform(dataframe)
        df_imputed = pd.DataFrame(df_imputed, columns=dataframe.columns)
        return df_imputed

    @ColumnTypeValidators.is_column_exists
    def replace_missing_values(self, dataframe: pd.DataFrame, column: Union[int, str] = 0, strategy: Strategy = Strategy.MEAN, const : Union[int, str, datetime] = np.nan) -> pd.DataFrame:
        if strategy == self.Strategy.MODE:
            return self.replace_mode(dataframe, column)
        elif strategy == self.Strategy.MEAN:
            return self.replace_mean(dataframe, column)
        elif strategy == self.Strategy.MEDIAN:
            return self.replace_median(dataframe, column)
        elif strategy == self.Strategy.CONSTANT:
            return self.replace_constant(dataframe, column, const)
        elif strategy == self.Strategy.REMOVE_ROW:
            return self.replace_remove_row(dataframe, column)
        elif strategy == self.Strategy.REMOVE_COLUMN:
            return self.replace_remove_column(dataframe, column)
        elif strategy == self.Strategy.FORWARD:
            return self.replace_forward_backward(dataframe, column, "ffill")
        elif strategy == self.Strategy.BACKWARD:
            return self.replace_forward_backward(dataframe, column, "bfill")
        elif strategy == self.Strategy.NONE:
            return dataframe
        else:
            raise ValueError("Invalid strategy")



