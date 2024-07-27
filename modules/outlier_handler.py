from typing import Union
import pandas as pd
import numpy as np
from scipy.stats import zscore
from enum import Enum 
from modules.helpers.validators import ColumnTypeValidators
from modules.missing_value_handler import MissingValueHandler

class OutlierHandler:
    class Identifier(Enum):
        IQR = 0
        ZSCORE = 1
        FREQUENCY = 2
    
    def __init__(self) -> None:
        """"""
        pass
    
    
    # -------------- DETECT OUTLIERS --------------
    @ColumnTypeValidators.numeric_required
    def identify_outliers_iqr(self, dataframe: pd.DataFrame, column: Union[str, int], threshold: float = 1.5) -> pd.DataFrame:
        Q1 = dataframe[column].quantile(0.25)
        Q3 = dataframe[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR

        outliers = dataframe[(dataframe[column] < lower_bound) | (dataframe[column] > upper_bound)]
        return outliers.index

    @ColumnTypeValidators.numeric_required
    def identify_outliers_zscore(self, dataframe: pd.DataFrame, column: Union[str, int], threshold=3):            
        z_scores = np.abs(zscore(dataframe[column]))
        outliers = dataframe[z_scores > threshold].index
        return outliers 
    
    @ColumnTypeValidators.numeric_required
    def identify_outliers_frequency(self, dataframe: pd.DataFrame, column: Union[str, int], threshold=0.05):
        value_counts = dataframe[column].value_counts(normalize=True)
        rare_values = value_counts[value_counts < threshold].index
        outlier_indices = dataframe[dataframe[column].isin(rare_values)].index

        return outlier_indices


    # -------------- HANDLE OUTLIERS --------------
    @ColumnTypeValidators.is_column_exists
    def handle_outliers(self, dataframe: pd.DataFrame, column: Union[str, int], identifier : Identifier = Identifier.IQR, 
                        method : str ='drop', filling_strategy : MissingValueHandler.Strategy = 1, const = 0):

        if identifier == self.Identifier.IQR:
            outlier_indices = self.identify_outliers_iqr(dataframe, column)
        elif identifier == self.Identifier.FREQUENCY:
            outlier_indices = self.identify_outliers_frequency(dataframe, column)
        elif identifier == self.Identifier.ZSCORE:
            outlier_indices = self.identify_outliers_zscore(dataframe, column)
        else:
            raise ValueError("Invalid Identifier")
        
        df_copy = dataframe.copy()

        if method == 'drop':
            df_copy.drop(outlier_indices, inplace=True)
            missing_handler = MissingValueHandler()
            df_copy = missing_handler.replace_missing_values(df_copy, filling_strategy, column, const)
        elif method == 'log':
            df_copy[column] = df_copy[column].apply(lambda x: np.log(x) if x > 0 else np.nan)
        elif method == 'sqrt':
            df_copy[column] = df_copy[column].apply(lambda x: np.sqrt(x) if x >= 0 else np.nan)

        return df_copy
    
    @ColumnTypeValidators.numeric_required
    def log_transform(self, dataframe: pd.DataFrame, column: Union[str, int]):
        df_copy = dataframe.copy()
        df_copy[column] = df_copy[column].apply(lambda x: np.log(x) if x > 0 else np.nan)
        return df_copy
    
    @ColumnTypeValidators.numeric_required
    def square_transform(self, dataframe: pd.DataFrame, column: Union[str, int]):
        df_copy = dataframe.copy()
        df_copy[column] = df_copy[column].apply(lambda x: np.sqrt(x) if x >= 0 else np.nan)
        return df_copy