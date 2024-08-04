from typing import Union
import pandas as pd
import numpy as np
from scipy.stats import zscore, kstest
from enum import Enum

from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest

from modules.helpers.validators import ColumnTypeValidators
from modules.missing_value_handler import MissingValueHandler

class OutlierHandler:
    class Identifier(Enum):
        IQR = 0
        ZSCORE = 1
        FREQUENCY = 2
        AUTO = 3
    
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

    def choose_outlier_method(self, dataframe: pd.DataFrame, column: Union[str, int]):
        # Check for normality using the Kolmogorov-Smirnov test
        stat, p_value = kstest(dataframe[column], 'norm', args=(dataframe[column].mean(), dataframe[column].std()))
        if p_value > 0.05:
            # Data is normally distributed
            return self.Identifier.ZSCORE
        else:
            # Data is not normally distributed
            return self.Identifier.IQR

    # -------------- ADVANCED OUTLIER DETECTION --------------
    def identify_outliers_isolation_forest(self, dataframe: pd.DataFrame, column: str, contamination: float = 0.1):
        model = IsolationForest(contamination=contamination)
        model.fit(dataframe[[column]])

        outliers = model.predict(dataframe[[column]])

        # Return indices of outliers
        outlier_indices = dataframe.index[outliers == -1].tolist()
        return outlier_indices

    def identify_outliers_elliptic_envolpe(self, dataframe: pd.DataFrame, column: str, contamination: float = 0.1):
        model = EllipticEnvelope(contamination=contamination)

        data = dataframe[[column]].values.reshape(-1, 1)
        model.fit(data)

        outliers = model.predict(data)

        # Return indices of outliers
        outlier_indices = dataframe.index[outliers == -1].tolist()
        return outlier_indices



    # -------------- HANDLE OUTLIERS --------------
    @ColumnTypeValidators.is_column_exists
    def handle_outliers(self, dataframe: pd.DataFrame, column: Union[str, int], identifier: Identifier = Identifier.IQR,
                        filling_strategy: MissingValueHandler.Strategy = 1, const=0):

        if identifier == self.Identifier.AUTO:
            identifier = self.choose_outlier_method(dataframe, column)
        if identifier == self.Identifier.IQR:
            outlier_indices = self.identify_outliers_iqr(dataframe, column)
        elif identifier == self.Identifier.FREQUENCY:
            outlier_indices = self.identify_outliers_frequency(dataframe, column)
        elif identifier == self.Identifier.ZSCORE:
            outlier_indices = self.identify_outliers_zscore(dataframe, column)
        else:
            raise ValueError("Invalid Identifier")

        print(f"Detected Outlier Values {'_'*60}")
        print((dataframe.loc[outlier_indices])[column].head())
        df_copy = dataframe.copy()
        df_copy.loc[outlier_indices, column] = np.nan

        missing_handler = MissingValueHandler()
        df_copy = missing_handler.replace_missing_values(df_copy, column, filling_strategy, const)

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