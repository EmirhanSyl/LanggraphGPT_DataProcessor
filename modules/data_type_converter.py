from typing import Union
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import category_encoders as ce
from modules.helpers.validators import ColumnTypeValidators
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, PowerTransformer


class DataTypeConverter:
    def __init__(self) -> None:
        """"""
        pass

    # -------------- CATEGORICAL ENCODERS --------------
    @ColumnTypeValidators.string_required
    def label_encoding(self, dataframe: pd.DataFrame, column: Union[str, int]):
        """Convert categorical variables into numerical values"""

        df_copy = dataframe.copy()
        encoder = LabelEncoder()
        df_copy[column] = encoder.fit_transform(df_copy[column])
        return df_copy
    
    def one_hot_encoding(self, dataframe: pd.DataFrame, column: Union[str, int]):
        df_copy = dataframe.copy()
        one_hot_encoded = pd.get_dummies(df_copy[column], prefix=column)
        df_copy = df_copy.drop(column, axis=1)
        df_copy = df_copy.join(one_hot_encoded)
        return df_copy
    
    
    # -------------- SCALAR CONVERTIONS --------------
    @ColumnTypeValidators.numeric_required
    def standardize_data(self, dataframe: pd.DataFrame, column: Union[str, int]):
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(dataframe[column])
        return pd.DataFrame(scaled_data, columns=dataframe.columns)

    @ColumnTypeValidators.numeric_required
    def normalize_data(self, dataframe: pd.DataFrame, column: Union[str, int]):
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(dataframe[column])
        return pd.DataFrame(scaled_data, columns=dataframe.columns)

    @ColumnTypeValidators.numeric_required
    def normalize_vectors(self, dataframe: pd.DataFrame, column: Union[str, int]):
        norm = np.linalg.norm(dataframe[column], axis=1)
        normalized_data = dataframe[column].div(norm, axis=0)
        return pd.DataFrame(normalized_data, columns=dataframe.columns)

