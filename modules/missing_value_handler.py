import pandas as pd
import numpy as np
from datetime import datetime
from typing import Union
from enum import Enum

from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.linear_model import LinearRegression
from sklearn.mixture import GaussianMixture
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors

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
        MICE = 9
        REGRESSION = 10
        INTERPOLATION = 11
        GMM_EM = 12
        HOT_DECK_IMPUTATION = 13
        NONE = 14
    
    
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
        knn_imputer = KNNImputer(n_neighbors=n_neighbors)
        df_imputed = knn_imputer.fit_transform(dataframe)
        df_imputed = pd.DataFrame(df_imputed, columns=dataframe.columns)
        return df_imputed

    def mice_imputation(self, dataframe: pd.DataFrame, max_iter: int = 10, random_state: int = 0):
        imputer = IterativeImputer(max_iter=max_iter, random_state=random_state)
        imputed_data = imputer.fit_transform(dataframe)
        imputed_data = pd.DataFrame(imputed_data, columns=dataframe.columns)
        return imputed_data

    def regression_imputation(self,  dataframe: pd.DataFrame, target_column: str, predictor_columns: list[str]):
        predictors = dataframe[predictor_columns]
        target = dataframe[target_column]

        x_train = predictors[target.notna()]
        y_train = target[target.notna()]
        x_test = predictors[target.isna()]

        # Train and Predict
        model = LinearRegression()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        # Fill in the missing values
        dataframe.loc[target.isna(), target_column] = y_pred
        return dataframe

    def interpolate_missings(self, dataframe: pd.DataFrame, column: Union[str, int], method: str = 'linear'):
        """
        Interpolates missing values in a specified column of a DataFrame.

        Parameters:
        ----------
        dataframe : pd.DataFrame
            The DataFrame containing the data to be interpolated.
        column : Union[str, int]
            The column in which to interpolate missing values.
        method : str, default 'linear'
            Interpolation technique to use. One of:

            'linear': Ignore the index and treat the values as equally spaced. This is the only method supported on MultiIndexes.
            'time': Works on daily and higher resolution data to interpolate given length of interval.
            'index', 'values': Use the actual numerical values of the index.
            'pad': Fill in NaNs using existing values.
            'nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'barycentric', 'polynomial': Passed to scipy.interpolate.interp1d, whereas ‘spline’ is passed to scipy.interpolate.UnivariateSpline. These methods use the numerical values of the index. Both ‘polynomial’ and ‘spline’ require that you also specify an order (int), e.g., df.interpolate(method='polynomial', order=5). Note that, slinear method in Pandas refers to the Scipy first order spline instead of Pandas first order spline.
            'krogh', 'piecewise_polynomial', 'spline', 'pchip', 'akima', 'cubicspline': Wrappers around the SciPy interpolation methods of similar names.
            'from_derivatives': Refers to scipy.interpolate.BPoly.from_derivatives.

        Returns:
        -------
        pd.DataFrame
            The DataFrame with missing values interpolated in the specified column.
        """
        return dataframe[[column]].interpolate(method=method)

    def expectation_maximization_with_gmm(self, dataframe: pd.DataFrame, n_components=3, max_iter=100):
        # Separate features and initialize the SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        data_imputed = imputer.fit_transform(dataframe)

        # Fit the Gaussian Mixture Model
        gmm = GaussianMixture(n_components=n_components, max_iter=max_iter, random_state=42)
        gmm.fit(data_imputed)

        # Predict the missing values using the fitted GMM
        missing_mask = np.isnan(dataframe)
        for feature in range(dataframe.shape[1]):
            if np.any(missing_mask[:, feature]):
                # Predict the component probabilities for each sample
                responsibilities = gmm.predict_proba(data_imputed)

                # Compute the expected value for the missing feature based on GMM components
                expected_values = np.dot(responsibilities, gmm.means_[:, feature])

                # Replace the missing values with the expected values
                data_imputed[missing_mask[:, feature], feature] = expected_values[missing_mask[:, feature]]

        return pd.DataFrame(data_imputed, columns=dataframe.columns)

    def hot_deck_imputation(self, dataframe: pd.DataFrame, n_neighbors=5):
        missing_indices = np.where(dataframe.isnull())
        imputed_data = dataframe.copy()

        # Use NearestNeighbors to find similar records
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(dataframe.dropna())

        for row, col in zip(missing_indices[0], missing_indices[1]):
            if dataframe.iloc[row].isnull().any():
                row_values = dataframe.iloc[row].values.reshape(1, -1)
                mask = np.isnan(row_values)

                # Find the nearest neighbors
                distances, indices = nbrs.kneighbors(row_values[~mask].reshape(1, -1))
                neighbors = dataframe.iloc[indices[0]].dropna()

                # Randomly select a donor from the neighbors
                donor = neighbors.sample(1)
                imputed_data.iat[row, col] = donor.iloc[0, col]

        return imputed_data


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



