import pandas as pd
import numpy as np
from scipy import stats
import json


class MissingHandler:
    def __init__(self, dataset):
        """
        Initialize the MissingHandler with a dataset.
        Args:
            dataset (pd.DataFrame): The dataset to be handled.
        """
        self.dataset = dataset
        self.change_log = {}  # Log of changes applied to the dataset

    def handle_missing_value(self):
        """
        Handle missing values, outliers, and perform type conversions on the dataset.
        Returns a summary of changes in JSON format.
        """
        # Step 1: Convert object types to appropriate types
        self.change_log.update(self._convert_column_types())

        # Step 2: Check missing value ratio and remove columns with more than 35% missing values
        self.change_log.update(self._check_and_remove_missing_columns())

        # Step 3: Handle numeric columns
        self.change_log.update(self._handle_numeric_columns())

        # Step 4: Handle string columns
        self.change_log.update(self._handle_string_columns())

        # Step 5: Handle datetime columns
        self.change_log.update(self._handle_datetime_columns())

        return json.dumps(self.change_log, indent=4)

    def _convert_column_types(self):
        """
        Convert 'Object' types to 'String' or 'Datetime' if necessary.
        Returns a log of changes.
        """
        log = {}
        for col in self.dataset.columns:
            if self.dataset[col].dtype == 'O':  # Object type
                try:
                    self.dataset[col] = pd.to_datetime(self.dataset[col])
                    log[col] = f"Converted column '{col}' from Object to Datetime"
                except (ValueError, TypeError):
                    self.dataset[col] = self.dataset[col].astype(str)
                    log[col] = f"Converted column '{col}' from Object to String"
        return log

    def _check_and_remove_missing_columns(self):
        """
        Check missing value ratio and remove columns with more than 35% missing values.
        Returns a log of removed columns.
        """
        log = {}
        missing_ratio = self.dataset.isnull().mean()
        columns_to_drop = missing_ratio[missing_ratio > 0.35].index

        for col in columns_to_drop:
            log[col] = f"Removed column '{col}' due to missing ratio {missing_ratio[col]:.2%}"

        self.dataset.drop(columns=columns_to_drop, inplace=True)
        return log

    def _handle_numeric_columns(self):
        """
        Handle missing values and outliers in numeric columns.
        Returns a log of changes for each numeric column.
        """
        log = {}
        numeric_cols = self.dataset.select_dtypes(include=np.number).columns

        for col in numeric_cols:
            if self.dataset[col].isnull().mean() > 0:
                outlier_info = {}

                # Outlier detection
                if self._is_normal_distribution(self.dataset[col]):
                    outlier_mask = self._z_score_outliers(self.dataset[col])
                    outlier_info['method'] = 'Z-score'
                else:
                    outlier_mask = self._iqr_outliers(self.dataset[col])
                    outlier_info['method'] = 'IQR'

                outlier_ratio = outlier_mask.sum() / len(self.dataset[col])
                outlier_info['outlier_ratio'] = outlier_ratio

                # Handling missing values based on outlier ratio
                if outlier_ratio > 0.10:
                    self.dataset[col].fillna(self.dataset[col].median(), inplace=True)
                    log[col] = f"Filled missing values in '{col}' with median (Outlier ratio {outlier_ratio:.2%})"
                else:
                    self.dataset[col].fillna(self.dataset[col].mean(), inplace=True)
                    log[col] = f"Filled missing values in '{col}' with mean (Outlier ratio {outlier_ratio:.2%})"

        return log

    def _handle_string_columns(self):
        """
        Handle missing values in string columns.
        Returns a log of changes for each string column.
        """
        log = {}
        string_cols = self.dataset.select_dtypes(include='object').columns
        for col in string_cols:
            if self.dataset[col].isnull().mean() > 0:
                self.dataset[col].fillna('unknown', inplace=True)
                log[col] = f"Filled missing values in '{col}' with 'unknown'"
        return log

    def _handle_datetime_columns(self):
        """
        Handle missing values and outliers in datetime columns.
        Returns a log of changes for each datetime column.
        """
        log = {}
        datetime_cols = self.dataset.select_dtypes(include='datetime64').columns

        for col in datetime_cols:
            if self.dataset[col].isnull().mean() > 0:
                numeric_col = self.dataset[col].apply(lambda x: x.timestamp() if not pd.isnull(x) else np.nan)

                # Outlier detection
                if self._is_normal_distribution(numeric_col):
                    outlier_mask = self._z_score_outliers(numeric_col)
                    outlier_method = 'Z-score'
                else:
                    outlier_mask = self._iqr_outliers(numeric_col)
                    outlier_method = 'IQR'

                outlier_ratio = outlier_mask.sum() / len(numeric_col)

                if self._is_time_series(col):
                    self.dataset[col].fillna(method='ffill', inplace=True)
                    self.dataset[col].fillna(method='bfill', inplace=True)
                    log[col] = f"Filled missing values in '{col}' using forward/backward fill (Time Series)"
                else:
                    if outlier_ratio > 0.15:
                        self.dataset[col].fillna(self.dataset[col].median(), inplace=True)
                        log[
                            col] = f"Filled missing values in '{col}' with median (Outlier ratio {outlier_ratio:.2%}, method: {outlier_method})"
                    else:
                        self.dataset[col].fillna(self.dataset[col].mean(), inplace=True)
                        log[
                            col] = f"Filled missing values in '{col}' with mean (Outlier ratio {outlier_ratio:.2%}, method: {outlier_method})"

        return log

    def _is_normal_distribution(self, series):
        """
        Check if a numeric column follows a normal distribution using the Shapiro-Wilk test.
        Returns True if normally distributed, otherwise False.
        """
        series_clean = series.dropna()
        stat, p_value = stats.shapiro(series_clean)
        return p_value > 0.05  # Normally distributed if p > 0.05

    def _z_score_outliers(self, series):
        """
        Detect outliers using the Z-score method for normally distributed columns.
        Returns a boolean mask where True indicates an outlier.
        """
        z_scores = np.abs(stats.zscore(series.dropna()))
        return z_scores > 3  # Z-score > 3 indicates an outlier

    def _iqr_outliers(self, series):
        """
        Detect outliers using the IQR method for non-normally distributed columns.
        Returns a boolean mask where True indicates an outlier.
        """
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        return ((series < (Q1 - 1.5 * IQR)) | (series > (Q3 + 1.5 * IQR)))

    def _is_time_series(self, col):
        """
        Determine if a datetime column is a time series based on sorting and regular frequency.
        Returns True if the column is a time series, otherwise False.
        """
        is_sorted = self.dataset[col].is_monotonic
        frequency = pd.infer_freq(self.dataset[col].dropna())
        return is_sorted and frequency is not None
