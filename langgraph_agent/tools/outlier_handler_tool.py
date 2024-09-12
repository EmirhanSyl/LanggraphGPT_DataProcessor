import numpy as np
import pandas as pd
from scipy import stats


class OutlierHandler:
    def __init__(self, dataset):
        self.dataset = dataset

    def handle_outliers(self):
        """
        Handle outliers by applying appropriate transformations or replacing them.
        It uses square root, log transformations, or replaces outliers with mean, median, or mode depending on distribution.
        """
        log = {}

        numeric_cols = self.dataset.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            outliers = self._detect_outliers(self.dataset[col])
            outlier_ratio = outliers.sum() / len(self.dataset[col])

            if outlier_ratio == 0:
                log[col] = "No outliers detected"
                continue

            if self._is_normal_distribution(self.dataset[col]):
                # Normally distributed: consider Z-score or log transformation
                if outlier_ratio < 0.10:
                    # If outlier ratio is small, apply Z-score or log transformation
                    if (self.dataset[col] > 0).all():
                        self.dataset[col] = np.log1p(self.dataset[col])  # Log transform for positive data
                        log[col] = f"Applied log transformation"
                    else:
                        # Replace outliers with mean if not suitable for log
                        self.dataset[col].loc[outliers] = self.dataset[col].mean()
                        log[col] = f"Replaced outliers with mean"
                else:
                    # Replace with mean if there are too many outliers
                    self.dataset[col].loc[outliers] = self.dataset[col].mean()
                    log[col] = f"Replaced outliers with mean due to high outlier ratio"
            else:
                # Not normally distributed: use square root or median replacement
                if outlier_ratio < 0.10:
                    # Apply square root transformation for moderate skewness
                    if (self.dataset[col] >= 0).all():  # Square root works for non-negative data
                        self.dataset[col] = np.sqrt(self.dataset[col])
                        log[col] = f"Applied square root transformation"
                    else:
                        # Replace outliers with median if negative values are present
                        self.dataset.loc[outliers, col] = self.dataset[col].median()
                        log[col] = f"Replaced outliers with median"
                else:
                    # Replace outliers with median if outlier ratio is high
                    self.dataset.loc[outliers, col] = self.dataset[col].median()
                    log[col] = f"Replaced outliers with median due to high outlier ratio"

        return log

    def _detect_outliers(self, series):
        """
        Detects outliers in a numeric series using the IQR method or Z-score for normal distribution.
        Returns a boolean mask where True indicates an outlier.
        """
        if self._is_normal_distribution(series):
            z_scores = np.abs(stats.zscore(series.dropna()))
            return z_scores > 3  # Z-score > 3 indicates outliers
        else:
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            return (series < (Q1 - 1.5 * IQR)) | (series > (Q3 + 1.5 * IQR))

    def _is_normal_distribution(self, series):
        """
        Check if a numeric column follows a normal distribution using the Shapiro-Wilk test.
        Returns True if normally distributed, otherwise False.
        """
        series_clean = series.dropna()
        stat, p_value = stats.shapiro(series_clean)
        return p_value > 0.05  # Normally distributed if p > 0.05
