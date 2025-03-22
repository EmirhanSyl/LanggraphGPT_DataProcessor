from collections import defaultdict
from statistics import mode, StatisticsError

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import shapiro, normaltest, stats

import pandas as pd
from langchain.tools import tool
from langgraph.prebuilt import ToolExecutor, ToolNode

from . import hypothesis_tests_tool
from .test import parametric, regression, correlation, nonparametric
from .missing_handler_tool import MissingHandler
from .outlier_handler_tool import OutlierHandler

global dataset


class ToolEditor:
    def __init__(self) -> None:
        self.tools = self.get_tools()
        self.hypothesis_tools = self.get_hypothesis_tools()
        self.dataset_summarizer_tools = self.get_dataset_summarizer_tools()
        self.tool_executor = ToolExecutor(self.tools)
        self.tool_node = ToolNode(self.tools)

        self.regression_tests = regression.regression_tests
        self.correlation_tests = correlation.correlation_tests
        self.nonparametric_tests = nonparametric.nonparametric_tests
        self.parametric_tests = parametric.parametric_tests

    def get_tools(self) -> list:
        functions = [summarize_dataset, calculate_missing_values, handle_missing_values, replace_with_mean,
                     replace_with_mode, replace_with_median, handle_outliers]

        # functions += hypothesis_tests_tool.hypothesis_test_functions
        functions += regression.regression_tests
        functions += correlation.correlation_tests
        functions += nonparametric.nonparametric_tests
        functions += parametric.parametric_tests
        return functions

    def get_hypothesis_tools(self) -> list:
        functions = []
        functions += regression.regression_tests
        functions += correlation.correlation_tests
        functions += nonparametric.nonparametric_tests
        functions += parametric.parametric_tests
        return functions

    def get_dataset_summarizer_tools(self) -> list:
        return [summarize_dataset, calculate_missing_values, is_normal_distribution]

    def get_regression_tools(self):
        return regression.regression_tests

    def get_correlation_tools(self):
        return correlation.correlation_tests

    def get_parametric_tools(self):
        return parametric.parametric_tests

    def get_nonparametric_tools(self):
        return nonparametric.nonparametric_tests


def set_dataset(path):
    global dataset
    dataset = pd.read_csv(path)


def get_dataset_sample():
    return str(dataset.head().to_json(orient='records'))

def pdf():
    df = dataset

    # Set up a matplotlib figure
    fig, ax = plt.subplots(figsize=(48, 24))  # Adjust figure size based on your data

    # Hide axes
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_frame_on(False)  # No frame for the table

    # Create a table from the DataFrame
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')

    # Adjust the table style
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)

    # Save the figure as a PDF
    plt.savefig("output.pdf", bbox_inches="tight", dpi=300)

    plt.close()

@tool
def summarize_dataset() -> dict:
    """
    Summarizes the global 'dataset' by calculating various statistics for each column.

    Returns:
    --------
    dict
        A dictionary where each key is a column name from the DataFrame and each value
        is another dictionary containing the following keys:
        - 'type': The data type of the column.
        - 'min': The minimum value in the column.
        - 'max': The maximum value in the column.
        - 'mean': The mean value of the column (if numeric).
        - 'median': The median value of the column (if numeric).
        - 'mode': The mode of the column (if applicable; None if multimodal).
        - 'missing_count': The number of missing values in the column.
        - 'missing_ratio': The ratio of missing values in the column.
        - 'normality_test': The result of a normality test (Shapiro-Wilk).
        - 'outlier_count': The number of outliers in the column (using IQR for numeric columns).

    Notes:
    ------
    - The 'mean' and 'median' are calculated only for columns with numeric data types ('int64', 'float64').
    - The 'mode' is calculated by dropping NaN values from the column. If the column is multimodal
      or contains no valid values, the 'mode' will be set to None.
    - The normality test is applied to numeric columns only.
    - Outliers are detected using the IQR method for numeric columns.
    """
    summary = defaultdict(dict)

    for column in dataset.columns:
        col_data = dataset[column]

        # Data type
        summary[column]['type'] = str(col_data.dtype)

        # Missing values
        missing_count = int(col_data.isna().sum())
        missing_ratio = missing_count / len(dataset)
        summary[column]['missing_count'] = missing_count
        summary[column]['missing_ratio'] = missing_ratio

        summary[column]['min'] = col_data.min() if not pd.isna(col_data.min()) else None
        summary[column]['max'] = col_data.max() if not pd.isna(col_data.max()) else None

        try:
            summary[column]['mode'] = mode(col_data.dropna())
        except StatisticsError:
            summary[column]['mode'] = None

        if col_data.dtype in ['int64', 'float64']:
            # Numeric statistics
            summary[column]['mean'] = float(col_data.mean()) if not pd.isna(col_data.mean()) else None
            summary[column]['median'] = float(col_data.median()) if not pd.isna(col_data.median()) else None

            # To avoid Json serialization errors
            summary[column]['min'] = float(summary[column]['min'])
            summary[column]['max'] = float(summary[column]['max'])

            # Normality test (using Shapiro-Wilk test)
            try:
                if len(col_data.dropna()) >= 3:
                    stat, p_value = shapiro(col_data.dropna())
                    summary[column]['normality_test'] = {
                        "statistic": float(stat),
                        "p_value": float(p_value),
                        "is_normal": "True" if p_value > 0.05 else "False"
                    }
                else:
                    summary[column]['normality_test'] = None
            except ValueError:
                summary[column]['normality_test'] = None

            # Outlier detection using IQR method
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            outliers = col_data[(col_data < (Q1 - 1.5 * IQR)) | (col_data > (Q3 + 1.5 * IQR))]
            summary[column]['outlier_count'] = len(outliers)

        else:
            # For non-numeric columns, set None for numeric values
            summary[column]['min'] = None
            summary[column]['max'] = None
            summary[column]['mean'] = None
            summary[column]['median'] = None
            summary[column]['normality_test'] = None
            summary[column]['outlier_count'] = None

    return dict(summary)


def check_preprocess_needed():
    """
    Checks the dataset for outliers and missing values.
    If outliers or missing values are found, return 'preprocess', otherwise return 'skip'.

    Args:
        dataset (pd.DataFrame): The dataset to be checked.

    Returns:
        str: 'preprocess' if missing values or outliers are found, 'skip' otherwise.
    """

    # Check for missing values
    if dataset.isnull().sum().sum() > 0:
        return "preprocess"

    # Check for outliers in numeric columns
    numeric_cols = dataset.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        if is_outlier_present(dataset[col]):
            return "preprocess"

    return "skip"


def should_handle_outliers():
    # Check for outliers in numeric columns
    numeric_cols = dataset.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        if is_outlier_present(dataset[col]):
            return "handle"

    return "skip"


def is_outlier_present(series):
    """
    Checks if a numeric series contains any outliers using both Z-score and IQR methods.

    Args:
        series (pd.Series): The numeric column to be checked for outliers.

    Returns:
        bool: True if outliers are present, False otherwise.
    """
    if series.isnull().all():
        return False  # Skip if the column is completely NaN

    # Outlier detection using the Z-score (for normal distribution)
    if is_normal_distribution(series):
        z_scores = np.abs(stats.zscore(series.dropna()))
        if np.any(z_scores > 3):  # Z-score > 3 indicates outliers
            return True

    # Outlier detection using IQR (for non-normal distribution)
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    outliers = (series < (Q1 - 1.5 * IQR)) | (series > (Q3 + 1.5 * IQR))

    return outliers.any()

@tool
def is_normal_distribution(column):
    """
    Check if a numeric column follows a normal distribution using the Shapiro-Wilk test.

    Args:
        column (str): The column name to be checked.

    Returns:
        bool: True if the column is normally distributed, False otherwise.
    """
    series_clean = dataset[column].dropna()
    stat, p_value = shapiro(series_clean)
    return p_value > 0.05  # Normally distributed if p > 0.05


@tool
def calculate_missing_values() -> dict:
    """
    Calculate the missing value ratios and counts for each column in a DataFrame.

    Returns:
    - dict: A dictionary where the keys are column names and the values are dictionaries containing the
      count and ratio of missing values for each column.
    """
    missing_values_info = {}

    total_rows = len(dataset)

    for column in dataset.columns:
        missing_count = dataset[column].isna().sum()
        missing_ratio = missing_count / total_rows

        missing_values_info[column] = {
            'missing_count': int(missing_count),
            'missing_ratio': missing_ratio
        }

    return missing_values_info


@tool
def handle_missing_values() -> str:
    """
    Handle missing values, outliers, and perform type conversions on the global 'dataset' DataFrame.

    This function follows a detailed multistep process to manage missing values and outliers for different types
    of columns (numeric, string, and datetime). It modifies the global 'dataset' DataFrame in place by applying
    the following operations:

    Steps:
    1. **Type Conversion**:
       - For all columns, if the column is of 'Object' dtype, it is checked whether it can be converted
         to either 'String' or 'Datetime'. Conversion attempts:
         - Convert to 'datetime' using `pd.to_datetime()`. If the conversion fails (ValueError or TypeError),
           the column is converted to a 'String' type instead.

    2. **Missing Value Ratio Check**:
       - For each column in the DataFrame, the percentage of missing values is calculated.
       - If a column has more than 35% missing values, it is dropped from the DataFrame completely.

    3. **Handling Numeric Columns**:
       - For each numeric column in the dataset:
         - A normality test (Shapiro-Wilk) is conducted to determine if the column is normally distributed.
         - If the column is normally distributed, outliers are detected using the Z-score method (values with Z > 3
           are considered outliers).
         - If the column is not normally distributed, the IQR (Interquartile Range) method is used for outlier detection.
         - The ratio of outliers to total values is calculated:
           - If the outlier ratio is greater than 10%, missing values in the column are filled with the **median**.
           - Otherwise, missing values are filled with the **mean** of the column.

    4. **Handling String Columns**:
       - For each string column (dtype 'Object' or 'String'), missing values are filled with the constant value
         "unknown".

    5. **Handling Datetime Columns**:
       - For each datetime column:
         - Datetime values are converted to numeric timestamps for outlier detection.
         - Based on the normality of the column, outliers are detected using either the Z-score or IQR method.
         - The ratio of outliers to total values is calculated:
           - If the outlier ratio is greater than 15%, missing values in the column are filled with the **median**
             of the column.
           - If the outlier ratio is less than 15%, the column is checked to see if it represents a time series:
             - A time series is determined by checking if the data is sorted by time and has a regular frequency.
             - If the column is a time series, missing values are filled using **forward fill** (ffill) or
               **backward fill** (bfill).
             - If it is not a time series, missing values are filled with the **mean** of the column.

    Methods used:
    - `convert_column_types()`: Converts 'Object' columns to 'String' or 'Datetime' where necessary.
    - `check_and_remove_missing_columns()`: Removes columns with more than 35% missing values.
    - `handle_numeric_columns()`: Handles missing values and outliers in numeric columns.
    - `handle_string_columns()`: Fills missing values in string columns with 'unknown'.
    - `handle_datetime_columns()`: Detects outliers, handles time series, and fills missing values for datetime columns.
    - `is_normal_distribution()`: Tests if a column is normally distributed using the Shapiro-Wilk test.
    - `z_score_outliers()`: Detects outliers using Z-scores for normally distributed columns.
    - `iqr_outliers()`: Detects outliers using the IQR method for non-normal columns.
    - `is_time_series()`: Determines if a datetime column is a time series based on sorting and frequency.

    This method modifies the 'dataset' DataFrame directly.
    """
    return MissingHandler(dataset).handle_missing_value()


@tool
def handle_outliers() -> str:
    """
    Handle outliers in numeric columns of the dataset by applying appropriate transformations
    or replacing outliers with statistical measures (mean, median).

    This method performs the following steps:

    1. **Outlier Detection**:
       - Detects outliers in each numeric column of the dataset using either the Z-score method
         (for normally distributed data) or the Interquartile Range (IQR) method (for non-normally distributed data).
       - Outliers are values that fall outside of 1.5 times the IQR for skewed data or have a Z-score greater than 3
         for normally distributed data.

    2. **Outlier Handling**:
       - For each numeric column, the method checks the distribution and the proportion of outliers
         (outlier ratio) to decide how to handle them:
         - **Normally Distributed Data**:
           - If the column is normally distributed and the outlier ratio is low (< 10%), the method applies a
             **log transformation** (if all values are positive) to reduce the effect of outliers.
           - If the outlier ratio is high (≥ 10%) or log transformation is not applicable, outliers are replaced
             with the **mean** of the column.
         - **Non-Normally Distributed Data (Skewed)**:
           - If the column is not normally distributed and the outlier ratio is low (< 10%), the method applies a
             **square root transformation** (if all values are non-negative).
           - If the outlier ratio is high (≥ 10%) or square root transformation is not suitable, outliers are replaced
             with the **median** of the column.

    3. **Return Log**:
       - The method keeps a log of the actions taken for each numeric column, including whether a transformation was
         applied or outliers were replaced.
       - This log is returned as a dictionary, with the column names as keys and the action taken as the value.

    Returns:
        dict: A log of actions taken for each numeric column, indicating whether outliers were detected, transformed,
        or replaced, and what method was used.

    Example Log:
    {
        'A': 'Replaced outliers with mean',
        'B': 'No outliers detected',
        'C': 'Applied square root transformation'
    }
    """
    return OutlierHandler(dataset).handle_outliers()


@tool
def replace_with_mean(column_name: str) -> str:
    """
    Replace missing values in the specified column with the mean of the column.

    Parameters:
    - df: pd.DataFrame: The DataFrame containing the column.
    - column_name: str: The name of the column to process.

    Returns:
    - str: A message describing the operation performed, including the mean value used and the number of missing values filled.
    """
    if pd.api.types.is_numeric_dtype(dataset[column_name]):
        mean_value = dataset[column_name].mean()
        missing_count = dataset[column_name].isna().sum()
        dataset[column_name].fillna(mean_value, inplace=True)
        return f"Filled {missing_count} missing values in '{column_name}' with mean value {mean_value}."
    else:
        raise ValueError(f"Column '{column_name}' is not numeric.")


@tool
def replace_with_median(column_name: str) -> str:
    """
    Replace missing values in the specified column with the median of the column.

    Parameters:
    - df: pd.DataFrame: The DataFrame containing the column.
    - column_name: str: The name of the column to process.

    Returns:
    - str: A message describing the operation performed, including the median value used and the number of missing values filled.
    """
    if pd.api.types.is_numeric_dtype(dataset[column_name]):
        median_value = dataset[column_name].median()
        missing_count = dataset[column_name].isna().sum()
        dataset[column_name].fillna(median_value, inplace=True)
        return f"Filled {missing_count} missing values in '{column_name}' with median value {median_value}."
    else:
        raise ValueError(f"Column '{column_name}' is not numeric.")


@tool
def replace_with_mode(column_name: str) -> str:
    """
    Replace missing values in the specified column with the mode of the column.

    Parameters:
    - df: pd.DataFrame: The DataFrame containing the column.
    - column_name: str: The name of the column to process.

    Returns:
    - str: A message describing the operation performed, including the mode value used and the number of missing values filled.
    """
    mode_value = dataset[column_name].mode()[0]
    missing_count = dataset[column_name].isna().sum()
    dataset[column_name].fillna(mode_value, inplace=True)
    return f"Filled {missing_count} missing values in '{column_name}' with mode value '{mode_value}'."


def calculate_total_missing_values() -> int:
    total_missing = dataset.isna().sum().sum()
    return total_missing


def generate_tool_calls_for_missing_values() -> list:
    tool_calls = []

    for column in dataset.columns:
        if dataset[column].isna().sum() == 0:
            continue
        if pd.api.types.is_numeric_dtype(dataset[column]):
            tool_calls.append(f"Tool: replace_with_mean(), Args: column: '{column}'\n")
        elif pd.api.types.is_string_dtype(dataset[column]):
            tool_calls.append(f"Tool: replace_with_mode(), Args: column: '{column}'\n")
        elif pd.api.types.is_datetime64_any_dtype(dataset[column]):
            tool_calls.append(f"Tool: replace_with_median(), Args: column: '{column}'\n")

    return tool_calls
