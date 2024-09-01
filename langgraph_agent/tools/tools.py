from collections import defaultdict
from statistics import mode, StatisticsError

import pandas
import pandas as pd
from langchain.tools import tool
from langgraph.prebuilt import ToolExecutor, ToolNode
import matplotlib.pyplot as plt

class ToolEditor:
    def __init__(self) -> None:
        self.tools = self.get_tools()
        self.tool_executor = ToolExecutor(self.tools)
        self.tool_node = ToolNode(self.tools)

    def get_tools(self) -> list:
        return [summarize_dataset, handle_missing_values]


df = pd.read_csv(r"C:\Users\emirs\Documents\Projects\python\LanggraphGPT_DataProcessor\dataset\death_causes.csv")
dataset = None


def set_dataset(path):
    global dataset
    dataset = pd.read_csv(path)


@tool
def summarize_dataset() -> dict:
    """
        Summarizes a pandas DataFrame by calculating the data type, minimum, maximum, mean, median,
        and mode for each column in the dataset.

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

        Notes:
        ------
        - The 'mean' and 'median' are calculated only for columns with numeric data types ('int64', 'float64').
        - The 'mode' is calculated by dropping NaN values from the column. If the column is multimodal
          or contains no valid values, the 'mode' will be set to None.
        - This function is designed to handle columns with mixed data types, and it ensures that
          the appropriate statistical measures are calculated based on the column's data type.
        """
    summary = defaultdict(dict)

    for column in dataset.columns:
        col_data = dataset[column]
        summary[column]['type'] = str(col_data.dtype)
        summary[column]['min'] = col_data.min()
        summary[column]['max'] = col_data.max()
        summary[column]['mean'] = col_data.mean() if col_data.dtype in ['int64', 'float64'] else None
        summary[column]['median'] = col_data.median() if col_data.dtype in ['int64', 'float64'] else None

        try:
            summary[column]['mode'] = mode(col_data.dropna())
        except StatisticsError:
            summary[column]['mode'] = None

    return dict(summary)


@tool
def get_dataset_summary():
    """
    Generate a summary of the dataset.
    This function retrieves a dataset,
    computes descriptive statistics for all numerical columns in the dataset,
    and returns the summary as a formatted string.
    Parameters:
    ----------
    query : str
        The query should be a valid string that can be used to fetch the relevant subset of the dataset.
    Returns:
    -------
    str
        A string representation of the summary statistics for the dataset,
        including metrics such as count, mean, standard deviation, min, max,
        and quartiles for each numerical column.
    """
    return df.describe().to_string()


@tool
def handle_missing_values(column_name: str) -> str:
    """
    Replace missing values in the specified column with the column's mean value.
    This function calculates the mean of the specified column, replaces any missing
    values (NaNs) with this mean, and returns a summary string indicating the mean
    value and the number of missing values that were replaced.
    Parameters:
    ----------
    column_name : str
        The name of the column in which missing values will be replaced with the mean value.
    Returns:
    -------
    str
        A string summarizing the operation, including the mean value used for replacement
        and the number of missing values that were filled.
    """
    mean_value = df[column_name].mean()
    missing_count = df[column_name].isna().sum()
    df[column_name] = df[column_name].fillna(mean_value)
    return (f"The mean value for '{column_name}' is {mean_value:.2f}. "
            f"Replaced {missing_count} missing values with this mean.")
