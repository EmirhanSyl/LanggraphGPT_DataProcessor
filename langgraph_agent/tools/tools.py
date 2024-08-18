import pandas as pd
from langchain.tools import tool
from langgraph.prebuilt import ToolExecutor


class ToolEditor:
    def __init__(self) -> None:
        self.df = pd.read_csv(r"C:\Users\emirs\Documents\Projects\python\LanggraphGPT_DataProcessor\dataset\death_causes.csv")
        self.tools = self.get_tools()
        self.tool_executor = ToolExecutor(self.tools)

    @tool
    def get_dataset_summary(self, query: str):
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
        return self.df.describe().to_string()


    @tool
    def handle_missing_values(self, column_name: str) -> str:
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
        mean_value = self.df[column_name].mean()
        missing_count = self.df[column_name].isna().sum()
        self.df[column_name] = self.df[column_name].fillna(mean_value)
        return (f"The mean value for '{column_name}' is {mean_value:.2f}. "
                f"Replaced {missing_count} missing values with this mean.")

    def get_tools(self) -> list:
        return [self.get_dataset_summary, self.handle_missing_values]

