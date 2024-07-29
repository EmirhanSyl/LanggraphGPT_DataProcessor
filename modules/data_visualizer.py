import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from typing import Union, List
from modules.helpers.validators import ColumnTypeValidators

class DataVisualizer:
    def __init__(self) -> None:
        sns.set(style="whitegrid")
        pass

    @ColumnTypeValidators.is_column_exists
    def plot_histogram(self, dataframe: pd.DataFrame, column: Union[str, int], bins: int = 10) -> None:
        """Plot a histogram for a specified column."""
        plt.figure(figsize=(10, 6))
        sns.histplot(dataframe.iloc[:, column], bins=bins, kde=True)
        plt.title(f'Histogram of {dataframe.columns[column]}')
        plt.xlabel(dataframe.columns[column])
        plt.ylabel('Frequency')
        plt.show()

    @ColumnTypeValidators.is_column_exists
    def plot_bar(self, dataframe: pd.DataFrame, column: Union[str, int], column2: Union[str, int]) -> None:
        """Plot a bar chart for a specified column."""
        plt.figure(figsize=(12, 8))

        sns.barplot(x=column, y=column2, data=dataframe.sort_values(column, ascending=False), palette='viridis')
        name1 = str(column)
        name2 = str(column2)
        plt.title(f'Bar Chart of {name1}, {name2}')
        plt.xlabel(column)
        plt.ylabel(column2)
        plt.show()

    @ColumnTypeValidators.is_column_exists
    def plot_box(self, dataframe: pd.DataFrame, column: Union[str, int]) -> None:
        """Plot a box plot for a specified column."""
        plt.figure(figsize=(10, 6))
        sns.boxplot(y=dataframe.iloc[:, column])
        plt.title(f'Box Plot of {dataframe.columns[column]}')
        plt.ylabel(dataframe.columns[column])
        plt.show()

    def plot_scatter(self, dataframe: pd.DataFrame, x_column: Union[str, int], y_column: Union[str, int]) -> None:
        """Plot a scatter plot between two specified columns."""
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=dataframe.iloc[:, x_column], y=dataframe.iloc[:, y_column])
        plt.title(f'Scatter Plot of {dataframe.columns[x_column]} vs {dataframe.columns[y_column]}')
        plt.xlabel(dataframe.columns[x_column])
        plt.ylabel(dataframe.columns[y_column])
        plt.show()

    def plot_heatmap(self, dataframe: pd.DataFrame) -> None:
        """Plot a heatmap of the correlation matrix."""
        plt.figure(figsize=(12, 8))
        correlation_matrix = dataframe.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Correlation Heatmap')
        plt.show()

    def plot_pairplot(self, dataframe: pd.DataFrame, columns: List[Union[str, int]] = None) -> None:
        """Plot a pair plot for a specified list of columns."""
        if columns:
            dataframe = dataframe.iloc[:, columns]
        sns.pairplot(dataframe)
        plt.show()


    def plot_pie(self, dataframe: pd.DataFrame, data_column: Union[str, int], label_column: Union[str, int], threshold:float=1) -> None:
        """Plot a pie chart for a specified column."""
        df_copy = dataframe.copy()
        total_deaths_sum = dataframe['Total'].sum()
        df_copy['Percentage'] = (dataframe['Total'] / total_deaths_sum) * 100

        others = df_copy[df_copy['Percentage'] < threshold].sum()
        df_copy = df_copy[df_copy['Percentage'] >= threshold]

        others_row = pd.DataFrame([['Other', others['Total'], others['Percentage']]],
                                  columns=['CAUSE', 'Total', 'Percentage'])
        df_copy = pd.concat([df_copy, others_row], ignore_index=True)

        plt.figure(figsize=(10, 10))
        plt.pie(df_copy.sort_values(data_column, ascending=False)[data_column], labels=df_copy[label_column], autopct='%1.1f%%', startangle=140,
                colors=sns.color_palette("viridis", len(df_copy[label_column])))
        plt.title(f'Proportion of {data_column} by {label_column}')
        plt.axis('equal')
        plt.show()

    def plot_line(self, dataframe: pd.DataFrame, x_column: Union[str, int], y_column: Union[str, int]) -> None:
        """Plot a line chart between two specified columns."""
        plt.figure(figsize=(10, 6))
        sns.lineplot(x=dataframe.iloc[:, dataframe.columns.get_loc(x_column)], y=dataframe.iloc[:, dataframe.columns.get_loc(x_column)])
        plt.title(f'Line Chart of {x_column} vs {y_column}')
        plt.xlabel(f"{x_column}")
        plt.ylabel(f"{y_column}")
        plt.show()

    def plot_interactive_scatter(self, dataframe: pd.DataFrame, x_column: Union[str, int], y_column: Union[str, int]) -> None:
        """Plot an interactive scatter plot between two specified columns."""
        fig = px.scatter(dataframe, x=dataframe.columns[x_column], y=dataframe.columns[y_column])
        fig.show()

    def plot_interactive_heatmap(self, dataframe: pd.DataFrame) -> None:
        """Plot an interactive heatmap of the correlation matrix."""
        correlation_matrix = dataframe.corr()
        fig = px.imshow(correlation_matrix, text_auto=True, aspect="auto", color_continuous_scale='coolwarm')
        fig.show()