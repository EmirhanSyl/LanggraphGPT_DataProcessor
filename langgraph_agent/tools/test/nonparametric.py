from langchain_core.tools import tool


@tool
def mann_whitney_u_test(column1: str, column2: str, alternative: str = 'two-sided') -> dict:
    """
    Perform a Mann-Whitney U test between two independent columns.

    Parameters:
    - column1 (str): Name of the first column.
    - column2 (str): Name of the second column.
    - alternative (str): Defines the alternative hypothesis ('two-sided', 'less', 'greater'). Default is 'two-sided'.

    Returns:
    - dict: Contains the U statistic, p-value, and a conclusion based on the hypothesis.
    """
    pass


@tool
def kruskal_wallis_test(group_column: str, value_column: str) -> dict:
    """
    Perform a Kruskal-Wallis H-test for independent samples across multiple groups.

    Parameters:
    - group_column (str): Name of the column that defines the groupings.
    - value_column (str): Name of the column with values to be compared across groups.

    Returns:
    - dict: Contains the H statistic, p-value, and a conclusion based on the hypothesis.
    """
    pass


@tool
def wilcoxon_signed_rank_test(column1: str, column2: str, alternative: str = 'two-sided') -> dict:
    """
    Perform a Wilcoxon signed-rank test for two paired samples.

    Parameters:
    - column1 (str): Name of the first paired column.
    - column2 (str): Name of the second paired column.
    - alternative (str): Defines the alternative hypothesis ('two-sided', 'less', 'greater'). Default is 'two-sided'.

    Returns:
    - dict: Contains the W statistic, p-value, and a conclusion based on the hypothesis.
    """
    pass


@tool
def friedman_test(group_column: str, value_columns: list[str]) -> dict:
    """
    Perform a Friedman test for repeated measures across multiple conditions.

    Parameters:
    - group_column (str): Name of the column defining the group or subject identifier.
    - value_columns (list): List of column names representing the repeated measures or conditions.

    Returns:
    - dict: Contains the Q statistic, p-value, and a conclusion based on the hypothesis.
    """
    pass


nonparametric_tests = [mann_whitney_u_test, kruskal_wallis_test, wilcoxon_signed_rank_test, friedman_test]
