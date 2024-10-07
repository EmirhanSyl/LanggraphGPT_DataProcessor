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
    Call to perform a Wilcoxon signed-rank test for two paired samples. You don't have to know actual data

    Parameters:
    - column1 (str): Name of the first paired column.
    - column2 (str): Name of the second paired column.
    - alternative (str): Defines the alternative hypothesis ('two-sided', 'less', 'greater'). Default is 'two-sided'.

    Returns:
    - dict: Contains the W statistic, p-value, and a conclusion based on the hypothesis.
    """
    return {
        "W_statistic": 105.0,
        "p_value": 0.015,
        "conclusion": "Since the p-value is less than 0.05, we reject the null hypothesis. There is a statistically "
                      "significant difference between column_A and column_B under the two-sided alternative hypothesis."
    }


@tool
def friedman_test(group_column: str, value_columns: list[str]) -> dict:
    """
    Call to perform a Friedman test for repeated measures across multiple conditions. You don't have to know actual data

    Parameters:
    - group_column (str): Name of the column defining the group or subject identifier.
    - value_columns (list[str]): List of column names representing the repeated measures or conditions.

    Returns:
    - dict: Contains the Q statistic, p-value, and a conclusion based on the hypothesis.
    """
    return {
        "Q": 16.6,
        "p-value": 0.001,
        "conclusion": "The null hypothesis can be rejected at a significance level of 0.05, indicating that there is a "
                      "significant increase in deaths over the years."
    }


nonparametric_tests = [mann_whitney_u_test, kruskal_wallis_test, wilcoxon_signed_rank_test, friedman_test]
