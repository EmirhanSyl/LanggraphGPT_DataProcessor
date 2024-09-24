from langchain_core.tools import tool

@tool
def independent_t_test(column1: str, column2: str, equal_var: bool = True, alternative: str = 'two-sided') -> dict:
    """
    Perform an independent t-test between two independent columns (samples).

    Parameters:
    - column1 (str): Name of the first column (sample 1).
    - column2 (str): Name of the second column (sample 2).
    - equal_var (bool): Assumes equal variance if True, otherwise performs Welch’s t-test. Default is True.
    - alternative (str): Defines the alternative hypothesis ('two-sided', 'less', 'greater'). Default is 'two-sided'.

    Returns:
    - dict: Contains the t statistic, p-value, and a conclusion based on the hypothesis.
    """
    pass


@tool
def dependent_t_test(column1: str, column2: str, alternative: str = 'two-sided') -> dict:
    """
    Perform a dependent (paired) t-test between two related columns.

    Parameters:
    - column1 (str): Name of the first column (paired sample 1).
    - column2 (str): Name of the second column (paired sample 2).
    - alternative (str): Defines the alternative hypothesis ('two-sided', 'less', 'greater'). Default is 'two-sided'.

    Returns:
    - dict: Contains the t statistic, p-value, and a conclusion based on the hypothesis.
    """
    pass


@tool
def one_way_anova(group_column: str, value_column: str) -> dict:
    """
    Perform a one-way ANOVA to compare means across multiple groups.

    Parameters:
    - group_column (str): Name of the column that defines the groupings.
    - value_column (str): Name of the column with values to be compared across groups.

    Returns:
    - dict: Contains the F statistic, p-value, and a conclusion based on the hypothesis.
    """
    pass


@tool
def two_way_anova(factor1_column: str, factor2_column: str, value_column: str) -> dict:
    """
    Perform a two-way ANOVA to assess the interaction between two factors on the values.

    Parameters:
    - factor1_column (str): Name of the first factor column.
    - factor2_column (str): Name of the second factor column.
    - value_column (str): Name of the column with values to be compared across the factors.

    Returns:
    - dict: Contains the F statistic for the main effects and interaction, p-values, and a conclusion based on the hypothesis.
    """
    pass


parametric_tests = [independent_t_test, dependent_t_test, one_way_anova, two_way_anova]