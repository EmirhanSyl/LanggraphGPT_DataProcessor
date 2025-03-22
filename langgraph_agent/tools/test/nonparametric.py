from langchain_core.tools import tool


@tool
def mann_whitney_u_test(column1: str, column2: str, alternative: str = 'two-sided') -> dict:
    """
    Call to perform a Mann-Whitney U test between two independent columns. Only use column names from dataset summary
    to fill function parameters. You can set 'alternative' paramater to set the alternative hypothesis
    ('two-sided', 'less', 'greater'). Default is 'two-sided'.
    """
    return {
        "U_statistic": 375.0,
        "p_value": 0.045,
        "conclusion": "Since the p-value is less than 0.05, we reject the null hypothesis. There is a statistically "
                      "significant difference between group_A and group_B based on the two-sided alternative hypothesis"
    }


@tool
def kruskal_wallis_test(group_column: str, value_column: str) -> dict:
    """
    Call to perform a Kruskal-Wallis H-test for independent samples across multiple groups. Only use column names from
    dataset summary to fill function parameters
    """
    return {
        "H_statistic": 7.85,
        "p_value": 0.02,
        "conclusion": "Since the p-value is less than 0.05, we reject the null hypothesis. There is a statistically "
                      "significant difference in recovery times across the treatment groups."
    }


@tool
def wilcoxon_signed_rank_test(column1: str, column2: str, alternative: str = 'two-sided') -> dict:
    """
    Call to perform a Wilcoxon signed-rank test for two paired samples. Only use column names from dataset summary to fill function parameters
    you have one optional parameter:
    - alternative (str): Defines the alternative hypothesis ('two-sided', 'less', 'greater'). Default is 'two-sided'.
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
    Call to perform a Friedman test for repeated measures across multiple conditions. Only use column names from
    dataset summary to fill function parameters
    """
    return {
        "Q": 16.6,
        "p-value": 0.001,
        "conclusion": "The null hypothesis can be rejected at a significance level of 0.05, indicating that there is a "
                      "significant increase in deaths over the years."
    }


nonparametric_tests = [mann_whitney_u_test, kruskal_wallis_test, wilcoxon_signed_rank_test, friedman_test]
