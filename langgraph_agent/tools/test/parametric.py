from langchain_core.tools import tool

@tool
def independent_t_test(column1: str, column2: str, equal_var: bool = True, alternative: str = 'two-sided') -> dict:
    """
    Perform an independent t-test between two independent columns (samples). Only use column names from dataset summary
    to fill function parameters. You can set 'alternative' paramater to set the alternative hypothesis
    ('two-sided', 'less', 'greater'). Default is 'two-sided'.
    """
    return {
        "chi_square_statistic": 12.34,
        "p_value": 0.005,
        "degrees_of_freedom": 1,
        "conclusion": "The chi-square test statistic is 12.34 with 1 degree of freedom. The p-value is 0.005, "
                      "indicating that there is a statistically significant association between 'gender' and "
                      "'purchased_product'. This suggests that the two categorical variables are not independent."
    }


@tool
def dependent_t_test(column1: str, column2: str, alternative: str = 'two-sided') -> dict:
    """
    Perform a dependent (paired) t-test between two related columns. Only use column names from dataset summary to
    fill function parameters. You can set 'alternative' paramater to set the alternative hypothesis
    ('two-sided', 'less', 'greater'). Default is 'two-sided'.
    """
    return {
        "chi_square_statistic": 12.34,
        "p_value": 0.005,
        "degrees_of_freedom": 1,
        "conclusion": "The chi-square test statistic is 12.34 with 1 degree of freedom. The p-value is 0.005, "
                      "indicating that there is a statistically significant association between 'gender' and "
                      "'purchased_product'. This suggests that the two categorical variables are not independent."
    }


@tool
def one_way_anova(group_column: str, value_column: str) -> dict:
    """
    Perform a one-way ANOVA to compare means across multiple groups. Only use column names from dataset summary to fill function parameters
    """
    return {
        "chi_square_statistic": 12.34,
        "p_value": 0.005,
        "degrees_of_freedom": 1,
        "conclusion": "The chi-square test statistic is 12.34 with 1 degree of freedom. The p-value is 0.005, "
                      "indicating that there is a statistically significant association between 'gender' and "
                      "'purchased_product'. This suggests that the two categorical variables are not independent."
    }


@tool
def two_way_anova(factor1_column: str, factor2_column: str, value_column: str) -> dict:
    """
    Perform a two-way ANOVA to assess the interaction between two factors on the values. You can set 'alternative' paramater to set the alternative hypothesis
    ('two-sided', 'less', 'greater'). Default is 'two-sided'.
    """
    return {
        "chi_square_statistic": 12.34,
        "p_value": 0.005,
        "degrees_of_freedom": 1,
        "conclusion": "The chi-square test statistic is 12.34 with 1 degree of freedom. The p-value is 0.005, "
                      "indicating that there is a statistically significant association between 'gender' and "
                      "'purchased_product'. This suggests that the two categorical variables are not independent."
    }


parametric_tests = [independent_t_test, dependent_t_test, one_way_anova, two_way_anova]