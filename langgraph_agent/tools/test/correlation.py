from langchain_core.tools import tool


@tool
def correlation(column1: str, column2: str, method: str = 'pearson') -> dict:
    """
    Call to calculate the correlation between two variables. Only use column names from dataset summary to fill function parameters
    You can change 'method' parameter to set the type of correlation ('pearson', 'spearman', or 'kendall'). Default is 'pearson'.
    """
    return {
        "correlation_coefficient": 0.65,
        "p_value": 0.001,
        "conclusion": "The Pearson correlation between the two variables is 0.65, indicating a moderately strong "
                      "positive linear relationship. The p-value of 0.001 suggests that this correlation is "
                      "statistically significant, meaning the relationship is unlikely to have occurred by chance."
    }


@tool
def partial_correlation(target_column: str, control_columns: list, method: str = 'pearson') -> dict:
    """
    Call to calculate the partial correlation between the target variable and other variables.
    Only use column names from dataset summary to fill function parameters. You can change 'method' parameter to set
    the type of correlation ('pearson', 'spearman', or 'kendall'). Default is 'pearson'.
    """
    return {
        "partial_correlation_coefficient": 0.45,
        "p_value": 0.03,
        "conclusion": "The partial correlation between 'income' and the other variables, controlling for 'age' and "
                      "'education', is 0.45. This suggests a moderate positive relationship between income and the "
                      "other factors, even after accounting for the effects of age and education. The p-value of 0.03 "
                      "indicates that the partial correlation is statistically significant."
    }


@tool
def chi_square_test(column1: str, column2: str) -> dict:
    """
    Call to perform a chi-square test of independence to determine if there is a relationship between two categorical
    variables. Only use column names from dataset summary to fill function parameters
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
def reliability_analysis(items_columns: list) -> dict:
    """
    Call to perform a reliability analysis to assess the internal consistency of a set of items. Only use column names
    from dataset summary to fill function parameters
    """
    return {
        "cronbach_alpha": 0.82,
        "item_total_correlations": {
            "item1": 0.75,
            "item2": 0.72,
            "item3": 0.78,
            "item4": 0.69
        },
        "conclusion": "The Cronbach's Alpha value of 0.82 suggests good internal consistency for the items. The "
                      "item-total correlations also indicate that each item is sufficiently correlated with the overall"
                      " scale, supporting the reliability of the measure."
    }


@tool
def exploratory_factor_analysis(columns: list, n_factors: int, method: str = 'principal',
                                rotation: str = 'varimax') -> dict:
    """
    Call to perform exploratory factor analysis (EFA) to identify underlying relationships between measured variables.
    Only use column names from dataset summary to fill function parameters. You can set 'method' parameter to set extraction method
     ('principal' for Principal Axis Factoring, 'maximum_likelihood', etc.). Default is 'principal'.
    You can set 'rotation' parameter to set rotation method ('varimax', 'promax', etc.). Default is 'varimax'.
    """
    return {
        "factor_loadings": {
            "Factor1": {"question1": 0.75, "question2": 0.68, "question3": 0.72},
            "Factor2": {"question4": 0.80, "question5": 0.77}
        },
        "explained_variance": {
            "Factor1": 40.3,
            "Factor2": 35.2
        },
        "communalities": {
            "question1": 0.65,
            "question2": 0.58,
            "question3": 0.62,
            "question4": 0.78,
            "question5": 0.75
        },
        "conclusion": "The exploratory factor analysis identified two factors, with Factor 1 explaining 40.3% of the "
                      "variance and Factor 2 explaining 35.2%. The factor loadings suggest that 'question1', "
                      "'question2', and 'question3' are primarily related to Factor 1, while 'question4' and "
                      "'question5' are associated with Factor 2. This indicates two underlying structures in the data."
    }


correlation_tests = [correlation, partial_correlation, chi_square_test, reliability_analysis,
                     exploratory_factor_analysis]
