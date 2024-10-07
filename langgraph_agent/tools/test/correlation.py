from langchain_core.tools import tool


@tool
def correlation(column1: str, column2: str, method: str = 'pearson') -> dict:
    """
    Call to calculate the correlation between two variables. You don't have to know actual data

    Parameters:
    - column1 (str): Name of the first variable.
    - column2 (str): Name of the second variable.
    - method (str): Type of correlation ('pearson', 'spearman', or 'kendall'). Default is 'pearson'.

    Returns:
    - dict: Contains the correlation coefficient, p-value, and a conclusion on the relationship between the variables.
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
    Call to calculate the partial correlation between the target variable and other variables, controlling for the effect of
    additional variables. You don't have to know actual data

    Parameters:
    - target_column (str): Name of the target variable.
    - control_columns (list): List of variable names to control for.
    - method (str): Type of correlation ('pearson', 'spearman', or 'kendall'). Default is 'pearson'.

    Returns:
    - dict: Contains the partial correlation coefficient, p-value, and a conclusion about the relationship after controlling for the control variables.
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
    variables. You don't have to know actual data

    Parameters:
    - column1 (str): Name of the first categorical variable.
    - column2 (str): Name of the second categorical variable.

    Returns:
    - dict: Contains the chi-square statistic, p-value, degrees of freedom, and a conclusion on the association between the variables.
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
    Call to perform a reliability analysis (typically using Cronbach's Alpha) to assess the internal consistency of a
    set of items. You don't have to know actual data

    Parameters:
    - items_columns (list): List of item/variable names to assess internal consistency (e.g., survey or test items).

    Returns:
    - dict: Contains Cronbach's Alpha value, item-total correlations, and a conclusion on the reliability of the scale.
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
    You don't have to know actual data

    Parameters:
    - columns (list): List of variable names to be included in the factor analysis.
    - n_factors (int): Number of factors to extract.
    - method (str): Extraction method ('principal' for Principal Axis Factoring, 'maximum_likelihood', etc.). Default is
     'principal'.
    - rotation (str): Rotation method ('varimax', 'promax', etc.). Default is 'varimax'.

    Returns:
    - dict: Contains factor loadings, explained variance, communalities, and a conclusion on the underlying structure of
     the data.
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
