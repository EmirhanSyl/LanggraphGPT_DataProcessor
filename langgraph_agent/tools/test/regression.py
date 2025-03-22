from langchain_core.tools import tool
import numpy as np
import scipy.stats as stats


@tool
def basic_linear_regression(independent_column: str, dependent_column: str) -> dict:
    """
    Call to perform a basic linear regression to model the relationship between an independent variable and a dependent
    variable. Only use column names from dataset summary to fill function parameters
    """

    return {
        "coefficients": {
            "intercept": 30000,
            "slope": 5000
        },
        "p_value": 0.01,
        "r_squared": 0.85,
        "conclusion": "The linear regression model shows a statistically significant relationship between the "
                      "independent variable and the dependent variable. The positive slope indicates that as the "
                      "independent variable increases, the dependent variable tends to increase. The R-squared value of"
                      " 0.85 suggests that a substantial portion of the variability in the dependent variable can be "
                      "explained by the independent variable."
    }


@tool
def multiple_linear_regression(independent_columns: list, dependent_column: str) -> dict:
    """
    Call to perform a multiple linear regression to model the relationship between several independent variables and a
    dependent variable. Only use column names from dataset summary to fill function parameters
    """
    return {
        "p_values": {
            "age": 0.02,
            "education_level": 0.001,
            "years_of_experience": 0.03
        },
        "r_squared": 0.90,
        "adjusted_r_squared": 0.88,
        "conclusion": "The multiple linear regression model indicates a statistically significant relationship between "
                      "the independent variables and the dependent variable. The R-squared value of 0.90 suggests that "
                      "the independent variables together explain 90% of the variance in the dependent variable. The "
                      "adjusted R-squared value of 0.88 accounts for the number of predictors, showing that the model "
                      "still explains a high proportion of the variance when adjusting for the number of variables."
    }


@tool
def hierarchical_regression(blocks: list, dependent_column: str) -> dict:
    """
    Perform a hierarchical regression where independent variables are entered in blocks to assess the incremental value of each block.
    Only use column names from dataset summary to fill function parameters
    """
    return {
        "step_1": {
            "coefficients": {
                "intercept": 25000,
                "age": 500
            },
            "p_values": {
                "age": 0.02
            },
            "r_squared": 0.30,
            "adjusted_r_squared": 0.28,
            "delta_r_squared": 0.30,
            "conclusion": "In the first step, age explains 30% of the variance in the dependent variable."
        },
        "step_2": {
            "coefficients": {
                "intercept": 20000,
                "age": 400,
                "education_level": 3500
            },
            "p_values": {
                "age": 0.04,
                "education_level": 0.01
            },
            "r_squared": 0.55,
            "adjusted_r_squared": 0.53,
            "delta_r_squared": 0.25,
            "conclusion": "In the second step, education level is added and improves the model by explaining an additional 25% of the variance."
        },
        "final_conclusion": "The hierarchical regression demonstrates that each step adds significant explanatory power to the model. Initially, age accounts for 30% of the variance, but the addition of education level and years of experience increases this to 75%, with years of experience contributing the most in the final step."
    }


@tool
def logistic_regression(independent_columns: list, dependent_column: str) -> dict:
    """
    Perform a logistic regression to model the probability of a binary outcome based on one or more independent variables.
    Only use column names from dataset summary to fill function parameters
    """
    return {
        "coefficients": {
            "intercept": -1.5,
            "age": 0.03,
            "education_level": 0.75,
            "years_of_experience": 1.2
        },
        "p_values": {
            "age": 0.01,
            "education_level": 0.005,
            "years_of_experience": 0.0001
        },
        "odds_ratios": {
            "age": 1.03,
            "education_level": 2.12,
            "years_of_experience": 3.32
        },
        "model_accuracy": 0.85,
        "conclusion": "The logistic regression model suggests that all independent variables significantly contribute "
                      "to the prediction of the binary outcome. The odds ratios indicate that with each additional year"
                      " of age, the odds of the outcome occurring increase by 3%, while higher education level and more"
                      " years of experience significantly increase the odds of the binary outcome. The model has an "
                      "accuracy of 85%, suggesting a good fit for predicting the binary outcome."
    }


regression_tests = [basic_linear_regression, multiple_linear_regression, hierarchical_regression, logistic_regression]
