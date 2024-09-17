# One-sample t-test
from langchain_core.tools import tool


@tool
def one_sample_t_test(sample_data: list[float], population_mean: float, alpha: float = 0.05) -> dict:
    """
    Performs a one-sample t-test.
    Arguments:
    - sample_data: list of sample values
    - population_mean: population mean to test against
    - alpha: significance level (default 0.05)

    Returns:
    - dict with t-statistic, p-value, and conclusion (reject or fail to reject H₀)
    """
    pass


# Two-sample t-test (independent)
@tool
def two_sample_t_test(sample_data_1: list[float], sample_data_2: list[float], alpha: float = 0.05,
                      equal_var: bool = True) -> dict:
    """
    Performs an independent two-sample t-test.
    Arguments:
    - sample_data_1: list of values for group 1
    - sample_data_2: list of values for group 2
    - alpha: significance level (default 0.05)
    - equal_var: assume equal variance (default True)

    Returns:
    - dict with t-statistic, p-value, and conclusion
    """
    pass


# Paired t-test
@tool
def paired_t_test(sample_data_1: list[float], sample_data_2: list[float], alpha: float = 0.05) -> dict:
    """
    Performs a paired t-test.
    Arguments:
    - sample_data_1: list of values for condition 1
    - sample_data_2: list of values for condition 2
    - alpha: significance level (default 0.05)

    Returns:
    - dict with t-statistic, p-value, and conclusion
    """
    pass


# One-way ANOVA
@tool
def one_way_anova(*groups: list[list[float]], alpha: float = 0.05) -> dict:
    """
    Performs a one-way ANOVA.
    Arguments:
    - groups: lists of values for each group (2 or more groups)
    - alpha: significance level (default 0.05)

    Returns:
    - dict with F-statistic, p-value, and conclusion
    """
    pass


# Two-way ANOVA
@tool
def two_way_anova(data: list[list[float]], factors: tuple[list, list], alpha: float = 0.05) -> dict:
    """
    Performs a two-way ANOVA.
    Arguments:
    - data: 2D list of dependent variable values
    - factors: tuple of lists containing levels for the two factors
    - alpha: significance level (default 0.05)

    Returns:
    - dict with F-statistics, p-values for each factor and interaction, and conclusion
    """
    pass


# Chi-square goodness-of-fit
@tool
def chi_square_goodness_of_fit(observed: list[int], expected: list[int], alpha: float = 0.05) -> dict:
    """
    Performs a chi-square goodness-of-fit test.
    Arguments:
    - observed: observed frequencies
    - expected: expected frequencies
    - alpha: significance level (default 0.05)

    Returns:
    - dict with chi-square statistic, p-value, and conclusion
    """
    pass


# Chi-square test of independence
@tool
def chi_square_test_of_independence(observed: list[list[int]], alpha: float = 0.05) -> dict:
    """
    Performs a chi-square test of independence.
    Arguments:
    - observed: contingency table (2D list of frequencies)
    - alpha: significance level (default 0.05)

    Returns:
    - dict with chi-square statistic, p-value, and conclusion
    """
    pass


# Z-test for a single proportion
@tool
def z_test_single_proportion(sample_proportion: float, population_proportion: float, sample_size: int,
                             alpha: float = 0.05) -> dict:
    """
    Performs a z-test for a single proportion.
    Arguments:
    - sample_proportion: proportion from the sample
    - population_proportion: known population proportion
    - sample_size: size of the sample
    - alpha: significance level (default 0.05)

    Returns:
    - dict with z-statistic, p-value, and conclusion
    """
    pass


# Two-proportion z-test
@tool
def two_proportion_z_test(sample_proportion_1: float, sample_size_1: int, sample_proportion_2: float,
                          sample_size_2: int, alpha: float = 0.05) -> dict:
    """
    Performs a z-test for comparing two proportions.
    Arguments:
    - sample_proportion_1: proportion for group 1
    - sample_size_1: size of sample 1
    - sample_proportion_2: proportion for group 2
    - sample_size_2: size of sample 2
    - alpha: significance level (default 0.05)

    Returns:
    - dict with z-statistic, p-value, and conclusion
    """
    pass


# Mann-Whitney U test (independent samples)
@tool
def mann_whitney_u_test(sample_data_1: list[float], sample_data_2: list[float], alpha: float = 0.05) -> dict:
    """
    Performs a Mann-Whitney U test (non-parametric test for independent samples).
    Arguments:
    - sample_data_1: list of values for group 1
    - sample_data_2: list of values for group 2
    - alpha: significance level (default 0.05)

    Returns:
    - dict with U-statistic, p-value, and conclusion
    """
    pass


# Wilcoxon Signed-Rank Test (paired samples)
@tool
def wilcoxon_signed_rank_test(sample_data_1: list[float], sample_data_2: list[float], alpha: float = 0.05) -> dict:
    """
    Performs a Wilcoxon Signed-Rank Test (non-parametric test for paired samples).
    Arguments:
    - sample_data_1: list of values for condition 1
    - sample_data_2: list of values for condition 2
    - alpha: significance level (default 0.05)

    Returns:
    - dict with W-statistic, p-value, and conclusion
    """
    pass


# Kruskal-Wallis Test (multiple groups)
@tool
def kruskal_wallis_test(*groups: list[list[float]], alpha: float = 0.05) -> dict:
    """
    Performs a Kruskal-Wallis Test (non-parametric test for comparing multiple groups).
    Arguments:
    - groups: lists of values for each group (2 or more groups)
    - alpha: significance level (default 0.05)

    Returns:
    - dict with H-statistic, p-value, and conclusion
    """
    pass


# Linear Regression
@tool
def linear_regression(x: list[float], y: list[float]) -> dict:
    """
    Performs linear regression analysis.
    Arguments:
    - x: list of independent variable values
    - y: list of dependent variable values

    Returns:
    - dict with regression coefficients, R-squared, p-value, and conclusion
    """
    pass


# Logistic Regression
@tool
def logistic_regression(x: list[list[float]], y: list[int]) -> dict:
    """
    Performs logistic regression analysis.
    Arguments:
    - x: 2D list of independent variable values (multiple predictors)
    - y: list of binary dependent variable values

    Returns:
    - dict with regression coefficients, odds ratios, p-value, and conclusion
    """
    pass


# Shapiro-Wilk test for normality
@tool
def shapiro_wilk_test(data: list[float], alpha: float = 0.05) -> dict:
    """
    Performs the Shapiro-Wilk test for normality.
    Arguments:
    - data: list of sample values
    - alpha: significance level (default 0.05)

    Returns:
    - dict with W-statistic, p-value, and conclusion
    """
    pass


# Levelness's test for equal variances
@tool
def levene_test(*groups: list[list[float]], alpha: float = 0.05) -> dict:
    """
    Performs Levene's test for equal variances.
    Arguments:
    - groups: lists of values for each group (2 or more groups)
    - alpha: significance level (default 0.05)

    Returns:
    - dict with F-statistic, p-value, and conclusion
    """
    pass


hypothesis_test_functions = [
    # t-tests
    one_sample_t_test,
    two_sample_t_test,
    paired_t_test,

    # ANOVA tests
    one_way_anova,
    two_way_anova,

    # Chi-Square tests
    chi_square_goodness_of_fit,
    chi_square_test_of_independence,

    # Proportion tests
    z_test_single_proportion,
    two_proportion_z_test,

    # Non-parametric tests
    mann_whitney_u_test,
    wilcoxon_signed_rank_test,
    kruskal_wallis_test,

    # Regression analysis
    linear_regression,
    logistic_regression,

    # Assumption testing
    shapiro_wilk_test,
    levene_test
]