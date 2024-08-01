from enum import Enum
from typing import Union
import pandas as pd
from scipy import stats
from scipy.stats._morestats import AndersonResult

from modules.helpers.validators import ColumnTypeValidators


class NormalityTesting:
    """"""
    class NormalityTests(Enum):
        AUTO = 0
        SHAPIRO = 1
        KOLMOGOROV_SMIRNOW = 2
        K2 = 3

    def __init__(self):
        """"""
        pass

    @ColumnTypeValidators.numeric_required
    def shapiro_test(self, dataframe: pd.DataFrame, column: Union[str, int]):
        return stats.shapiro(dataframe[column])

    @ColumnTypeValidators.numeric_required
    def kolmogorov_smirnov_test(self, dataframe: pd.DataFrame, column: Union[str, int]):
        return stats.kstest(dataframe[column], cdf="norm")

    @ColumnTypeValidators.numeric_required
    def k2_test(self, dataframe: pd.DataFrame, column: Union[str, int]):
        return stats.normaltest(dataframe[column])

    @ColumnTypeValidators.numeric_required
    def anderson_test(self, dataframe: pd.DataFrame, column: Union[str, int]) -> AndersonResult:
        return stats.anderson(dataframe[column], dist='norm')

    def test_normality(self, dataframe: pd.DataFrame, column: Union[str, int], test: NormalityTests = NormalityTests.AUTO):
        if test == self.NormalityTests.AUTO:
            test = self.NormalityTests.SHAPIRO if (len(dataframe[column]) < 5000) else self.NormalityTests.KOLMOGOROV_SMIRNOW
        if test == self.NormalityTests.SHAPIRO:
            stat, p_value = self.shapiro_test(dataframe, column)
            return p_value > 0.05
        elif test == self.NormalityTests.KOLMOGOROV_SMIRNOW:
            stat, p_value = self.kolmogorov_smirnov_test(dataframe, column)
            return p_value > 0.05
        elif test == self.NormalityTests.K2:
            stat, p_value = self.k2_test(dataframe, column)
            return p_value > 0.05
        else:
            print("Unexpected Normality Test Type!")
