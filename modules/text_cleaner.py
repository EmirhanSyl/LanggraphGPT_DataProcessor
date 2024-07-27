import pandas as pd
from typing import Union
from nltk.corpus import stopwords
from modules.helpers.validators import ColumnTypeValidators
from enum import Enum

class TextCleaner:
    class RegexPatterns(Enum):
        MULTI_BLANK_CHARACTERS = r'\s{2,}'
        PUNCTUATION = r'[^\w\s]'
        EMAIL = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        URL = r'^(https?|ftp)://[^\s/$.?#].[^\s]*$'
        PHONE = r'^\+?1?\d{9,15}$'
        DATE_YYYY_MM_DD = r'^\d{4}-\d{2}-\d{2}$'
        DATE_DD_MM_YYYY = r'^\d{2}/\d{2}/\d{4}$'
        TIME_24H = r'^[0-2][0-3]:[0-5][0-9]$'
        CREDIT_CARD = r'^(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3(?:0[0-5]|[68][0-9])[0-9]{11}|6(?:011|5[0-9]{2})[0-9]{12}|(?:2131|1800|35\d{3})\d{11})$'
        IPV4_ADDRESS = r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$'
        IPV6_ADDRESS = r'^(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
        USERNAME = r'^[a-zA-Z0-9._-]{3,}$'
        HASHTAG = r'^#[a-zA-Z0-9_]+$'
        MENTION = r'^@[a-zA-Z0-9_]+$'


    def __init__(self) -> None:
        """"""
        pass
    
    @ColumnTypeValidators.string_required
    def remove_repetitive_words(self, dataframe: pd.DataFrame, column: Union[str, int]) -> pd.DataFrame:
        def remove_duplicates(text):
            if pd.isna(text):
                return text
            words = text.split()
            seen = set()
            unique_words = []
            for word in words:
                if word not in seen:
                    unique_words.append(word)
                    seen.add(word)
            return ' '.join(unique_words)

        df_copy = dataframe.copy()
        df_copy[column] = df_copy[column].apply(remove_duplicates)
        return df_copy

    @ColumnTypeValidators.string_required
    def replace_regex(self, dataframe: pd.DataFrame, column: Union[str, int], regex:Union[RegexPatterns, str] = RegexPatterns.PUNCTUATION, replacement:str=''):
        df_copy = dataframe.copy()
        df_copy[column] = df_copy[column].str.replace(regex, replacement, regex=True)
        return df_copy


    @ColumnTypeValidators.string_required
    def filter_words(self, dataframe: pd.DataFrame, column: Union[str, int], remove=['fword']):
        df_copy = dataframe.copy()
        remove_set = set(remove)
        df_copy[column] = df_copy[column].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in remove_set]))
        return df_copy
