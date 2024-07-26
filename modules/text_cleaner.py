import pandas as pd
from typing import Union
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from contractions import contractions_dict

class TextCleaner:
    
    def __init__(self) -> None:
        pass
    
    def remove_repetitive_words(self, dataframe: pd.DataFrame, column: Union[str, int]) -> pd.DataFrame:
        if column not in dataframe.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame.")
        assert pd.api.types.is_string_dtype(dataframe[column]), f"Column '{column} is not string type'"

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
