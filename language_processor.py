import pandas as pd
from typing import Union
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from contractions import contractions_dict
from modules.helpers.validators import ColumnTypeValidators


class LanguageProcessor:
    def __init__(self) -> None:
        pass

    @ColumnTypeValidators.is_column_exists
    @ColumnTypeValidators.string_required
    def remove_stopwords(self, dataframe: pd.DataFrame, column: Union[str, int], language: str = 'english'):
        try:
            stop_words = set(stopwords.words(language))
        except OSError:
            raise ValueError(f"Language '{language}' is not supported for stopword removal.")
        
        df_copy = dataframe.copy()
        df_copy[column] = df_copy[column].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in stop_words]))
        return df_copy

    @ColumnTypeValidators.is_column_exists
    @ColumnTypeValidators.string_required
    def expand_contractions(self, dataframe: pd.DataFrame, column: Union[str, int]):
        contraction_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))
        def expand_text(text):
            def replace(match):
                return contractions_dict[match.group(0)]
            return contraction_re.sub(replace, text)

        df_copy = dataframe.copy()
        df_copy[column] = df_copy[column].apply(lambda x: expand_text(x))
        return df_copy

    @ColumnTypeValidators.is_column_exists
    @ColumnTypeValidators.string_required
    def lemmatization(dataframe: pd.DataFrame, column: Union[str, int]):
        df_copy = dataframe.copy()
        lemmatizer = WordNetLemmatizer()
        df_copy[column] = df_copy[column].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))
        return df_copy
    
    @ColumnTypeValidators.is_column_exists
    @ColumnTypeValidators.string_required
    def stemming(self, dataframe: pd.DataFrame, column: Union[str, int]):
        df_copy = dataframe.copy()
        stemmer = PorterStemmer()
        df_copy[column] = df_copy[column].apply(lambda x: ' '.join([stemmer.stem(word) for word in x.split()]))
        return df_copy

    @ColumnTypeValidators.is_column_exists
    @ColumnTypeValidators.string_required
    def bag_of_words(self, dataframe: pd.DataFrame, column: Union[str, int]):
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(dataframe[column])
        bow_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
        return bow_df

    @ColumnTypeValidators.is_column_exists
    @ColumnTypeValidators.string_required
    def tf_idf(self, dataframe: pd.DataFrame, column: Union[str, int]):
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(dataframe[column])
        tfidf_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
        return tfidf_df

    @ColumnTypeValidators.is_column_exists
    @ColumnTypeValidators.string_required
    def n_grams(self, dataframe: pd.DataFrame, column: Union[str, int], n: int = 2):
        vectorizer = CountVectorizer(ngram_range=(n, n))
        X = vectorizer.fit_transform(dataframe[column])
        ngram_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
        return ngram_df 