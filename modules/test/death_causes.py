import pandas as pd
import numpy as np
from modules import text_cleaner, missing_value_handler, language_processor, data_type_converter

# Load Data
df = pd.read_csv("../../dataset/death_causes.csv")

# Get missing values ratio
missing_handler = missing_value_handler.MissingValueHandler()
missing_handler.print_nan_ratios(df)

# Missing Values analysed. No missing value found. Skipping replacement...


# Clean the texts on CAUSES column
text_cleaner = text_cleaner.TextCleaner()

# Remove punctuations
df = text_cleaner.replace_regex(df, 'CAUSE', text_cleaner.RegexPatterns.PUNCTUATION.value, '')
# Remove multiple blanks
df = text_cleaner.replace_regex(df, 'CAUSE', text_cleaner.RegexPatterns.MULTI_BLANK_CHARACTERS.value, '')
print(f"\nAFTER TEXT CLEANING PROCESS {'_'*60}\n")
print(df)


# Preprocess the texts on CAUSES column for NLP models
language_processor = language_processor.LanguageProcessor()
# Remove Stopwords
df = language_processor.remove_stopwords(df, 'CAUSE')

# Calculate TF-IDF
tf_idf_data = language_processor.tf_idf(df, 'CAUSE')
print(f"\nTF-IDF VECTORS {'_'*60}")
print(tf_idf_data.head())

# Perform BOW
bow_data = language_processor.bag_of_words(df, 'CAUSE')
print(f"\nBOW {'_'*60}")
print(bow_data.head())

# Perform N-GRAM
ngram_data = language_processor.n_grams(df, 'CAUSE', n=2)
print(f"\nN-GRAM {'_'*60}")
print(ngram_data.head())


# Use Scalar operations on numeric columns
data_type_converter = data_type_converter.DataTypeConverter()
# Standardize Data
# standard_data = data_type_converter.standardize_data(df, '2020')
# print(f"\nStandardized Data{'_'*60}")
# print(standard_data.head())

# Normalize Data
normalized_data = data_type_converter.normalize_data(df, '2020')
print(f"\nNormalized Data {'_'*60}")
print(normalized_data.head())

