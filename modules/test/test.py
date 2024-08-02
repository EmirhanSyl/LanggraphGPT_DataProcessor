import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

# Sample data with missing values
data = {
    'A': [1, 2, np.nan, 4, 5],
    'B': [np.nan, 2, 3, np.nan, 5],
    'C': [1, np.nan, 3, 4, 5]
}

df = pd.DataFrame(data)
print("Original Data:")
print(df)

# Initialize the KNNImputer
knn_imputer = KNNImputer(n_neighbors=3)

# Fit and transform the dataset
df_imputed = knn_imputer.fit_transform(df)

# Convert the result back to a DataFrame
df_imputed = pd.DataFrame(df_imputed, columns=df.columns)

print("\nData After KNN Imputation:")
print(df_imputed)
