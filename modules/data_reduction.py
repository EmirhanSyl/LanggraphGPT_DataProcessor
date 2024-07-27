import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.preprocessing import StandardScaler

class DataReduction:
    def __init__(self) -> None:
        pass

    # ----------- Principal Component Analysis (PCA) -----------
    def apply_pca(self, dataframe: pd.DataFrame, n_components: int) -> pd.DataFrame:
        """Apply PCA to reduce dimensionality."""
        df_copy = dataframe.copy()
        features = df_copy.select_dtypes(include=[float, int]).columns
        x = df_copy.loc[:, features].values
        x = StandardScaler().fit_transform(x)
        
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(x)
        
        pca_df = pd.DataFrame(data=principal_components, columns=[f'principal_component_{i+1}' for i in range(n_components)])
        return pd.concat([df_copy.reset_index(drop=True), pca_df], axis=1)

    # ----------- Singular Value Decomposition (SVD) -----------
    def apply_svd(self, dataframe: pd.DataFrame, n_components: int) -> pd.DataFrame:
        """Apply SVD to reduce dimensionality."""
        df_copy = dataframe.copy()
        features = df_copy.select_dtypes(include=[float, int]).columns
        x = df_copy.loc[:, features].values
        x = StandardScaler().fit_transform(x)
        
        svd = TruncatedSVD(n_components=n_components)
        svd_components = svd.fit_transform(x)
        
        svd_df = pd.DataFrame(data=svd_components, columns=[f'svd_component_{i+1}' for i in range(n_components)])
        return pd.concat([df_copy.reset_index(drop=True), svd_df], axis=1)

    # ----------- Feature Selection: SelectKBest (Chi-Square) -----------
    def select_k_best_chi2(self, dataframe: pd.DataFrame, target_column: str, k: int) -> pd.DataFrame:
        """Select K best features based on Chi-Square."""
        df_copy = dataframe.copy()
        features = df_copy.select_dtypes(include=[float, int]).drop(columns=[target_column])
        target = df_copy[target_column]
        
        selector = SelectKBest(chi2, k=k)
        selector.fit(features, target)
        
        selected_features = features.columns[selector.get_support()]
        return df_copy[selected_features]

    # ----------- Feature Selection: SelectKBest (ANOVA F-value) -----------
    def select_k_best_anova(self, dataframe: pd.DataFrame, target_column: str, k: int) -> pd.DataFrame:
        """Select K best features based on ANOVA F-value."""
        df_copy = dataframe.copy()
        features = df_copy.select_dtypes(include=[float, int]).drop(columns=[target_column])
        target = df_copy[target_column]
        
        selector = SelectKBest(f_classif, k=k)
        selector.fit(features, target)
        
        selected_features = features.columns[selector.get_support()]
        return df_copy[selected_features]
