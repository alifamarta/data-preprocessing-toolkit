import pandas as pd
from sklearn.impute import SimpleImputer

def impute_missing_values(
        df: pd.DataFrame, 
        numeric_strategy: str = 'median', 
        categorical_strategy: str = 'most_frequent'
        ) -> pd.DataFrame:
    
    df = df.copy()
    
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns

    num_imputer = SimpleImputer(strategy=numeric_strategy)
    cat_imputer = SimpleImputer(strategy=categorical_strategy)
    
    df[num_cols] = num_imputer.fit_transform(df[num_cols])
    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

    return df