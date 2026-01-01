import pandas as pd 

def remove_outliers_iqr(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    df = df.copy()

    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr 
        upper = q3 + 1.5 * iqr
        
        df = df[(df[col] >= lower) & (df[col] <= upper)]

    return df