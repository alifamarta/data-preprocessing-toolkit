import pandas as pd 
from sklearn.preprocessing import LabelEncoder

def label_encode(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    df = df.copy()
    le = LabelEncoder()

    for col in columns:
        df[col] = le.fit_transform(df[col])

    return df

def one_hot_encode(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    df = df.copy()
    return pd.get_dummies(df, columns=columns, drop_first=True)

