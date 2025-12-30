import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def scale_features(df: pd.DataFrame, columns: list, method: str = "standard") -> pd.DataFrame:
    
    df = df.copy()

    if method == "standard":
        scaler = StandardScaler()

    elif method == "minmax":
        scaler = MinMaxScaler()

    else:
        raise ValueError("Scaling method must be 'standard' or 'minmax'")
    
    df[columns] = scaler.fit_transform(df[columns])
    return df