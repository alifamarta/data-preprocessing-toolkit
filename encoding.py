import pandas as pd 
from sklearn.preprocessing import LabelEncoder

def encode_features(df: pd.DataFrame,columns: list,method: str = "onehot") -> pd.DataFrame:

    df = df.copy()

    if method == "label":
        le = LabelEncoder()
        for col in columns:
            df[col] = le.fit_transform(df[col])

    elif method == "onehot":
        df = pd.get_dummies(
            df,
            columns=columns,
            drop_first=True
        )

    else:
        raise ValueError("Encoding method must be 'label' or 'onehot'")
    
    return df