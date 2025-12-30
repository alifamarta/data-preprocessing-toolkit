import pandas as pd

def data_overview(df: pd.DataFrame) -> dict:
    return {
        "shape": df.shape,
        "columns" : df.columns.to_list(),
        "dtypes": df.dtypes,
        "missing values": df.isnull().sum(),
        "duplicate_rows": df.duplicated().sum()
    }
