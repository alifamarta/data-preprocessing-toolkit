# Basic Usage

```python
import pandas as pd
from data_preprocessing_toolkit.pipeline import cleaning_pipeline

df = pd.read_csv("data.csv")

X_train, X_test, y_train, y_test = cleaning_pipeline(
    df,
    target_column="label",
    outlier_columns=["income"],
    encoding_columns=["gender", "city"],
    encoding_method="onehot",
    scaling_columns=["age", "income"],
    scaling_method="standard"
)
```