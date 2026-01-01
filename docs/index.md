# Data Preprocessing Toolkit

Python toolkit for data cleaning and preprocessing

## Features
- Data Inspection
- Missing value handling
- Outlier removal
- Encoding (Label Encoding & One-Hot Encoding)
- Feature scaling
- Train and test splitting 

## Quick Example 

```python
from data_preprocessing_toolkit.pipeline import cleaning_pipeline

X_train, X_test, y_train, y_test = cleaning_pipeline(
    df,
    target_column="label",
    encoding_columns=["gender", "city"],
    scaling_columns=["age", "income"]
)
```