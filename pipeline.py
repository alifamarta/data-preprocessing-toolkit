from .missing_values import impute_missing_values
from .outliers import remove_outliers_iqr 
from .encoding import encode_features
from .scaling import scale_features
from .splitting import split_data

def cleaning_pipeline(df, target_column, outlier_columns=None, encoding_columns=None, encoding_method="onehot", scaling_columns=None, scaling_method='standard'):

    try:
        steps = []

        # missing values
        df = impute_missing_values(df)
        steps.append('missing values handled')

        # outlier handling
        if outlier_columns:
            df = remove_outliers_iqr(df, outlier_columns)
            steps.append('outliers removed')

        # encoding
        if encoding_columns:
            df = encode_features(
                df,
                columns=encoding_columns,
                method=encoding_method
            )
            steps.append(f'{encoding_method} encoding applied')

        if scaling_columns:
            df = scale_features(
                df,
                columns=scaling_columns,
                method=scaling_method
            )
            steps.append(f'{scaling_method} scaling applied')

        # split
        X_train, X_test, y_train, y_test = split_data(df, target_column)

        # message
        print("Data has been cleaned.")
        print("steps executed:")
        for step in steps:
            print(f'- {step}')

        return X_train, X_test, y_train, y_test
    
    except Exception as e:
        raise RuntimeError(f"Data cleaning failed. Reason: {str(e)}")