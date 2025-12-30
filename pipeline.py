from missing_values import impute_missing_values
from outliers import remove_outliers_iqr 
from encoding import encode_features
from scaling import scale_features
from splitting import split_data

def cleaning_pipeline(df, target_column, outlier_columns=None, encoding_columns=None, encoding_method="onehot", scaling_columns=None, scaling_method='standard'):

    # missing values
    df = impute_missing_values(df)

    # outlier handling
    if outlier_columns:
        df = remove_outliers_iqr(df, outlier_columns)

    # encoding
    if encoding_columns:
        df = encode_features(
            df,
            columns=encoding_columns,
            method=encoding_method
        )

    if scaling_columns:
        df = scale_features(
            df,
            columns=scaling_columns,
            method=scaling_method
        )

    return split_data(df, target_column)