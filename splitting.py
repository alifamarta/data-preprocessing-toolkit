from sklearn.model_selection import train_test_split

def split_data(df, label, test_size=0.2, random_state=42):
    x = df.drop(columns=[label])
    y = df[label]

    return train_test_split(x, y, test_size=test_size, random_state=random_state)