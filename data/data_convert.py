import numpy as np
import pandas as pd

def sklearn2df(sklearn_dataset):
    df = pd.DataFrame(sklearn_dataset.data, columns=sklearn_dataset.feature_names)
    df['target'] = pd.Series(sklearn_dataset.target)
    return df


def df2ndarray(df):
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    #converting text labels to numeric form
    code, unique = pd.factorize(y)
    return X, code

def train_test_split(X, y, test_size = 0.40):
    test_size = int(len(X) * test_size)
    idxs = np.random.choice(len(X), size = test_size)
    mask = np.ones(len(X), dtype = bool)
    mask[idxs] = False

    return X[mask], X[idxs], y[mask], y[idxs]