import pandas as pd
from sklearn.preprocessing import LabelEncoder

FEATURES = [
    'RowNumber', 'CustomerId', 'CreditScore',
    'Geography', 'Gender', 'Age', 'Tenure',
    'Balance', 'NumOfProducts', 'HasCrCard',
    'IsActiveMember', 'EstimatedSalary'
]

CATEGORICAL_COLS = ['Geography', 'Gender']


def load_data(path="data/churn.csv"):
    return pd.read_csv(path)


def clean_data(df):
    df = df.copy()
    df = df.ffill()
    return df


def encode_data(df, encoders=None, fit=True):
    df = df.copy()

    if fit:
        encoders = {}

        for col in CATEGORICAL_COLS:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le

        return df, encoders

    else:
        for col in CATEGORICAL_COLS:
            df[col] = encoders[col].transform(df[col])

        return df