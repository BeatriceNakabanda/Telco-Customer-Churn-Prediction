import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    return df

def fill_missing(df):
    df['Internet Type'] = df['Internet Type'].fillna('No Internet')
    df['Offer'] = df['Offer'].fillna('No Offer')
    return df

def encode_binary(df, binary_cols):
    for col in binary_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0})
    return df

def encode_multiclass(df, multi_cols):
    df = pd.get_dummies(df, columns=multi_cols, drop_first=True)
    return df

def encode_target(df):
    df.rename(columns={'Churn Label': 'Churn'}, inplace=True)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    return df

def drop_unnecessary(df):
    cols_to_drop = [
        'Customer Status', 'Churn Label', 'Churn Score',
        'Churn Category', 'Churn Reason', 'Customer ID',
        'Country','State','City','Zip Code','Latitude','Longitude','Quarter'
    ]
    df = df.drop(cols_to_drop, axis=1, errors='ignore')
    return df

def save_processed(df, path="data/processed/telco_cleaned.csv"):
    df.to_csv(path, index=False)