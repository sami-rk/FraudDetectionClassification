import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from config import BOOL_COLS, NUM_COLS, CAT_COLS

global_medians = {}

def normalize_bool(val):
    """Normalize messy boolean values to 0/1."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip().lower()
    if s in ['1', 'true', 'yes', 'y']:
        return 1
    if s in ['0', 'false', 'no', 'n']:
        return 0
    return np.nan

def compute_global_medians(train):
    global global_medians

    train_temp = train.copy()
    for col in BOOL_COLS:
        train_temp[col] = train_temp[col].apply(normalize_bool)

    train_temp['transaction_frequency_24h'] = train_temp['transaction_frequency_24h'].astype(str).str.strip().str.lower()
    train_temp.loc[train_temp['transaction_frequency_24h'].str.contains('many', na=False), 'transaction_frequency_24h'] = '100'

    train_temp['account_age_days'] = train_temp['account_age_days'].astype(str).str.strip().str.lower()
    train_temp.loc[train_temp['account_age_days'].str.contains('new_user', na=False), 'account_age_days'] = '0'

    for col in NUM_COLS:
        train_temp[col] = pd.to_numeric(train_temp[col], errors='coerce')
    train_temp.loc[train_temp['failed_transaction_count_24h'] < 0, 'failed_transaction_count_24h'] = np.nan

    for col in NUM_COLS + BOOL_COLS:
        global_medians[col] = train_temp[col].median()

    return global_medians

def preprocess(df, is_train=True):
    df = df.copy()

    for col in BOOL_COLS:
        df[col] = df[col].apply(normalize_bool)

    df['transaction_timestamp'] = pd.to_datetime(df['transaction_timestamp'], errors='coerce')
    df['hour'] = df['transaction_timestamp'].dt.hour
    df['dayofweek'] = df['transaction_timestamp'].dt.dayofweek
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)

    if 'transaction_frequency_24h' in df.columns:
        df['transaction_frequency_24h'] = df['transaction_frequency_24h'].astype(str).str.strip().str.lower()
        df.loc[df['transaction_frequency_24h'].str.contains('many', na=False), 'transaction_frequency_24h'] = '100'

    if 'account_age_days' in df.columns:
        df['account_age_days'] = df['account_age_days'].astype(str).str.strip().str.lower()
        df.loc[df['account_age_days'].str.contains('new_user', na=False), 'account_age_days'] = '0'

    df['failed_transaction_count_24h'] = pd.to_numeric(df['failed_transaction_count_24h'], errors='coerce')
    df.loc[df['failed_transaction_count_24h'] < 0, 'failed_transaction_count_24h'] = np.nan

    for col in NUM_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    for col in NUM_COLS + BOOL_COLS:
        if col in df.columns:
            med = df[col].median() if is_train else global_medians.get(col, 0)
            df[col] = df[col].fillna(med)

    for col in CAT_COLS:
        df[col] = df[col].fillna('unknown')
        df[col] = df[col].astype(str).str.strip().str.lower()

    return df

def fit_encoders(train_p, test_p):
    encoders = {}
    for col in CAT_COLS:
        le = LabelEncoder()
        combined = pd.concat([train_p[col], test_p[col]], axis=0).astype(str)
        le.fit(combined)
        train_p[col] = le.transform(train_p[col].astype(str))
        test_p[col] = le.transform(test_p[col].astype(str))
        encoders[col] = le
    return encoders
