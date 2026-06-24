import pandas as pd
import numpy as np

def add_features(df):
    df = df.copy()
    df['amount_vs_avg_ratio'] = df['transaction_amount'] / (df['avg_transaction_amount_7d'] + 1e-5)
    df['amount_deviation'] = df['transaction_amount'] - df['avg_transaction_amount_7d']
    df['risk_score'] = (df['unusual_amount_flag'].fillna(0) +
                        df['multiple_transactions_short_time'].fillna(0) +
                        df['high_risk_device_flag'].fillna(0) +
                        df['is_international'].fillna(0))
    df['high_freq_flag'] = (df['transaction_frequency_24h'] > df['transaction_frequency_24h'].quantile(0.9)).astype(int)
    df['failed_x_freq'] = df['failed_transaction_count_24h'] * df['transaction_frequency_24h']
    df['new_account'] = (df['account_age_days'] < 90).astype(int)
    return df
