import pandas as pd
import numpy as np
from datetime import datetime
from config import FEATURE_COLS
from models.registry import get_model


def train_final_and_predict(train_p, test_p, best_thresh, global_mean, model_name='lightgbm', model_params=None):
    cust_stats_full = train_p.groupby('customer_id')['label'].agg(['count', 'mean'])
    merch_stats_full = train_p.groupby('merchant_category')['label'].mean()
    loc_stats_full = train_p.groupby('location')['label'].mean()

    X_full_df = train_p[FEATURE_COLS].copy()
    X_test_df = test_p[FEATURE_COLS].copy()

    X_full_df['cust_txn_count'] = train_p['customer_id'].map(cust_stats_full['count']).fillna(0.0)
    X_full_df['cust_fraud_rate'] = train_p['customer_id'].map(cust_stats_full['mean']).fillna(global_mean)
    X_full_df['merchant_fraud_rate'] = train_p['merchant_category'].map(merch_stats_full).fillna(global_mean)
    X_full_df['location_fraud_rate'] = train_p['location'].map(loc_stats_full).fillna(global_mean)

    X_test_df['cust_txn_count'] = test_p['customer_id'].map(cust_stats_full['count']).fillna(0.0)
    X_test_df['cust_fraud_rate'] = test_p['customer_id'].map(cust_stats_full['mean']).fillna(global_mean)
    X_test_df['merchant_fraud_rate'] = test_p['merchant_category'].map(merch_stats_full).fillna(global_mean)
    X_test_df['location_fraud_rate'] = test_p['location'].map(loc_stats_full).fillna(global_mean)

    X_full = X_full_df.values
    X_test = X_test_df.values

    model = get_model(model_name, model_params)
    model.fit(X_full, train_p['label'].values)

    test_proba = model.predict_proba(X_test)
    test_labels = (test_proba > best_thresh).astype(int)

    submission = pd.DataFrame({'id': test_p['id'], 'label': test_labels})
    filename = f'task1_classification_submission_{model_name}.csv'
    submission.to_csv(filename, index=False)
    print()
    print(f"Submission saved: {filename} (Shape: {submission.shape})")
    print(f"Predicted label distribution:")
    print(submission['label'].value_counts())

    return submission
