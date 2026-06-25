import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import datetime
from config import FEATURE_COLS

def train_final_and_predict(train_p, test_p, best_thresh, global_mean):
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

    final_model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=7,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    final_model.fit(X_full, train_p['label'].values)

    test_proba = final_model.predict_proba(X_test)[:, 1]
    test_labels = (test_proba > best_thresh).astype(int)

    submission = pd.DataFrame({'id': test_p['id'], 'label': test_labels})
    submission.to_csv(f'task1_classification_submission.csv', index=False)
    print()
    print(f"Submission saved. Shape: {submission.shape}")
    print(f"Predicted label distribution:")
    print(submission['label'].value_counts())
