import pandas as pd
from config import TRAIN_DATA_PATH, TEST_DATA_PATH, FEATURE_COLS
from preprocessing import compute_global_medians, preprocess, fit_encoders
from feature_engineering import add_features
from training import cross_validate_and_evaluate
from prediction import train_final_and_predict

# ===== Load Data =====
train = pd.read_excel(TRAIN_DATA_PATH)
test  = pd.read_excel(TEST_DATA_PATH)

print(f"Train shape: {train.shape}, Test shape: {test.shape}")
print(f"Label distribution:\n{train['label'].value_counts()}")

# ===== Preprocessing =====
compute_global_medians(train)

train_p = preprocess(train, is_train=True)
test_p  = preprocess(test, is_train=False)

fit_encoders(train_p, test_p)

# ===== Feature Engineering =====
train_p = add_features(train_p)
test_p  = add_features(test_p)

print()
print(f"Features used: {len(FEATURE_COLS)}")

# Initialize columns for aggregation metrics
for col in ['cust_fraud_rate', 'cust_txn_count', 'merchant_fraud_rate', 'location_fraud_rate']:
    train_p[col] = 0.0
    test_p[col] = 0.0

y = train_p['label'].values
global_mean = y.mean()

# ===== Training & Evaluation =====
best_thresh = cross_validate_and_evaluate(train_p, y, global_mean)

# ===== Final Prediction =====
train_final_and_predict(train_p, test_p, best_thresh, global_mean)
