import warnings
warnings.filterwarnings('ignore')

TRAIN_DATA_PATH = 'datasets/train.xlsx'
TEST_DATA_PATH = 'datasets/student_test.xlsx'

BOOL_COLS = ['is_international', 'unusual_amount_flag', 'multiple_transactions_short_time', 'high_risk_device_flag']
NUM_COLS = ['transaction_amount', 'transaction_frequency_24h', 'avg_transaction_amount_7d', 'failed_transaction_count_24h', 'account_age_days']
CAT_COLS = ['payment_method', 'device_type', 'location', 'merchant_category']

FEATURE_COLS = [
    'transaction_amount', 'transaction_frequency_24h', 'avg_transaction_amount_7d',
    'failed_transaction_count_24h', 'account_age_days',
    'is_international', 'unusual_amount_flag', 'multiple_transactions_short_time', 'high_risk_device_flag',
    'payment_method', 'device_type', 'location', 'merchant_category',
    'hour', 'dayofweek', 'is_weekend', 'is_night',
    'amount_vs_avg_ratio', 'amount_deviation', 'risk_score',
    'high_freq_flag', 'failed_x_freq', 'new_account',
    'cust_fraud_rate', 'cust_txn_count', 'merchant_fraud_rate', 'location_fraud_rate'
]
