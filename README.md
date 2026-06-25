# Task 1: Fraud Detection Classification

Binary classification model to predict whether a retail transaction is fraudulent.

## Project Structure

| File | Description |
|---|---|
| `config.py` | Constants, column lists, feature definitions |
| `preprocessing.py` | Boolean normalization, timestamp parsing, median imputation, label encoding |
| `feature_engineering.py` | Derived features (ratios, risk scores, interaction terms) |
| `training.py` | Stratified 5-fold CV, threshold optimization, evaluation metrics |
| `prediction.py` | Full-data retraining, test prediction, submission CSV generation |
| `pipeline.py` | Orchestrates all modules in sequence |
| `task1_classification.py` | Original monolithic script (reference) |

## How to Run

```bash
python pipeline.py
```

## Pipeline Steps

1. **Load Data** — Read train (`datasets/train.xlsx`) and test (`datasets/student_test.xlsx`)
2. **Preprocessing** — Normalize booleans, fix noisy strings (`many`→100, `new_user?`→0), parse timestamps, fill missing with medians
3. **Label Encoding** — Fit encoders on combined train+test categorical columns
4. **Feature Engineering** — Create `amount_vs_avg_ratio`, `risk_score`, `high_freq_flag`, `failed_x_freq`, `new_account`
5. **Aggregation Features** — Compute customer/merchant/location fraud rates within each CV fold (leakage-free)
6. **Cross-Validation** — 5-fold StratifiedKFold with LightGBM, early stopping
7. **Threshold Optimization** — Search 0.0–1.0 to maximize F1 on OOF predictions
8. **Final Model** — Retrain on full training data, predict test set

## Model

- **Algorithm**: LightGBM (Gradient Boosting)
- **Parameters**: n_estimators=500, lr=0.05, max_depth=7, num_leaves=63, subsample=0.8, colsample_bytree=0.8

## Results (OOF)

| Metric | Value |
|---|---|
| Accuracy | 0.6982 |
| Precision | 0.6272 |
| Recall | 0.9011 |
| F1 Score | 0.7396 |
| ROC-AUC | 0.7979 |
| MCC | 0.4458 |
| Cohen Kappa | 0.4070 |
| Specificity | 0.5142 |
| **Best Threshold** | **0.27** |

### Confusion Matrix

|  | Predicted 0 | Predicted 1 |
|---|---|---|
| **Actual 0** | 21,569 | 20,378 |
| **Actual 1** | 3,764 | 34,289 |

### Fold-wise F1 (@0.50)

| Fold | F1 |
|---|---|
| 1 | 0.6660 |
| 2 | 0.6688 |
| 3 | 0.6743 |
| 4 | 0.6699 |
| 5 | 0.6718 |

## Features Used (27)

- **Numerical**: `transaction_amount`, `transaction_frequency_24h`, `avg_transaction_amount_7d`, `failed_transaction_count_24h`, `account_age_days`
- **Boolean**: `is_international`, `unusual_amount_flag`, `multiple_transactions_short_time`, `high_risk_device_flag`
- **Categorical**: `payment_method`, `device_type`, `location`, `merchant_category`
- **Temporal**: `hour`, `dayofweek`, `is_weekend`, `is_night`
- **Engineered**: `amount_vs_avg_ratio`, `amount_deviation`, `risk_score`, `high_freq_flag`, `failed_x_freq`, `new_account`
- **Aggregation**: `cust_fraud_rate`, `cust_txn_count`, `merchant_fraud_rate`, `location_fraud_rate`
