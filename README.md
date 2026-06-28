# Task 1: Fraud Detection Classification

Binary classification model to predict whether a retail transaction is fraudulent.

## Project Structure

| File | Description |
|---|---|
| `config.py` | Constants, column lists, feature definitions, CV settings |
| `preprocessing.py` | Boolean normalization, timestamp parsing, median imputation, label encoding |
| `feature_engineering.py` | Derived features (ratios, risk scores, interaction terms) |
| `training.py` | Stratified 5-fold CV, threshold optimization, evaluation metrics, multi-model comparison |
| `prediction.py` | Full-data retraining, test prediction, submission CSV generation |
| `pipeline.py` | CLI orchestrator with single/compare/finetune modes |
| `models/` | Modular model registry with 8 swappable classifiers |
| `finetune/` | Hyperparameter tuning (Optuna TPE + GridSearchCV) |

## Models

| Model | Class | Key Features |
|---|---|---|
| `lightgbm` | Gradient Boosting | Early stopping, original defaults |
| `xgboost` | Gradient Boosting | scale_pos_weight for imbalance |
| `catboost` | Gradient Boosting | Built-in F1 eval, categorical handling |
| `random_forest` | Bagging | class_weight='balanced' |
| `logistic_regression` | Linear | StandardScaler, class_weight='balanced' |
| `linear_svc` | Linear SVM | CalibratedClassifierCV for probabilities |
| `svc` | RBF SVM | StandardScaler, probability=True |
| `nu_svc` | Nu-SVM | StandardScaler, probability=True |

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run default model (lightgbm)
python pipeline.py

# Run specific model
python pipeline.py xgboost

# Compare all available models
python pipeline.py compare

# Compare specific models
python pipeline.py compare lightgbm xgboost catboost

# Finetune with Optuna (default, 50 trials)
python pipeline.py finetune lightgbm optuna 50

# Finetune with GridSearch
python pipeline.py finetune xgboost grid_search

# List available models
python pipeline.py list
```

## Pipeline Steps

1. **Load Data** — Read train and test datasets
2. **Preprocessing** — Normalize booleans, fix noisy strings, parse timestamps, fill missing with medians
3. **Label Encoding** — Fit encoders on combined train+test categorical columns
4. **Feature Engineering** — Create ratios, risk scores, interaction terms
5. **Aggregation Features** — Compute customer/merchant/location fraud rates per CV fold (leakage-free)
6. **Cross-Validation** — 5-fold StratifiedKFold with early stopping (gradient boosting models)
7. **Threshold Optimization** — Search 0.0–1.0 to maximize F1 on OOF predictions
8. **Final Model** — Retrain on full training data, predict test set

## Finetuning

### Optuna (Default)
- TPE-based Bayesian optimization
- Predefined search spaces per model
- Configurable n_trials (default 50)
- Inner CV loop with threshold search

### GridSearchCV
- Exhaustive grid search over smaller parameter grids
- Sklearn-compatible interface
- Parallel execution with n_jobs=-1

## Results (OOF — LightGBM baseline)

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

## Features Used (27)

- **Numerical**: `transaction_amount`, `transaction_frequency_24h`, `avg_transaction_amount_7d`, `failed_transaction_count_24h`, `account_age_days`
- **Boolean**: `is_international`, `unusual_amount_flag`, `multiple_transactions_short_time`, `high_risk_device_flag`
- **Categorical**: `payment_method`, `device_type`, `location`, `merchant_category`
- **Temporal**: `hour`, `dayofweek`, `is_weekend`, `is_night`
- **Engineered**: `amount_vs_avg_ratio`, `amount_deviation`, `risk_score`, `high_freq_flag`, `failed_x_freq`, `new_account`
- **Aggregation**: `cust_fraud_rate`, `cust_txn_count`, `merchant_fraud_rate`, `location_fraud_rate`
