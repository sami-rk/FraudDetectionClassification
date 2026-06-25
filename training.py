import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score, confusion_matrix, matthews_corrcoef, cohen_kappa_score
import lightgbm as lgb
from config import FEATURE_COLS

def cross_validate_and_evaluate(train_p, y, global_mean):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(y))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(train_p, y)):
        df_tr = train_p.iloc[tr_idx]
        df_val = train_p.iloc[val_idx]

        cust_stats = df_tr.groupby('customer_id')['label'].agg(['count', 'mean'])
        merch_stats = df_tr.groupby('merchant_category')['label'].mean()
        loc_stats = df_tr.groupby('location')['label'].mean()

        X_tr_df = df_tr[FEATURE_COLS].copy()
        X_val_df = df_val[FEATURE_COLS].copy()

        X_tr_df['cust_txn_count'] = df_tr['customer_id'].map(cust_stats['count']).fillna(0.0)
        X_tr_df['cust_fraud_rate'] = df_tr['customer_id'].map(cust_stats['mean']).fillna(global_mean)
        X_tr_df['merchant_fraud_rate'] = df_tr['merchant_category'].map(merch_stats).fillna(global_mean)
        X_tr_df['location_fraud_rate'] = df_tr['location'].map(loc_stats).fillna(global_mean)

        X_val_df['cust_txn_count'] = df_val['customer_id'].map(cust_stats['count']).fillna(0.0)
        X_val_df['cust_fraud_rate'] = df_val['customer_id'].map(cust_stats['mean']).fillna(global_mean)
        X_val_df['merchant_fraud_rate'] = df_val['merchant_category'].map(merch_stats).fillna(global_mean)
        X_val_df['location_fraud_rate'] = df_val['location'].map(loc_stats).fillna(global_mean)

        X_tr, X_val = X_tr_df.values, X_val_df.values
        y_tr, y_val = y[tr_idx], y[val_idx]

        model = lgb.LGBMClassifier(
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

        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])

        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]

        fold_f1 = f1_score(y_val, (oof_preds[val_idx] > 0.50).astype(int))
        print(f"Fold {fold+1} F1 (@0.50): {fold_f1:.4f}")

    best_thresh = 0.5
    best_f1 = 0
    for t in np.arange(0.0, 1.0, 0.01):
        f1 = f1_score(y, (oof_preds > t).astype(int))
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    print()
    print(f"OOF Best threshold: {best_thresh:.2f}, F1: {best_f1:.4f}")
    oof_labels = (oof_preds > best_thresh).astype(int)

    print("\n── Evaluation Metrics (OOF) ──")
    print(f"Accuracy:  {accuracy_score(y, oof_labels):.4f}")
    print(f"Precision: {precision_score(y, oof_labels):.4f}")
    print(f"Recall:    {recall_score(y, oof_labels):.4f}")
    print(f"F1 Score:  {f1_score(y, oof_labels):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y, oof_preds):.4f}")
    print(f"MCC:       {matthews_corrcoef(y, oof_labels):.4f}")
    print(f"Cohen Kappa: {cohen_kappa_score(y, oof_labels):.4f}")
    cm = confusion_matrix(y, oof_labels)
    tn, fp, fn, tp = cm.ravel()
    print(f"Confusion Matrix:")
    print(cm)
    print(f"Specificity: {tn / (tn + fp):.4f}")

    return best_thresh
