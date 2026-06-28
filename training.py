import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score, confusion_matrix, matthews_corrcoef, cohen_kappa_score
from config import FEATURE_COLS, CV_N_SPLITS, CV_RANDOM_STATE
from models.registry import get_model


def cross_validate_and_evaluate(train_p, y, global_mean, model_name='lightgbm', model_params=None):
    skf = StratifiedKFold(n_splits=CV_N_SPLITS, shuffle=True, random_state=CV_RANDOM_STATE)
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

        model = get_model(model_name, model_params)
        model.fit(X_tr, y_tr, X_val, y_val)
        oof_preds[val_idx] = model.predict_proba(X_val)

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

    return {
        'best_thresh': best_thresh,
        'f1': f1_score(y, oof_labels),
        'roc_auc': roc_auc_score(y, oof_preds),
        'precision': precision_score(y, oof_labels),
        'recall': recall_score(y, oof_labels),
        'accuracy': accuracy_score(y, oof_labels),
        'mcc': matthews_corrcoef(y, oof_labels),
        'specificity': tn / (tn + fp),
    }


def cross_validate_all_models(train_p, y, global_mean, model_names=None, model_params_map=None):
    from models import list_models as _list_models
    if model_names is None:
        model_names = _list_models()
    if model_params_map is None:
        model_params_map = {}

    results = {}
    for name in model_names:
        print(f"\n{'='*50}")
        print(f"  Training: {name}")
        print(f"{'='*50}")
        try:
            params = model_params_map.get(name)
            res = cross_validate_and_evaluate(train_p, y, global_mean, model_name=name, model_params=params)
            results[name] = res
        except Exception as e:
            print(f"  FAILED: {e}")
            results[name] = {'error': str(e)}

    return results
