import sys
import time
import pandas as pd
from config import TRAIN_DATA_PATH, TEST_DATA_PATH, FEATURE_COLS, DEFAULT_MODEL
from preprocessing import compute_global_medians, preprocess, fit_encoders
from feature_engineering import add_features
from training import cross_validate_and_evaluate, cross_validate_all_models
from prediction import train_final_and_predict
from models import list_models


def load_and_preprocess():
    train = pd.read_excel(TRAIN_DATA_PATH)
    test = pd.read_excel(TEST_DATA_PATH)
    print(f"Train shape: {train.shape}, Test shape: {test.shape}")
    print(f"Label distribution:\n{train['label'].value_counts()}")

    compute_global_medians(train)
    train_p = preprocess(train, is_train=True)
    test_p = preprocess(test, is_train=False)
    fit_encoders(train_p, test_p)
    train_p = add_features(train_p)
    test_p = add_features(test_p)

    for col in ['cust_fraud_rate', 'cust_txn_count', 'merchant_fraud_rate', 'location_fraud_rate']:
        train_p[col] = 0.0
        test_p[col] = 0.0

    y = train_p['label'].values
    global_mean = y.mean()

    print(f"\nFeatures used: {len(FEATURE_COLS)}")
    return train_p, test_p, y, global_mean


def run_single_model(model_name=None):
    model_name = model_name or DEFAULT_MODEL
    train_p, test_p, y, global_mean = load_and_preprocess()

    print(f"\nTraining model: {model_name}")
    result = cross_validate_and_evaluate(train_p, y, global_mean, model_name=model_name)
    train_final_and_predict(train_p, test_p, result['best_thresh'], global_mean, model_name=model_name)
    return result


def run_compare_models(model_names=None):
    available = list_models()
    if model_names is None:
        model_names = available
    else:
        invalid = [m for m in model_names if m not in available]
        if invalid:
            print(f"Unknown models: {invalid}")
            print(f"Available: {available}")
            sys.exit(1)

    train_p, test_p, y, global_mean = load_and_preprocess()
    start = time.time()
    results = cross_validate_all_models(train_p, y, global_mean, model_names=model_names)
    elapsed = time.time() - start

    rows = []
    for name, res in results.items():
        if 'error' in res:
            rows.append({'model': name, 'error': res['error']})
        else:
            rows.append({
                'model': name,
                'f1': res['f1'],
                'roc_auc': res['roc_auc'],
                'precision': res['precision'],
                'recall': res['recall'],
                'accuracy': res['accuracy'],
                'mcc': res['mcc'],
                'specificity': res['specificity'],
            })

    df = pd.DataFrame(rows)

    print(f"\n{'='*70}")
    print(f"  MODEL COMPARISON RESULTS")
    print(f"{'='*70}")
    print(df.to_string(index=False, float_format='%.4f'))
    print(f"\nTotal time: {elapsed:.1f}s")

    df.to_csv('results_comparison.csv', index=False)
    print(f"Results saved to results_comparison.csv")

    return results


def run_finetune(model_name, method='optuna', n_trials=50):
    from finetune.optuna_tuner import OptunaTuner
    from finetune.grid_search import GridSearchTuner

    available = list_models()
    if model_name not in available:
        print(f"Unknown model: {model_name}")
        print(f"Available: {available}")
        sys.exit(1)

    train_p, test_p, y, global_mean = load_and_preprocess()

    from config import FEATURE_COLS
    import numpy as np
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    X_all = train_p[FEATURE_COLS].values

    print(f"\nFinetuning {model_name} with {method} ({n_trials} trials)...")

    if method == 'optuna':
        tuner = OptunaTuner(model_name, n_trials=n_trials)
    elif method == 'grid_search':
        tuner = GridSearchTuner(model_name)
    else:
        print(f"Unknown method: {method}. Use 'optuna' or 'grid_search'.")
        sys.exit(1)

    result = tuner.tune(X_all, y, global_mean)

    print(f"\nBest params: {result['best_params']}")
    print(f"Best CV F1: {result['best_score']:.4f}")

    print(f"\nRetraining with best params...")
    cv_result = cross_validate_and_evaluate(train_p, y, global_mean, model_name=model_name, model_params=result['best_params'])
    train_final_and_predict(train_p, test_p, cv_result['best_thresh'], global_mean, model_name=model_name, model_params=result['best_params'])

    return result


def print_usage():
    print("Usage: python pipeline.py [command] [args]")
    print()
    print("Commands:")
    print("  (none)                          Run default model (lightgbm)")
    print("  <model_name>                    Run specific model")
    print("  compare [model1 model2 ...]     Compare models")
    print("  finetune <model> [method] [n]   Finetune model (optuna/grid_search)")
    print("  list                            List available models")
    print()
    print("Examples:")
    print("  python pipeline.py")
    print("  python pipeline.py xgboost")
    print("  python pipeline.py compare")
    print("  python pipeline.py compare lightgbm xgboost catboost")
    print("  python pipeline.py finetune lightgbm optuna 50")
    print("  python pipeline.py finetune xgboost grid_search")


if __name__ == '__main__':
    args = sys.argv[1:]

    if not args:
        run_single_model()
    elif args[0] == 'list':
        print("Available models:")
        for m in list_models():
            print(f"  - {m}")
    elif args[0] == 'compare':
        model_names = args[1:] if len(args) > 1 else None
        run_compare_models(model_names)
    elif args[0] == 'finetune':
        if len(args) < 2:
            print("Usage: python pipeline.py finetune <model> [optuna|grid_search] [n_trials]")
            sys.exit(1)
        model_name = args[1]
        method = args[2] if len(args) > 2 else 'optuna'
        n_trials = int(args[3]) if len(args) > 3 else 50
        run_finetune(model_name, method, n_trials)
    elif args[0] in ('-h', '--help', 'help'):
        print_usage()
    elif args[0] in list_models():
        run_single_model(args[0])
    else:
        print(f"Unknown command: {args[0]}")
        print_usage()
        sys.exit(1)
