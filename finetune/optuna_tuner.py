import optuna
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from models.registry import get_model


SEARCH_SPACES = {
    'lightgbm': {
        'n_estimators': (100, 1000),
        'learning_rate': (0.01, 0.3),
        'max_depth': (3, 12),
        'num_leaves': (15, 127),
        'min_child_samples': (5, 50),
        'subsample': (0.5, 1.0),
        'colsample_bytree': (0.5, 1.0),
    },
    'xgboost': {
        'n_estimators': (100, 1000),
        'learning_rate': (0.01, 0.3),
        'max_depth': (3, 12),
        'min_child_weight': (1, 20),
        'subsample': (0.5, 1.0),
        'colsample_bytree': (0.5, 1.0),
    },
    'catboost': {
        'iterations': (100, 1000),
        'learning_rate': (0.01, 0.3),
        'depth': (3, 10),
        'l2_leaf_reg': (1, 10),
    },
    'random_forest': {
        'n_estimators': (100, 1000),
        'max_depth': (5, 30),
        'min_samples_split': (2, 20),
        'min_samples_leaf': (1, 10),
    },
    'logistic_regression': {
        'C': (0.01, 100.0),
    },
    'linear_svc': {
        'C': (0.01, 100.0),
    },
    'svc': {
        'C': (0.01, 100.0),
        'gamma': (0.001, 1.0),
    },
    'nu_svc': {
        'nu': (0.1, 0.9),
        'gamma': (0.001, 1.0),
    },
}


class OptunaTuner:
    def __init__(self, model_name, n_trials=50, timeout=None):
        if model_name not in SEARCH_SPACES:
            raise ValueError(f"No search space defined for: {model_name}")
        self.model_name = model_name
        self.n_trials = n_trials
        self.timeout = timeout
        self.search_space = SEARCH_SPACES[model_name]

    def _suggest_params(self, trial):
        params = {}
        for name, (low, high) in self.search_space.items():
            if isinstance(low, int) and isinstance(high, int):
                params[name] = trial.suggest_int(name, low, high)
            else:
                params[name] = trial.suggest_float(name, low, high, log=True if name in ('learning_rate', 'C', 'gamma', 'l2_leaf_reg') else False)
        return params

    def _objective(self, trial, X, y, global_mean):
        params = self._suggest_params(trial)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_f1s = []

        for tr_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X[tr_idx], X[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]

            model = get_model(self.model_name, params)
            model.fit(X_tr, y_tr, X_val, y_val)
            proba = model.predict_proba(X_val)

            best_f1 = 0
            for t in np.arange(0.1, 0.9, 0.05):
                f1 = f1_score(y_val, (proba > t).astype(int))
                if f1 > best_f1:
                    best_f1 = f1
            fold_f1s.append(best_f1)

        return np.mean(fold_f1s)

    def tune(self, X, y, global_mean=0.5):
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(lambda trial: self._objective(trial, X, y, global_mean), n_trials=self.n_trials, timeout=self.timeout)

        return {
            'best_params': study.best_params,
            'best_score': study.best_value,
            'study': study,
        }
