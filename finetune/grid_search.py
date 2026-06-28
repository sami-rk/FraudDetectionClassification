import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import f1_score, make_scorer
from models.registry import get_model
from models.base import BaseModel
import warnings
warnings.filterwarnings('ignore')


PARAM_GRIDS = {
    'lightgbm': {
        'n_estimators': [200, 500],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [5, 7, 9],
    },
    'xgboost': {
        'n_estimators': [200, 500],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [5, 7, 9],
    },
    'catboost': {
        'iterations': [200, 500],
        'learning_rate': [0.01, 0.05, 0.1],
        'depth': [5, 7],
    },
    'random_forest': {
        'n_estimators': [200, 500],
        'max_depth': [10, 15, None],
    },
    'logistic_regression': {
        'C': [0.01, 0.1, 1.0, 10.0],
    },
    'linear_svc': {
        'C': [0.01, 0.1, 1.0, 10.0],
    },
    'svc': {
        'C': [0.1, 1.0, 10.0],
        'gamma': [0.01, 0.1, 1.0],
    },
    'nu_svc': {
        'nu': [0.2, 0.5, 0.8],
        'gamma': [0.01, 0.1, 1.0],
    },
}


class _SklearnWrapper(BaseModel):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self._needs_scale = hasattr(model, 'scaler') or 'SVC' in type(model).__name__

    def fit(self, X, y, X_val=None, y_val=None):
        self.model.fit(X, y)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]


class GridSearchTuner:
    def __init__(self, model_name, cv=5):
        if model_name not in PARAM_GRIDS:
            raise ValueError(f"No param grid defined for: {model_name}")
        self.model_name = model_name
        self.param_grid = PARAM_GRIDS[model_name]
        self.cv = cv

    def tune(self, X, y, global_mean=0.5):
        base_model = get_model(self.model_name, {'random_state': 42})

        scorer = make_scorer(f1_score, greater_is_better=True)

        gs = GridSearchCV(
            estimator=base_model.model,
            param_grid=self.param_grid,
            scoring=scorer,
            cv=StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=42),
            n_jobs=-1,
            refit=True,
        )
        gs.fit(X, y)

        return {
            'best_params': gs.best_params_,
            'best_score': gs.best_score_,
            'grid_search': gs,
        }
