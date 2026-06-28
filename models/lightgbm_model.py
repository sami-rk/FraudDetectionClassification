import lightgbm as lgb
from models.base import BaseModel
from models.registry import register_model


class LightGBMModel(BaseModel):
    DEFAULT_PARAMS = {
        'n_estimators': 500,
        'learning_rate': 0.05,
        'max_depth': 7,
        'num_leaves': 63,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1,
    }

    def __init__(self, params=None):
        super().__init__(params)
        merged = {**self.DEFAULT_PARAMS, **self.params}
        self.model = lgb.LGBMClassifier(**merged)

    def fit(self, X, y, X_val=None, y_val=None):
        callbacks = []
        if X_val is not None and y_val is not None:
            callbacks.append(lgb.early_stopping(50, verbose=False))
            self.model.fit(X, y, eval_set=[(X_val, y_val)], callbacks=callbacks)
        else:
            self.model.fit(X, y)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]


register_model('lightgbm', LightGBMModel)
