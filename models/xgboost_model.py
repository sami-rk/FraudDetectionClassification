import xgboost as xgb
from models.base import BaseModel
from models.registry import register_model


class XGBoostModel(BaseModel):
    DEFAULT_PARAMS = {
        'n_estimators': 500,
        'learning_rate': 0.05,
        'max_depth': 7,
        'min_child_weight': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0,
    }

    def __init__(self, params=None):
        super().__init__(params)
        merged = {**self.DEFAULT_PARAMS, **self.params}
        self.model = xgb.XGBClassifier(**merged)

    def fit(self, X, y, X_val=None, y_val=None):
        if X_val is not None and y_val is not None:
            self.model.fit(
                X, y,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
        else:
            self.model.fit(X, y)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]


register_model('xgboost', XGBoostModel)
