from catboost import CatBoostClassifier
from models.base import BaseModel
from models.registry import register_model


class CatBoostModel(BaseModel):
    DEFAULT_PARAMS = {
        'iterations': 500,
        'learning_rate': 0.05,
        'depth': 7,
        'l2_leaf_reg': 3,
        'random_state': 42,
        'verbose': 0,
    }

    def __init__(self, params=None):
        super().__init__(params)
        merged = {**self.DEFAULT_PARAMS, **self.params}
        self.model = CatBoostClassifier(**merged)

    def fit(self, X, y, X_val=None, y_val=None):
        if X_val is not None and y_val is not None:
            self.model.fit(
                X, y,
                eval_set=(X_val, y_val),
                early_stopping_rounds=50,
                verbose=False,
            )
        else:
            self.model.fit(X, y, verbose=False)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]


register_model('catboost', CatBoostModel)
