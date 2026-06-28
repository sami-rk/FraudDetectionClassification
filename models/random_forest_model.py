from sklearn.ensemble import RandomForestClassifier
from models.base import BaseModel
from models.registry import register_model


class RandomForestModel(BaseModel):
    DEFAULT_PARAMS = {
        'n_estimators': 500,
        'max_depth': 15,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': -1,
    }

    def __init__(self, params=None):
        super().__init__(params)
        merged = {**self.DEFAULT_PARAMS, **self.params}
        self.model = RandomForestClassifier(**merged)

    def fit(self, X, y, X_val=None, y_val=None):
        self.model.fit(X, y)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]


register_model('random_forest', RandomForestModel)
