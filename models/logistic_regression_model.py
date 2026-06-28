from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from models.base import BaseModel
from models.registry import register_model


class LogisticRegressionModel(BaseModel):
    DEFAULT_PARAMS = {
        'C': 1.0,
        'solver': 'lbfgs',
        'max_iter': 1000,
        'class_weight': 'balanced',
        'random_state': 42,
    }

    def __init__(self, params=None):
        super().__init__(params)
        merged = {**self.DEFAULT_PARAMS, **self.params}
        self.scaler = StandardScaler()
        self.model = LogisticRegression(**merged)

    def fit(self, X, y, X_val=None, y_val=None):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]


register_model('logistic_regression', LogisticRegressionModel)
