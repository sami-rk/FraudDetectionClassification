from sklearn.svm import NuSVC
from sklearn.preprocessing import StandardScaler
from models.base import BaseModel
from models.registry import register_model


class NuSVCModel(BaseModel):
    DEFAULT_PARAMS = {
        'nu': 0.5,
        'kernel': 'rbf',
        'gamma': 'scale',
        'probability': True,
        'class_weight': 'balanced',
        'random_state': 42,
    }

    def __init__(self, params=None):
        super().__init__(params)
        merged = {**self.DEFAULT_PARAMS, **self.params}
        self.scaler = StandardScaler()
        self.model = NuSVC(**merged)

    def fit(self, X, y, X_val=None, y_val=None):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]


register_model('nu_svc', NuSVCModel)
