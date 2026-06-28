from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from models.base import BaseModel
from models.registry import register_model


class LinearSVCModel(BaseModel):
    DEFAULT_PARAMS = {
        'C': 1.0,
        'loss': 'squared_hinge',
        'max_iter': 2000,
        'class_weight': 'balanced',
        'random_state': 42,
    }

    def __init__(self, params=None):
        super().__init__(params)
        merged = {**self.DEFAULT_PARAMS, **self.params}
        self.scaler = StandardScaler()
        base_svc = LinearSVC(**merged)
        self.model = CalibratedClassifierCV(base_svc, cv=3)

    def fit(self, X, y, X_val=None, y_val=None):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]


register_model('linear_svc', LinearSVCModel)
