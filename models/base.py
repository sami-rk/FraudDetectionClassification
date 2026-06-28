from abc import ABC, abstractmethod
import numpy as np


class BaseModel(ABC):
    def __init__(self, params=None):
        self.params = params or {}
        self.model = None

    @abstractmethod
    def fit(self, X, y, X_val=None, y_val=None):
        pass

    @abstractmethod
    def predict_proba(self, X):
        pass

    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)
        return (proba > threshold).astype(int)
