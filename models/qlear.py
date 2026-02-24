import numpy as np
from sklearn.linear_model import QuantileRegressor
from sklearn.multioutput import MultiOutputRegressor
from typing import Dict, List, Any
import warnings

class QLear:
    # QLEAR: Quantile Lasso-Estimated AutoRegressive

    def __init__(self, quantiles: List[float] = None, alpha: float = 1.0, params: Dict[str, Any] = None):
        self.quantiles = quantiles or [0.1, 0.5, 0.9]
        self.alpha = alpha
        self.params = params or {}
        self.models: Dict[float, Any] = {}
        self.fallback_solver = False
        
    def fit(self, X, y):
        print(f"Training QLEAR (Quantile Lasso) for quantiles {self.quantiles}...")
        n_samples = X.shape[0]
            
        for q in self.quantiles:
            base_estimator = QuantileRegressor(
                quantile=q, 
                alpha=self.alpha, 
                solver='highs' 
            )
            
            multi_model = MultiOutputRegressor(base_estimator)
            multi_model.fit(X, y)
            self.models[q] = multi_model
            
    def predict(self, X):
        preds = {}
        for q, model in self.models.items():
            preds[q] = model.predict(X)
        return preds
