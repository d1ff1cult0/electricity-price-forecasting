import numpy as np
from typing import Dict, List


class PersistenceResidual:
    # Persistence forecast with empirical residual quantiles
    # Point forecast = last observed value
    # Quantiles = point_forecast + empirical quantile of historical residuals


    def __init__(self, quantiles: List[float] = None):
        self.quantiles = quantiles or [0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975]
        self.residual_quantiles_: np.ndarray = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PersistenceResidual":
        last_price = X[:, -1, 0]
        point_pred = np.tile(last_price.reshape(-1, 1), (1, y.shape[1]))
        residuals = y - point_pred
        self.residual_quantiles_ = np.array(
            [[np.quantile(residuals[:, h], q) for q in self.quantiles]
             for h in range(y.shape[1])]
        )
        return self

    def predict(self, X: np.ndarray) -> Dict[float, np.ndarray]:
        last_price = X[:, -1, 0]
        point_pred = np.tile(
            last_price.reshape(-1, 1), (1, self.residual_quantiles_.shape[0])
        )
        preds = {}
        for i, q in enumerate(self.quantiles):
            preds[q] = point_pred + self.residual_quantiles_[:, i]
        return preds
