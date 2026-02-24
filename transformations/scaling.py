import numpy as np
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler
from .base import BaseTransformation

class StandardScalingTransformation(BaseTransformation):
    def __init__(self):
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        X_flat = X.reshape(-1, X.shape[-1])
        self.scaler_X.fit(X_flat)
        
        if y is not None:
            y_flat = y.reshape(-1, 1)
            self.scaler_y.fit(y_flat)
            
        self.is_fitted = True

    def transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if not self.is_fitted:
            raise ValueError("Transformation not fitted yet")
            
        # Transform X
        X_shape = X.shape
        X_flat = X.reshape(-1, X_shape[-1])
        X_scaled = self.scaler_X.transform(X_flat).reshape(X_shape)
        
        y_scaled = None
        if y is not None:
            y_shape = y.shape
            y_flat = y.reshape(-1, 1)
            y_scaled = self.scaler_y.transform(y_flat).reshape(y_shape)
            
        return X_scaled, y_scaled

    def inverse_transform(self, X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if not self.is_fitted:
            raise ValueError("Transformation not fitted yet")
            
        X_inv = None
        if X is not None:
            X_shape = X.shape
            X_flat = X.reshape(-1, X_shape[-1])
            X_inv = self.scaler_X.inverse_transform(X_flat).reshape(X_shape)
            
        y_inv = None
        if y is not None:
            y_shape = y.shape
            y_flat = y.reshape(-1, 1)
            y_inv = self.scaler_y.inverse_transform(y_flat).reshape(y_shape)
            
        return X_inv, y_inv
