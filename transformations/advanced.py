import numpy as np
from typing import Tuple, Optional
from sklearn.preprocessing import PowerTransformer, RobustScaler
from .base import BaseTransformation

class YeoJohnsonTransformation(BaseTransformation):
    # Yeo-Johnson Power Transformation which makes data more Gaussian

    def __init__(self):
        self.scaler_X = PowerTransformer(method='yeo-johnson', standardize=True)
        self.scaler_y = PowerTransformer(method='yeo-johnson', standardize=True)
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


class ArcsinhTransformation(BaseTransformation):
    # Inverse Hyperbolic Sine Transformation (y = arcsinh(x) = ln(x + sqrt(x^2 + 1)))

    def __init__(self):
        self._scaler_X_internal = None
        self._scaler_y_internal = None
        from sklearn.preprocessing import StandardScaler
        self._scaler_X_internal = StandardScaler()
        self._scaler_y_internal = StandardScaler()
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        # 1. Apply Arcsinh
        X_trans = np.arcsinh(X)
        X_flat = X_trans.reshape(-1, X.shape[-1])
        # 2. Fit Standard Scaler on transformed data
        self._scaler_X_internal.fit(X_flat)
        
        if y is not None:
            y_trans = np.arcsinh(y)
            y_flat = y_trans.reshape(-1, 1)
            self._scaler_y_internal.fit(y_flat)
            
        self.is_fitted = True

    def transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if not self.is_fitted:
            raise ValueError("Transformation not fitted yet")
            
        # X
        X_trans = np.arcsinh(X)
        X_shape = X.shape
        X_flat = X_trans.reshape(-1, X_shape[-1])
        X_scaled = self._scaler_X_internal.transform(X_flat).reshape(X_shape)
        
        y_scaled = None
        if y is not None:
            y_trans = np.arcsinh(y)
            y_shape = y.shape
            y_flat = y_trans.reshape(-1, 1)
            y_scaled = self._scaler_y_internal.transform(y_flat).reshape(y_shape)
            
        return X_scaled, y_scaled

    def inverse_transform(self, X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if not self.is_fitted:
            raise ValueError("Transformation not fitted yet")
            
        X_inv = None
        if X is not None:
            X_shape = X.shape
            X_flat = X.reshape(-1, X_shape[-1])
            # Inverse Standard Scaler
            X_unscaled = self._scaler_X_internal.inverse_transform(X_flat).reshape(X_shape)
            # Inverse Arcsinh (sinh)
            X_inv = np.sinh(X_unscaled)
            
        y_inv = None
        if y is not None:
            y_shape = y.shape
            y_flat = y.reshape(-1, 1)
            # Inverse Standard Scaler
            y_unscaled = self._scaler_y_internal.inverse_transform(y_flat).reshape(y_shape)
            # Inverse Arcsinh (sinh)
            y_inv = np.sinh(y_unscaled)
            
        return X_inv, y_inv


class RobustScalerTransformation(BaseTransformation):
    # Robust scaling using median and IQR

    def __init__(self):
        self.scaler_X = RobustScaler()
        self.scaler_y = RobustScaler()
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
