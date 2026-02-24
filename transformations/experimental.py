import numpy as np
import pandas as pd
from typing import Tuple, Optional, Union, List
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, RobustScaler
from scipy import stats
from .base import BaseTransformation
from .advanced import ArcsinhTransformation, YeoJohnsonTransformation


class MirrorLogTransformation(BaseTransformation):
    # Mirror-Log transform Handles negative values while compressing magnitude (f(x) = sign(x) * ln(1 + |x|))

    def __init__(self):
        self.is_fitted = False
        self._scaler_X = None
        self._scaler_y = None
        from sklearn.preprocessing import StandardScaler
        self._scaler_X = StandardScaler()
        self._scaler_y = StandardScaler()

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        # Apply Mirror-Log
        X_trans = np.sign(X) * np.log1p(np.abs(X))
        X_flat = X_trans.reshape(-1, X.shape[-1])
        self._scaler_X.fit(X_flat)
        
        if y is not None:
            y_trans = np.sign(y) * np.log1p(np.abs(y))
            y_flat = y_trans.reshape(-1, 1)
            self._scaler_y.fit(y_flat)
            
        self.is_fitted = True

    def transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if not self.is_fitted:
            raise ValueError("Not fitted")
            
        X_trans = np.sign(X) * np.log1p(np.abs(X))
        X_shape = X.shape
        X_flat = X_trans.reshape(-1, X_shape[-1])
        X_scaled = self._scaler_X.transform(X_flat).reshape(X_shape)
        
        y_scaled = None
        if y is not None:
            y_trans = np.sign(y) * np.log1p(np.abs(y))
            y_shape = y.shape
            y_flat = y_trans.reshape(-1, 1)
            y_scaled = self._scaler_y.transform(y_flat).reshape(y_shape)
            
        return X_scaled, y_scaled

    def inverse_transform(self, X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        # Inverse: x = sign(y) * (exp(|y|) - 1)
        X_inv = None
        if X is not None:
            X_shape = X.shape
            X_flat = X.reshape(-1, X_shape[-1])
            X_unscaled = self._scaler_X.inverse_transform(X_flat).reshape(X_shape)
            X_inv = np.sign(X_unscaled) * (np.expm1(np.abs(X_unscaled)))
            
        y_inv = None
        if y is not None:
            y_shape = y.shape
            y_flat = y.reshape(-1, 1)
            y_unscaled = self._scaler_y.inverse_transform(y_flat).reshape(y_shape)
            y_inv = np.sign(y_unscaled) * (np.expm1(np.abs(y_unscaled)))
            
        return X_inv, y_inv

class ProbabilityIntegralTransform(BaseTransformation):
    # Maps data to uniform distribution [0, 1] using empirical CDF (QuantileTransformer).

    def __init__(self):
        self.scaler_X = QuantileTransformer(output_distribution='uniform', n_quantiles=1000)
        self.scaler_y = QuantileTransformer(output_distribution='uniform', n_quantiles=1000)
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        X_flat = X.reshape(-1, X.shape[-1])
        if len(X_flat) > 100000:
             idx = np.random.choice(len(X_flat), 100000, replace=False)
             self.scaler_X.fit(X_flat[idx])
        else:
             self.scaler_X.fit(X_flat)
        
        if y is not None:
            y_flat = y.reshape(-1, 1)
            self.scaler_y.fit(y_flat)
            
        self.is_fitted = True

    def transform(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        X_shape = X.shape
        X_flat = X.reshape(-1, X_shape[-1])
        X_scaled = self.scaler_X.transform(X_flat).reshape(X_shape)
        
        y_scaled = None
        if y is not None:
            y_shape = y.shape
            y_flat = y.reshape(-1, 1)
            y_scaled = self.scaler_y.transform(y_flat).reshape(y_shape)
            
        return X_scaled, y_scaled

    def inverse_transform(self, X=None, y=None):
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


class BoxCoxTransformation(BaseTransformation):
    # Since Box-Cox needs strictly positive data we shift the data when needed

    def __init__(self):
        self.scaler_X = PowerTransformer(method='box-cox', standardize=True)
        self.scaler_y = PowerTransformer(method='box-cox', standardize=True)
        self.is_fitted = False
        self.shift_X = 0
        self.shift_y = 0

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        X_flat = X.reshape(-1, X.shape[-1])
        min_X = np.min(X_flat)
        if min_X <= 0:
            self.shift_X = abs(min_X) + 1e-6
        
        self.scaler_X.fit(X_flat + self.shift_X)
        
        if y is not None:
            y_flat = y.reshape(-1, 1)
            min_y = np.min(y_flat)
            if min_y <= 0:
                self.shift_y = abs(min_y) + 1e-6
            self.scaler_y.fit(y_flat + self.shift_y)
            
        self.is_fitted = True

    def transform(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        X_shape = X.shape
        X_flat = X.reshape(-1, X_shape[-1])
        X_shifted = np.clip(X_flat + self.shift_X, 1e-6, None)
        X_scaled = self.scaler_X.transform(X_shifted).reshape(X_shape)
        
        y_scaled = None
        if y is not None:
            y_shape = y.shape
            y_flat = y.reshape(-1, 1)
            y_shifted = np.clip(y_flat + self.shift_y, 1e-6, None)
            y_scaled = self.scaler_y.transform(y_shifted).reshape(y_shape)
            
        return X_scaled, y_scaled

    def inverse_transform(self, X=None, y=None):
        X_inv = None
        if X is not None:
            X_shape = X.shape
            X_flat = X.reshape(-1, X_shape[-1])
            X_inv = (self.scaler_X.inverse_transform(X_flat) - self.shift_X).reshape(X_shape)
            
        y_inv = None
        if y is not None:
            y_shape = y.shape
            y_flat = y.reshape(-1, 1)
            y_inv = (self.scaler_y.inverse_transform(y_flat) - self.shift_y).reshape(y_shape)
            
        return X_inv, y_inv

class MADScalingTransformation(BaseTransformation):
    # MAD Scaling (Median Absolute Deviation, scales by: (x - median) / MAD)

    def __init__(self):
        self.is_fitted = False
        self.median_X = None
        self.mad_X = None
        self.median_y = None
        self.mad_y = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        X_flat = X.reshape(-1, X.shape[-1])
        self.median_X = np.nanmedian(X_flat, axis=0)
        self.mad_X = np.nanmedian(np.abs(X_flat - self.median_X), axis=0)
        # Avoid division by zero
        self.mad_X[self.mad_X == 0] = 1.0

        if y is not None:
            y_flat = y.reshape(-1, 1)
            self.median_y = np.nanmedian(y_flat, axis=0)
            self.mad_y = np.nanmedian(np.abs(y_flat - self.median_y), axis=0)
            self.mad_y[self.mad_y == 0] = 1.0
            
        self.is_fitted = True

    def transform(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        X_shape = X.shape
        X_flat = X.reshape(-1, X_shape[-1])
        X_scaled = ((X_flat - self.median_X) / self.mad_X).reshape(X_shape)
        
        y_scaled = None
        if y is not None:
            y_shape = y.shape
            y_flat = y.reshape(-1, 1)
            y_scaled = ((y_flat - self.median_y) / self.mad_y).reshape(y_shape)
            
        return X_scaled, y_scaled

    def inverse_transform(self, X=None, y=None):
        X_inv = None
        if X is not None:
            X_shape = X.shape
            X_flat = X.reshape(-1, X_shape[-1])
            X_inv = ((X_flat * self.mad_X) + self.median_X).reshape(X_shape)
            
        y_inv = None
        if y is not None:
            y_shape = y.shape
            y_flat = y.reshape(-1, 1)
            y_inv = ((y_flat * self.mad_y) + self.median_y).reshape(y_shape)
        
        return X_inv, y_inv

class QuantileGaussianTransformation(BaseTransformation):
    # # Quantile Transformer: maps to Gaussian Distribution via Quantiles

    def __init__(self):
        self.scaler_X = QuantileTransformer(output_distribution='normal', n_quantiles=1000)
        self.scaler_y = QuantileTransformer(output_distribution='normal', n_quantiles=1000)
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        X_flat = X.reshape(-1, X.shape[-1])
        if len(X_flat) > 100000:
             idx = np.random.choice(len(X_flat), 100000, replace=False)
             self.scaler_X.fit(X_flat[idx])
        else:
             self.scaler_X.fit(X_flat)
        
        if y is not None:
            y_flat = y.reshape(-1, 1)
            self.scaler_y.fit(y_flat)
        self.is_fitted = True

    def transform(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        X_shape = X.shape
        X_flat = X.reshape(-1, X_shape[-1])
        X_scaled = self.scaler_X.transform(X_flat).reshape(X_shape)
        
        y_scaled = None
        if y is not None:
            y_shape = y.shape
            y_flat = y.reshape(-1, 1)
            y_scaled = self.scaler_y.transform(y_flat).reshape(y_shape)
        return X_scaled, y_scaled

    def inverse_transform(self, X=None, y=None):
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

class STLDecompositionTransformation(BaseTransformation):
    # Decomposition into Trend + Seasonal + Residual

    def __init__(self, period=24):
        self.period = period
        self.is_fitted = False
        self.overall_mean = 0
        
        from sklearn.preprocessing import StandardScaler
        self.scaler_X = StandardScaler()
        self.scaler_resid = StandardScaler()
        
        self.seasonal_component = None
        self.trend_model = None 
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        # Fit X normally
        X_flat = X.reshape(-1, X.shape[-1])
        self.scaler_X.fit(X_flat)
        
        if y is not None:
            pass
            
        self.scaler_resid.fit(y.reshape(-1, 1))
        self.is_fitted = True

    def transform(self, X, y=None):
        X_scaled = self.scaler_X.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        y_scaled = None
        if y is not None:
             y_scaled = self.scaler_resid.transform(y.reshape(-1, 1)).reshape(y.shape)
        return X_scaled, y_scaled

    def inverse_transform(self, X=None, y=None):
        X_inv = None
        if X is not None:
             X_inv = self.scaler_X.inverse_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        y_inv = None
        if y is not None:
             y_inv = self.scaler_resid.inverse_transform(y.reshape(-1, 1)).reshape(y.shape)
        return X_inv, y_inv



class WaveletTransformation(BaseTransformation):
    def __init__(self, wavelet='db4', level=None, threshold_mode='soft'):
        self.is_fitted = False
        self.wavelet = wavelet
        self.level = level
        self.threshold_mode = threshold_mode
        self._threshold = None

        import pywt
        self._pywt = pywt
        from sklearn.preprocessing import StandardScaler
        self._scaler_X = StandardScaler()
        self._scaler_y = StandardScaler()

    def _denoise(self, X):
        out = np.empty_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[2]):
                coeffs = self._pywt.wavedec(X[i, :, j], self.wavelet, level=self.level)
                coeffs[1:] = [self._pywt.threshold(c, self._threshold, mode=self.threshold_mode)
                              for c in coeffs[1:]]
                rec = self._pywt.waverec(coeffs, self.wavelet)
                out[i, :, j] = rec[:X.shape[1]]
        return out

    def fit(self, X, y=None):
        coeffs = self._pywt.wavedec(X[:, :, 0].ravel(), self.wavelet, level=self.level)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        self._threshold = sigma * np.sqrt(2 * np.log(X.shape[1]))

        X_dn = self._denoise(X)
        self._scaler_X.fit(X_dn.reshape(-1, X.shape[-1]))

        if y is not None:
            self._scaler_y.fit(y.reshape(-1, 1))

        self.is_fitted = True

    def transform(self, X, y=None):
        X_dn = self._denoise(X)
        X_scaled = self._scaler_X.transform(X_dn.reshape(-1, X.shape[-1])).reshape(X.shape)

        y_scaled = None
        if y is not None:
            y_scaled = self._scaler_y.transform(y.reshape(-1, 1)).reshape(y.shape)

        return X_scaled, y_scaled

    def inverse_transform(self, X=None, y=None):
        X_inv = None
        y_inv = None
        if X is not None:
            X_inv = self._scaler_X.inverse_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        if y is not None:
            y_inv = self._scaler_y.inverse_transform(y.reshape(-1, 1)).reshape(y.shape)
        return X_inv, y_inv


class VMDTransformation(BaseTransformation):
    # Variational Mode Decomposition (VMD)
    def __init__(self):
        self.scaler = RobustScaler()
        self.is_fitted = False
    
    def fit(self, X, y=None):
        self.scaler.fit(X.reshape(-1, X.shape[-1]))
        if y is not None:
             pass
        self.is_fitted = True

    def transform(self, X, y=None):
        X_s = self.scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        y_s = None
        if y is not None:
             y_s = (y - np.median(y)) / (np.percentile(y, 75) - np.percentile(y, 25))
        return X_s, y_s
    
    def inverse_transform(self, X=None, y=None):
        return None, y 

class DifferencingTransformation(BaseTransformation):
    # Differencing (Stationarization) y_t = y_t - y_{t-1} (accumulate predictions to recover)

    def __init__(self, order=1):
        self.order = order
        self.last_known_values = None 
        self.is_fitted = False
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        X_diff = np.diff(X, axis=1, prepend=X[:, :1, :])
        self.is_fitted = True

    def transform(self, X, y=None):
        X_diff = np.diff(X, axis=1)
        X_diff = np.pad(X_diff, ((0, 0), (1, 0), (0, 0)), mode='constant')
        
        y_diff = None
        if y is not None:
            y_diff = y 
            
        return X_diff, y_diff
    
    def inverse_transform(self, X=None, y=None):
        return X, y

class WinsorizationTransformation(BaseTransformation):
    # Winsorization / Clipping

    def __init__(self, limits=(0.01, 0.01)):
        self.limits = limits
        self.lower = None
        self.upper = None
        self.is_fitted = False
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        X_flat = X.reshape(-1, X.shape[-1])
        self.lower = np.percentile(X_flat, self.limits[0]*100, axis=0)
        self.upper = np.percentile(X_flat, (1-self.limits[1])*100, axis=0)
        
        # Fit scaler on clipped data
        X_clipped = np.clip(X_flat, self.lower, self.upper)
        self.scaler.fit(X_clipped)
        self.is_fitted = True

    def transform(self, X, y=None):
        X_flat = X.reshape(-1, X.shape[-1])
        X_clipped = np.clip(X_flat, self.lower, self.upper)
        X_scaled = self.scaler.transform(X_clipped).reshape(X.shape)
        
        y_scaled = None
        if y is not None:
            y_scaled = y 
            
        return X_scaled, y_scaled
    
    def inverse_transform(self, X=None, y=None):
        # Inverse scaler
        return None, y
