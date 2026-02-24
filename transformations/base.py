from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional

class BaseTransformation(ABC):
    # Abstract Base Class for data transformations
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        # Fit the transformation parameters
        pass

    @abstractmethod
    def transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        # Apply the transformation
        pass

    @abstractmethod
    def inverse_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        # Revert the transformation
        pass
        
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        self.fit(X, y)
        return self.transform(X, y)
