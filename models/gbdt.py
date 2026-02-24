import numpy as np
import xgboost as xgb
import lightgbm as lgb
from typing import Dict, List, Any
from sklearn.base import BaseEstimator, RegressorMixin

class QuantileGBDT:
    # Wrapper for GBDT models to perform Quantile Regression which Trains K regressors (one for each quantile)
    def __init__(self, model_type: str = "xgboost", quantiles: List[float] = None, params: Dict[str, Any] = None):
        self.model_type = model_type
        self.quantiles = quantiles or [0.1, 0.5, 0.9]
        self.params = params or {}
        self.models: Dict[float, Any] = {}
        
    def fit(self, X, y):
        import time
        print(f"Training {self.model_type} for quantiles {self.quantiles}...")
        
        # Get horizon from y shape
        if len(y.shape) > 1:
            horizon = y.shape[1]
        else:
            horizon = 1
            y = y.reshape(-1, 1)
            
        for i, q in enumerate(self.quantiles):
            msg = f"[{time.strftime('%H:%M:%S')}] Quantile {q} ({i+1}/{len(self.quantiles)}): Starting training for {horizon} steps..."
            print(msg, flush=True)
            
            self.models[q] = []
            
            for h in range(horizon):
                if h % 1 == 0:
                     print(f"  > Step {h+1}/{horizon}...", end="\r", flush=True)
                
                y_h = y[:, h]
                
                if self.model_type == "xgboost":
                    n_est = min(self.params.get('n_estimators', 100), 300)
                    max_d = min(self.params.get('max_depth', 5), 7)
                    
                    model = xgb.XGBRegressor(
                        objective='reg:quantileerror',
                        quantile_alpha=q,
                        n_estimators=n_est,
                        learning_rate=self.params.get('learning_rate', 0.1),
                        max_depth=max_d,
                        subsample=self.params.get('subsample', 1.0),
                        colsample_bytree=self.params.get('colsample_bytree', 0.6),
                        max_bin=self.params.get('max_bin', 128),
                        tree_method='hist',
                        n_jobs=4 
                    )
                        
                elif self.model_type == "lightgbm":
                    model = lgb.LGBMRegressor(
                        objective='quantile',
                        alpha=q,
                        n_estimators=self.params.get('n_estimators', 100),
                        learning_rate=self.params.get('learning_rate', 0.1),
                        num_leaves=self.params.get('num_leaves', 31),
                        n_jobs=4, 
                        verbose=-1
                    )
                
                model.fit(X, y_h)
                self.models[q].append(model)
            
            print(f"  > Step {horizon}/{horizon} done", flush=True)
            
    def predict(self, X):
        # Returns dictionary of quantile predictions {q: (n_samples, horizon)}
        preds = {}
        for q, models_list in self.models.items():
            q_preds = []
            for h_model in models_list:
                q_preds.append(h_model.predict(X))
            
            preds[q] = np.column_stack(q_preds)
        return preds
