import numpy as np
import pandas as pd
from typing import Dict, Any, List
from models.transformer import ProbabilisticTransformer
from transformations.base import BaseTransformation

class Evaluator:
    # standardized evaluation to ensure consistent testing on the same test set
    def __init__(self, model: ProbabilisticTransformer, transform: BaseTransformation):
        self.model = model
        self.transform = transform

    def evaluate(self, X_test_scaled: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        evaluate the model on the test set
        X_test_scaled: scaled input features (already transformed/scaled)
        y_test: actual targets in original scale
        """
        if self.model.keras_model is None:
            raise ValueError("Model is not trained/built yet.")
            
        # Predict parameters (scaled)
        y_pred_params = self.model.keras_model.predict(X_test_scaled, verbose=0)
        
        # Generate point forecasts (mean, scaled)
        means_scaled = self.model.head.mean(y_pred_params.reshape(-1, y_pred_params.shape[-1]))
        means_scaled = means_scaled.reshape(y_pred_params.shape[0], -1) # restore batch dim (batch, horizon)
        
        # Inverse transform of mean forecast
        _, means_original = self.transform.inverse_transform(X=None, y=means_scaled)
        
        # Compute deterministic metrics
        mae = np.mean(np.abs(y_test - means_original))
        mse = np.mean((y_test - means_original)**2)
        # MAPE
        mask = y_test != 0
        mape = np.mean(np.abs((y_test[mask] - means_original[mask]) / y_test[mask])) * 100 if mask.any() else np.nan
        # R2
        ss_res = np.sum((y_test - means_original) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

        # Compute probabilistic Metrics
        alpha = 0.05
        q_lower = alpha / 2  # 0.025
        q_upper = 1 - (alpha / 2)  # 0.975
        q_pinball = [0.1, 0.5, 0.9]  # for Pinball losses

        # quantiles in scaled space
        params_flat = y_pred_params.reshape(-1, y_pred_params.shape[-1])
        all_quantiles = [q_lower, q_upper] + q_pinball
        quantiles_dict = self.model.head.quantiles(params_flat, all_quantiles)
        # lower quantile
        q_low_scaled = quantiles_dict[q_lower].reshape(y_pred_params.shape[0], -1)
        _, q_low_original = self.transform.inverse_transform(X=None, y=q_low_scaled)
        # upper quantile
        q_high_scaled = quantiles_dict[q_upper].reshape(y_pred_params.shape[0], -1)
        _, q_high_original = self.transform.inverse_transform(X=None, y=q_high_scaled)
        
        covered = (y_test >= q_low_original) & (y_test <= q_high_original)
        picp = np.mean(covered)
        mpiw = np.mean(q_high_original - q_low_original)
        y_range = np.max(y_test) - np.min(y_test)
        pinaw = mpiw / y_range if y_range > 0 else np.nan

        # Interval score
        # IS = (U - L) + 2/alpha * (L - y) * I(y < L) + 2/alpha *(y - U) * I(y > U)
        width = q_high_original - q_low_original
        lower_penalty = (2/alpha) * (q_low_original - y_test) * (y_test < q_low_original)
        upper_penalty = (2/alpha) * (y_test - q_high_original) * (y_test > q_high_original)
        interval_score = np.mean(width + lower_penalty + upper_penalty)
        
        # CRPS
        n_samples = 100
        samples_scaled = self.model.head.sample(params_flat, n_samples)
        
        samples_scaled = samples_scaled.reshape(n_samples, y_pred_params.shape[0], -1)
        
        samples_flat_scaled = samples_scaled.reshape(-1, samples_scaled.shape[-1])
        _, samples_flat_original = self.transform.inverse_transform(X=None, y=samples_flat_scaled)
        
        samples_original = samples_flat_original.reshape(n_samples, y_pred_params.shape[0], -1)
        
        y_test_exp = np.expand_dims(y_test, axis=0)
        
        term1 = np.mean(np.abs(samples_original - y_test_exp), axis=0) # (batch, horizon)
        samples_sorted = np.sort(samples_original, axis=0)
        term2_sum = 0
        N = n_samples
        for i in range(N):
            weight = 2*i - N + 1
            term2_sum += weight * samples_sorted[i]
            
        term2_final = term2_sum / (N * N)
        
        crps_per_element = term1 - term2_final
        crps = np.mean(crps_per_element)

        # Pinball loss
        def _pinball(y_true, y_pred_q, q):
            err = y_true - y_pred_q
            return np.mean(np.maximum(q * err, (q - 1.0) * err))

        pinball_metrics = {}
        for q in q_pinball:
            q_scaled = quantiles_dict[q].reshape(y_pred_params.shape[0], -1)
            _, q_original = self.transform.inverse_transform(X=None, y=q_scaled)
            pinball_metrics[f"Pinball_{int(q*100)}"] = _pinball(y_test, q_original, q)
        avg_pinball = np.mean(list(pinball_metrics.values()))
        pinball_metrics["Avg_Pinball"] = avg_pinball

        # NLL (Negative Log-Likelihood)
        nll = np.nan
        if hasattr(self.model.head, "log_pdf_np"):
            log_pdf = self.model.head.log_pdf_np(y_pred_params, y_test)
            nll = -np.mean(log_pdf)
        
        metrics = {
            "MAE": mae,
            "MSE": mse,
            "RMSE": np.sqrt(mse),
            "MAPE": mape,
            "R2": r2,
            "PICP": picp,
            "MPIW": mpiw,
            "PINAW": pinaw,
            "IntervalScore": interval_score,
            "CRPS": crps,
            "NLL": nll,
            **pinball_metrics,
        }
        
        return metrics

    def generate_forecasts(self, X_test_scaled: np.ndarray) -> np.ndarray:
        """
        generates mean forecasts in original scale
        """
        y_pred_params = self.model.keras_model.predict(X_test_scaled, verbose=0)
        means_scaled = self.model.head.mean(y_pred_params.reshape(-1, y_pred_params.shape[-1]))
        means_scaled = means_scaled.reshape(y_pred_params.shape[0], -1)
        _, means_original = self.transform.inverse_transform(X=None, y=means_scaled)
        return means_original
