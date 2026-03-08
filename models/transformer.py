import matplotlib
import matplotlib.artist
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Optional, Dict, List

from config import ModelConfig, ExperimentConfig
from .heads import (
    DistributionHead,
    JohnsonSUHead,
    JohnsonSUFloorHead,
    TruncatedNormalHead,
    GaussianHead,
    QuantileHead,
    MixtureGaussianHead,
    MixtureJohnsonSUHead,
)

class ProbabilisticTransformer:
    # Modular Probabilistic Transformer

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.model_config = config.model_config
        self.data_config = config.data_config
        
        # Instantiate head based on config
        self.head: DistributionHead
        head_params = config.head_params or {}
        
        if config.head_type == "johnson_su":
            self.head = JohnsonSUHead()
        elif config.head_type == "gaussian":
            self.head = GaussianHead()
        elif config.head_type == "quantile":
            quantiles = head_params.get("quantiles", None)
            self.head = QuantileHead(quantiles=quantiles)
        elif config.head_type == "mixture_gaussian":
            n_components = head_params.get("n_components", 3)
            self.head = MixtureGaussianHead(n_components=n_components)
        elif config.head_type == "mixture_johnson_su":
            n_components = head_params.get("n_components", 3)
            self.head = MixtureJohnsonSUHead(n_components=n_components)
        elif config.head_type == "johnson_su_floor":
            floor_w = head_params.get("floor_penalty_weight", 0.1)
            asym_w = head_params.get("asymmetric_weight", 1.5)
            self.head = JohnsonSUFloorHead(floor_penalty_weight=floor_w, asymmetric_weight=asym_w)
        elif config.head_type == "truncated_normal":
            self.head = TruncatedNormalHead()
        else:
            raise ValueError(f"Unknown head type: {config.head_type}")
            
        self.keras_model: Optional[keras.Model] = None

    def _transformer_encoder(self, inputs, d_model, num_heads, ff_dim, dropout):
        # Multi-head self-attention
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout
        )(inputs, inputs)

        # Skip connection and layer normalization
        attention_output = layers.Dropout(dropout)(attention_output)
        x1 = layers.LayerNormalization(epsilon=1e-6)(inputs + attention_output)

        # Feed-forward network
        ff_output = layers.Dense(ff_dim, activation="relu")(x1)
        ff_output = layers.Dropout(dropout)(ff_output)
        ff_output = layers.Dense(d_model)(ff_output)

        # Skip connection and layer normalization
        ff_output = layers.Dropout(dropout)(ff_output)
        x2 = layers.LayerNormalization(epsilon=1e-6)(x1 + ff_output)
        return x2

    def _positional_encoding(self, length, depth):
        depth = depth / 2
        positions = np.arange(length)[:, np.newaxis]
        depths = np.arange(depth)[np.newaxis, :] / depth
        angle_rates = 1 / (10000**depths)
        angle_rads = positions * angle_rates
        pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)
        return tf.cast(pos_encoding, dtype=tf.float32)

    def build_model(self, n_features: int):
        # Builds the compiled Keras model

        conf = self.model_config
        
        inputs = keras.Input(shape=(self.data_config.input_window, n_features), name="input")
        
        # Projection if needed
        if n_features != conf.d_model:
            x = layers.Dense(conf.d_model)(inputs)
        else:
            x = inputs
            
        # Positional Encoding
        pos_enc = self._positional_encoding(self.data_config.input_window, conf.d_model)
        x = x + pos_enc
        
        # Encoder Blocks
        for _ in range(conf.num_layers):
            x = self._transformer_encoder(x, conf.d_model, conf.num_heads, conf.ff_dim, conf.dropout)
            
        # Global pooling en head
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(conf.ff_dim, activation="relu")(x)
        x = layers.Dropout(conf.dropout)(x)
        x = layers.Dense(conf.ff_dim // 2, activation="relu")(x)
        x = layers.Dropout(conf.dropout)(x)
        
        # Output Head
        outputs = self.head.build_output_layer(x, self.data_config.output_horizon)
        
        self.keras_model = keras.Model(inputs=inputs, outputs=outputs)
        
        # Compile
        self.keras_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config.training_config.learning_rate),
            loss=lambda y_true, y_pred: self.head.loss(y_true, y_pred)
        )
        return self.keras_model


class OrnsteinUhlenbeckProcess:
    # Ornstein-Uhlenbeck process om residuals te modellen

    def __init__(self, dt: float = 1.0):
        self.dt = dt
        self.k = 0.0
        self.mu = 0.0
        self.sigma = 0.0

    def fit(self, residuals: np.ndarray):
        """
        Estimate parameters using Maximum Likelihood Estimation (MLE)
        
        sigma^2 = (1/(N*dt)) * sum( (x_t - k*mu*dt + (k*dt - 1)*x_{t-1})^2 )
        k       = -(1/dt) * sum((x_t - x_{t-1})*(x_{t-1} - mu)) / sum((x_{t-1} - mu)^2)
        mu      = (1/(k*N*dt)) * sum( x_t + (k*dt - 1)*x_{t-1} )
        
        These implicit equations solved via fixed-point iteration
        """
        x = residuals
        N = len(x) - 1
        
        if N < 2:
            self.k = 0.0
            self.mu = 0.0
            self.sigma = 0.0
            return
        
        dt = self.dt
        x_t = x[1:]
        x_prev = x[:-1]
        dx = x_t - x_prev
        
        # Initialize mu with sample mean of the series
        mu = np.mean(x)
        
        # Fixed-point iteration to solve implicti MLE equations
        for iteration in range(100):
            mu_old = mu
            
            # Eq 7b: k = -(1/dt) * sum((dx)*(x_prev - mu)) / sum((x_prev - mu)^2)
            x_prev_centered = x_prev - mu
            denom_k = np.sum(x_prev_centered ** 2)
            
            if denom_k < 1e-12:
                # All residuals are essentially constant
                self.k = 0.0
                self.mu = mu
                self.sigma = np.std(dx) / np.sqrt(dt) if dt > 0 else 0.0
                return
            
            k = -(1.0 / dt) * np.sum(dx * x_prev_centered) / denom_k
            
            # Eq 7c: mu = (1/(k*N*dt)) * sum(x_t + (k*dt - 1)*x_prev)
            if abs(k) < 1e-12:
                # k approx 0 means no mean reversion thus process is near random walk
                mu = np.mean(x)
            else:
                mu = (1.0 / (k * N * dt)) * np.sum(x_t + (k * dt - 1.0) * x_prev)
            
            # Check convergence
            if abs(mu - mu_old) < 1e-10:
                break
        
        # Eq 7a: sigma^2 = (1/(N*dt)) * sum((x_t - k*mu*dt + (k*dt - 1)*x_prev)^2)
        residuals_mle = x_t - k * mu * dt + (k * dt - 1.0) * x_prev
        sigma2 = np.sum(residuals_mle ** 2) / (N * dt)
        sigma = np.sqrt(max(sigma2, 0.0))
        
        # Ensure k > 0 for mean-reversion (negative k is divergence)
        self.k = max(k, 0.0)
        self.mu = mu
        self.sigma = sigma

    def simulate(self, current_x: np.ndarray, steps: int, n_paths: int = 1) -> np.ndarray:
        # Simulate OU paths using the exact analytical transition distribution
        # X(t+dt) | X(t) ~ Normal(mu + (X(t)-mu)*exp(-k*dt), sigma^2/(2k)*(1-exp(-2k*dt)))

        if np.ndim(current_x) == 0:
            current_x = np.array([current_x])
        
        n_samples = len(current_x)
        paths = np.zeros((n_samples, steps, n_paths))
        
        # Initialize x_t: (n_samples, n_paths)
        x_t = np.tile(current_x[:, np.newaxis], (1, n_paths))
        
        k = self.k
        mu = self.mu
        dt = self.dt
        sigma = self.sigma
        
        # Precompute exact transition parameters
        if k > 1e-12:
            exp_neg_k_dt = np.exp(-k * dt)
            # Conditional variance: sigma^2/(2k) * (1 - exp(-2k*dt))
            cond_var = (sigma**2 / (2.0 * k)) * (1.0 - np.exp(-2.0 * k * dt))
            cond_std = np.sqrt(max(cond_var, 0.0))
        else:
            # k ~ 0: degenerate to random walk with drift
            exp_neg_k_dt = 1.0
            cond_std = sigma * np.sqrt(dt)
        
        for t in range(steps):
            # Exact conditional mean: mu + (x_t - mu) * exp(-k*dt)
            cond_mean = mu + (x_t - mu) * exp_neg_k_dt
            # Sample from exact conditional distribution
            noise = np.random.normal(0, 1, size=(n_samples, n_paths))
            x_t = cond_mean + cond_std * noise
            paths[:, t, :] = x_t
             
        return paths

    def mean_path(self, current_x: np.ndarray, steps: int) -> np.ndarray:
        # Compute the deterministic mean OU path (no noise) using exact formula
        # E[X(t)] = mu + (x_0 - mu) * exp(-k*t)

        if np.ndim(current_x) == 0:
            current_x = np.array([current_x])
        
        n_samples = len(current_x)
        path = np.zeros((n_samples, steps))
        
        k = self.k
        mu = self.mu
        dt = self.dt
        
        for t in range(steps):
            # Exact mean at time (t+1)*dt from start
            elapsed = (t + 1) * dt
            if k > 1e-12:
                path[:, t] = mu + (current_x - mu) * np.exp(-k * elapsed)
            else:
                path[:, t] = current_x  # No mean-reversion
        
        return path


class ReflectedOUProcess:
    # Ornstein-Uhlenbeck process with reflection at 0, Keeps process >= 0

    def __init__(self, dt: float = 1.0):
        self.dt = dt
        self.k = 0.0
        self.mu = 0.0
        self.sigma = 0.0

    def fit(self, residuals: np.ndarray):
        # Fit using same MLE as OU, ensure mu >= 0 for reflection to make sense
        ou = OrnsteinUhlenbeckProcess(dt=self.dt)
        ou.fit(residuals)
        self.k = ou.k
        self.mu = max(ou.mu, 0.01)  # positive mean reversion level
        self.sigma = ou.sigma

    def simulate(self, current_x: np.ndarray, steps: int, n_paths: int = 1) -> np.ndarray:
        # Simulate OU paths with reflection at 0
        if np.ndim(current_x) == 0:
            current_x = np.array([current_x])

        n_samples = len(current_x)
        paths = np.zeros((n_samples, steps, n_paths))

        x_t = np.tile(np.maximum(current_x[:, np.newaxis], 0.0), (1, n_paths))
        k, mu, dt, sigma = self.k, self.mu, self.dt, self.sigma

        if k > 1e-12:
            exp_neg_k_dt = np.exp(-k * dt)
            cond_var = (sigma**2 / (2.0 * k)) * (1.0 - np.exp(-2.0 * k * dt))
            cond_std = np.sqrt(max(cond_var, 0.0))
        else:
            exp_neg_k_dt = 1.0
            cond_std = sigma * np.sqrt(dt)

        for t in range(steps):
            cond_mean = mu + (x_t - mu) * exp_neg_k_dt
            noise = np.random.normal(0, 1, size=(n_samples, n_paths))
            x_t = np.maximum(cond_mean + cond_std * noise, 0.0)
            paths[:, t, :] = x_t

        return paths

    def mean_path(self, current_x: np.ndarray, steps: int) -> np.ndarray:
        # Deterministic mean path
        if np.ndim(current_x) == 0:
            current_x = np.array([current_x])
        n_samples = len(current_x)
        path = np.zeros((n_samples, steps))
        x_0 = np.maximum(current_x, 0.0)
        k, mu, dt = self.k, self.mu, self.dt
        for t in range(steps):
            elapsed = (t + 1) * dt
            if k > 1e-12:
                path[:, t] = mu + (x_0 - mu) * np.exp(-k * elapsed)
            else:
                path[:, t] = x_0
        return path


class CIRProcess:
    # Cox-Ingersoll-Ross (Feller) process: dX = k(theta - X)dt + sigma*sqrt(X)dW. X >= 0

    def __init__(self, dt: float = 1.0):
        self.dt = dt
        self.k = 0.0
        self.theta = 0.0
        self.sigma = 0.0

    def fit(self, residuals: np.ndarray):
        # Fit CIR to non-negative residuals

        x = np.maximum(residuals, 1e-6)
        N = len(x) - 1
        if N < 2:
            self.k = 0.1
            self.theta = float(max(np.mean(x), 0.01))
            self.sigma = float(max(np.std(x) * 0.5, 1e-6))
            return

        x_t, x_prev = x[1:], x[:-1]
        dt = self.dt

        theta = np.mean(x)
        k = 0.5
        for _ in range(50):
            k_old = k
            drift = (theta - x_prev) * dt
            denom = np.mean(drift ** 2) + 1e-10
            k = np.mean((x_t - x_prev) * drift) / denom
            k = np.clip(k, 1e-6, 10.0)
            if abs(k - k_old) < 1e-6:
                break

        residuals_sq = (x_t - x_prev - k * (theta - x_prev) * dt) ** 2
        sigma2 = np.mean(residuals_sq) / (dt * np.mean(x_prev) + 1e-10)
        sigma = np.sqrt(max(sigma2, 1e-10))

        self.k = float(k)
        self.theta = float(max(theta, 0.01))
        self.sigma = float(sigma)

    def simulate(self, current_x: np.ndarray, steps: int, n_paths: int = 1) -> np.ndarray:
        """Euler-Maruyama for CIR. Use absorption at 0 for negative values."""
        if np.ndim(current_x) == 0:
            current_x = np.array([current_x])

        n_samples = len(current_x)
        paths = np.zeros((n_samples, steps, n_paths))

        x_t = np.tile(np.maximum(current_x[:, np.newaxis], 1e-8), (1, n_paths))
        k, theta, dt, sigma = self.k, self.theta, self.dt, self.sigma

        for t in range(steps):
            sqrt_x = np.sqrt(np.maximum(x_t, 1e-12))
            drift = k * (theta - x_t) * dt
            diff = sigma * sqrt_x * np.sqrt(dt)
            noise = np.random.normal(0, 1, size=(n_samples, n_paths))
            x_t = np.maximum(x_t + drift + diff * noise, 0.0)
            paths[:, t, :] = x_t

        return paths

    def mean_path(self, current_x: np.ndarray, steps: int) -> np.ndarray:
        # E[CIR] = theta + (x_0 - theta)*exp(-k*t)
        if np.ndim(current_x) == 0:
            current_x = np.array([current_x])

        n_samples = len(current_x)
        path = np.zeros((n_samples, steps))
        k, theta = self.k, self.theta
        x_0 = np.maximum(current_x, 0.0)

        for t in range(steps):
            elapsed = (t + 1) * self.dt
            if k > 1e-12:
                path[:, t] = theta + (x_0 - theta) * np.exp(-k * elapsed)
            else:
                path[:, t] = x_0
        return path


class HybridProbabilisticTransformer(ProbabilisticTransformer):
    # Extends ProbabilisticTransformer with OU process for residuals

    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        self.ou_process = OrnsteinUhlenbeckProcess()

    def fit_ou(self, X_train, y_train):
        #Fit the OU process on training residuals

        print("Fitting OU Process on training residuals: ")
        
        # Predict in small batches to avoid GPU OOM (especially after training)
        y_pred_dist = self.keras_model.predict(X_train, batch_size=256, verbose=0)
        
        if hasattr(self.head, "mean"):
            flat_params = y_pred_dist.reshape(-1, y_pred_dist.shape[-1])
            flat_means = self.head.mean(flat_params)
            y_pred_means = flat_means.reshape(y_train.shape)
            
            # Compute residuals per window: (n_windows, horizon)
            all_residuals = y_train - y_pred_means
            n_windows, horizon = all_residuals.shape
            
            # Fit OU on each window's temporal contiguous residual series and collect per-window parameter estimates
            k_estimates = []
            mu_estimates = []
            sigma_estimates = []
            
            temp_ou = OrnsteinUhlenbeckProcess(dt=self.ou_process.dt)
            for i in range(n_windows):
                window_resid = all_residuals[i, :]
                if len(window_resid) < 3:
                    continue
                temp_ou.fit(window_resid)
                # Only use windows where k > 0 (valid mean-reversion)
                if temp_ou.k > 1e-6 and temp_ou.sigma > 1e-6:
                    k_estimates.append(temp_ou.k)
                    mu_estimates.append(temp_ou.mu)
                    sigma_estimates.append(temp_ou.sigma)
            
            if len(k_estimates) > 10:
                # Use median for robustness (outliers)
                self.ou_process.k = float(np.median(k_estimates))
                self.ou_process.mu = float(np.median(mu_estimates))
                self.ou_process.sigma = float(np.median(sigma_estimates))
            else:
                # Fallback: fit on all residuals concatenated but only within each window (skip cross-window boundaries)
                concat_resids = all_residuals.flatten()
                self.ou_process.fit(concat_resids)
            
            # Store residual statistics for last-residual computation
            self._train_residuals = all_residuals
            
            print(f"OU Params: k={self.ou_process.k:.4f}, mu={self.ou_process.mu:.4f}, sigma={self.ou_process.sigma:.4f}")
            print(f"  (Estimated from {len(k_estimates)} valid windows out of {n_windows})")
        

    def compute_last_residual(self, X) -> np.ndarray:
        # Compute the starting residual for OU simulation

        # Use the last K time steps of the input to estimate recent residual trend
        K = min(5, X.shape[1])
        
        recent_observed = np.mean(X[:, -K:, 0], axis=1)
        
        y_pred_dist = self.keras_model.predict(X, verbose=0)
        flat_params = y_pred_dist.reshape(-1, y_pred_dist.shape[-1])
        y_pred_means = self.head.mean(flat_params).reshape(X.shape[0], -1)
        
        K_pred = min(3, y_pred_means.shape[1])
        pred_baseline = np.mean(y_pred_means[:, :K_pred], axis=1)
        
        last_residual = recent_observed - pred_baseline
        
        return last_residual

    def predict_hybrid(self, X, last_residuals: np.ndarray = None):
        # Predict using transformer + OU (mean path)

        # Transformer forecast (mean)
        y_pred_dist = self.keras_model.predict(X, verbose=0)
        flat_params = y_pred_dist.reshape(-1, y_pred_dist.shape[-1])
        y_pred_means = self.head.mean(flat_params).reshape(X.shape[0], -1)
        
        if last_residuals is None:
            last_residuals = self.compute_last_residual(X)
            
        # Add OU mean path
        horizon = y_pred_means.shape[1]
        ou_correction = self.ou_process.mean_path(last_residuals, steps=horizon)
            
        return y_pred_means + ou_correction

    def sample_hybrid(self, X, n_samples: int = 100, last_residuals: np.ndarray = None) -> np.ndarray:
        # Generate probabilistic samples from hybrid model

        # Sample from transformer head
        y_pred_dist = self.keras_model.predict(X, verbose=0)
        flat_params = y_pred_dist.reshape(-1, y_pred_dist.shape[-1])
        
        tf_samples_flat = self.head.sample(flat_params, n_samples)
        
        transformer_samples = tf_samples_flat.T.reshape(X.shape[0], -1, n_samples)
        
        # Sample from OU process
        if last_residuals is None:
            last_residuals = self.compute_last_residual(X)
            
        horizon = transformer_samples.shape[1]
        
        ou_paths = self.ou_process.simulate(last_residuals, steps=horizon, n_paths=n_samples)
        
        return transformer_samples + ou_paths

    def quantiles_hybrid(self, X, q_list: List[float], n_samples: int = 1000, last_residuals: np.ndarray = None) -> Dict[float, np.ndarray]:
        # Compute quantiles via Monte Carlo sampling

        samples = self.sample_hybrid(X, n_samples, last_residuals)
        
        results = {}
        for q in q_list:
            # Quantile over the last axis (n_samples)
            results[q] = np.quantile(samples, q, axis=-1)
            
        return results


class HybridProbabilisticTransformerReflectedOU(HybridProbabilisticTransformer):
    # Hybrid Transformer + Reflected OU (residuals reflected at 0)

    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        self.ou_process = ReflectedOUProcess(dt=1.0)

    def fit_ou(self, X_train, y_train):
        super().fit_ou(X_train, y_train)
        self.ou_process.mu = max(self.ou_process.mu, 0.01)


class HybridProbabilisticTransformerCIR(HybridProbabilisticTransformer):
    # Hybrid Transformer + CIR process for non-negative residuals

    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        self.ou_process = CIRProcess(dt=1.0)

    def fit_ou(self, X_train, y_train):
        # Fit CIR on max(residuals, 0) + epsilon
        print("Fitting CIR process on training residuals")
        y_pred_dist = self.keras_model.predict(X_train, batch_size=256, verbose=0)

        if hasattr(self.head, "mean"):
            flat_params = y_pred_dist.reshape(-1, y_pred_dist.shape[-1])
            flat_means = self.head.mean(flat_params)
            y_pred_means = flat_means.reshape(y_train.shape)
            all_residuals = y_train - y_pred_means
            n_windows, horizon = all_residuals.shape

            k_est, theta_est, sigma_est = [], [], []
            temp_cir = CIRProcess(dt=self.ou_process.dt)
            for i in range(n_windows):
                r = np.maximum(all_residuals[i, :], 1e-6)
                if len(r) < 3:
                    continue
                temp_cir.fit(r)
                if temp_cir.k > 1e-6 and temp_cir.sigma > 1e-6:
                    k_est.append(temp_cir.k)
                    theta_est.append(temp_cir.theta)
                    sigma_est.append(temp_cir.sigma)

            if len(k_est) > 10:
                self.ou_process.k = float(np.median(k_est))
                self.ou_process.theta = float(np.median(theta_est))
                self.ou_process.sigma = float(np.median(sigma_est))
            else:
                concat = np.maximum(all_residuals.flatten(), 1e-6)
                self.ou_process.fit(concat)

            self._train_residuals = all_residuals
            print(f"CIR Params: k={self.ou_process.k:.4f}, theta={self.ou_process.theta:.4f}, sigma={self.ou_process.sigma:.4f}")
        else:
            super().fit_ou(X_train, y_train)

    def compute_last_residual(self, X) -> np.ndarray:
        r = super().compute_last_residual(X)
        return np.maximum(r, 0.0)

    def predict_hybrid(self, X, last_residuals: np.ndarray = None):
        return super().predict_hybrid(X, last_residuals)

    def sample_hybrid(self, X, n_samples: int = 100, last_residuals: np.ndarray = None):
        return super().sample_hybrid(X, n_samples, last_residuals)


class HybridProbabilisticTransformerPostHocFloor(HybridProbabilisticTransformer):
    # Hybrid Transformer + OU with floor: max(pred, 0)

    def predict_hybrid(self, X, last_residuals: np.ndarray = None):
        pred = super().predict_hybrid(X, last_residuals)
        return np.maximum(pred, 0.0)

    def sample_hybrid(self, X, n_samples: int = 100, last_residuals: np.ndarray = None):
        samples = super().sample_hybrid(X, n_samples, last_residuals)
        return np.maximum(samples, 0.0)


# OU + Compound Poisson (Levy jump-diffusion)
class OUJumpProcess:
    # Ornstein-Uhlenbeck + compound Poisson jumps (Levy process)
    # dX = OU_dynamics + dJ, J = sum of N(t) jumps. Allows negative prices
    def __init__(self, dt: float = 1.0):
        self.dt = dt
        self.k = 0.0
        self.mu = 0.0
        self.sigma = 0.0
        self.lambda_jump = 0.0  # jump intensity (arrivals per unit time)
        self.jump_mean = 0.0   # mean jump size (can be positive for spike bias)
        self.jump_std = 0.0    # std of jump size

    def fit(self, residuals: np.ndarray):
        ou = OrnsteinUhlenbeckProcess(dt=self.dt)
        ou.fit(residuals)
        self.k = ou.k
        self.mu = ou.mu
        self.sigma = ou.sigma
        # Estimate jump component from tail of residuals
        abs_resid = np.abs(residuals - np.median(residuals))
        threshold = np.percentile(abs_resid, 95)
        jumps = residuals[np.abs(residuals - np.median(residuals)) > threshold] - np.median(residuals)
        if len(jumps) > 5:
            self.lambda_jump = min(len(jumps) / (len(residuals) * self.dt + 1e-10), 2.0)
            self.jump_mean = float(np.mean(jumps))
            self.jump_std = float(max(np.std(jumps), 1e-6))
        else:
            self.lambda_jump = 0.0
            self.jump_mean = 0.0
            self.jump_std = 0.0

    def simulate(self, current_x: np.ndarray, steps: int, n_paths: int = 1) -> np.ndarray:
        if np.ndim(current_x) == 0:
            current_x = np.array([current_x])
        n_samples = len(current_x)
        paths = np.zeros((n_samples, steps, n_paths))
        x_t = np.tile(current_x[:, np.newaxis], (1, n_paths))
        k, mu, dt, sigma = self.k, self.mu, self.dt, self.sigma
        lam, jmean, jstd = self.lambda_jump, self.jump_mean, self.jump_std

        if k > 1e-12:
            exp_neg_k_dt = np.exp(-k * dt)
            cond_var = (sigma**2 / (2.0 * k)) * (1.0 - np.exp(-2.0 * k * dt))
            cond_std = np.sqrt(max(cond_var, 0.0))
        else:
            exp_neg_k_dt = 1.0
            cond_std = sigma * np.sqrt(dt)

        for t in range(steps):
            cond_mean = mu + (x_t - mu) * exp_neg_k_dt
            noise = np.random.normal(0, 1, size=(n_samples, n_paths))
            x_t = cond_mean + cond_std * noise
            # Add compound Poisson jumps
            n_jumps = np.random.poisson(lam * dt, size=(n_samples, n_paths))
            for _ in range(int(np.max(n_jumps))):
                mask = n_jumps > 0
                jump_sizes = np.random.normal(jmean, max(jstd, 1e-8), size=(n_samples, n_paths))
                x_t = np.where(mask, x_t + jump_sizes, x_t)
                n_jumps = np.maximum(n_jumps - 1, 0)
            paths[:, t, :] = x_t
        return paths

    def mean_path(self, current_x: np.ndarray, steps: int) -> np.ndarray:
        if np.ndim(current_x) == 0:
            current_x = np.array([current_x])
        n_samples = len(current_x)
        path = np.zeros((n_samples, steps))
        k, mu, dt, jmean, lam = self.k, self.mu, self.dt, self.jump_mean, self.lambda_jump
        for t in range(steps):
            elapsed = (t + 1) * dt
            if k > 1e-12:
                path[:, t] = mu + (current_x - mu) * np.exp(-k * elapsed)
            else:
                path[:, t] = current_x
            path[:, t] += lam * dt * jmean  # mean jump contribution
        return path


class HybridProbabilisticTransformerOUJump(HybridProbabilisticTransformer):
    # Hybrid Transformer + OU with compound Poisson jumps (Levy process

    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        self.ou_process = OUJumpProcess(dt=1.0)


# Soft barrier OU (prices repelled from below zero, but can go negative)
class SoftBarrierOUProcess:
    #OU with soft floor: when x < 0, mean-revert toward 0 when x >= 0, toward mu
    # Creates tendency to cluster just above zero without hard truncation

    def __init__(self, dt: float = 1.0):
        self.dt = dt
        self.k = 0.0
        self.mu = 0.0
        self.sigma = 0.0

    def fit(self, residuals: np.ndarray):
        ou = OrnsteinUhlenbeckProcess(dt=self.dt)
        ou.fit(residuals)
        self.k = ou.k
        self.mu = ou.mu
        self.sigma = ou.sigma

    def _target(self, x: np.ndarray) -> np.ndarray:
        return np.where(x < 0, 0.0, self.mu)

    def simulate(self, current_x: np.ndarray, steps: int, n_paths: int = 1) -> np.ndarray:
        if np.ndim(current_x) == 0:
            current_x = np.array([current_x])
        n_samples = len(current_x)
        paths = np.zeros((n_samples, steps, n_paths))
        x_t = np.tile(current_x[:, np.newaxis], (1, n_paths))
        k, dt, sigma = self.k, self.dt, self.sigma

        if k > 1e-12:
            exp_neg_k_dt = np.exp(-k * dt)
            cond_var = (sigma**2 / (2.0 * k)) * (1.0 - np.exp(-2.0 * k * dt))
            cond_std = np.sqrt(max(cond_var, 0.0))
        else:
            exp_neg_k_dt = 1.0
            cond_std = sigma * np.sqrt(dt)

        for t in range(steps):
            target = self._target(x_t)
            cond_mean = target + (x_t - target) * exp_neg_k_dt
            noise = np.random.normal(0, 1, size=(n_samples, n_paths))
            x_t = cond_mean + cond_std * noise
            paths[:, t, :] = x_t
        return paths

    def mean_path(self, current_x: np.ndarray, steps: int) -> np.ndarray:
        if np.ndim(current_x) == 0:
            current_x = np.array([current_x])
        n_samples = len(current_x)
        path = np.zeros((n_samples, steps))
        k, mu, dt = self.k, self.mu, self.dt
        x = current_x.copy()
        for t in range(steps):
            elapsed = (t + 1) * dt
            target = np.where(x < 0, 0.0, mu)
            if k > 1e-12:
                x = target + (x - target) * np.exp(-k * elapsed)
            path[:, t] = x
        return path


class HybridProbabilisticTransformerSoftBarrierOU(HybridProbabilisticTransformer):
    # Hybrid Transformer + OU with soft floor near zero (no hard truncation)

    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        self.ou_process = SoftBarrierOUProcess(dt=1.0)

    def fit_ou(self, X_train, y_train):
        super().fit_ou(X_train, y_train)  # uses parent to get k, mu, sigma


# Asymmetric jump-diffusion (larger upward spikes than downward)
class AsymmetricJumpProcess:
    # OU + two-sided compound Poisson: positive jumps (spikes) larger than negative
    # Captures asymmetric volatility: upward spikes dominate

    def __init__(self, dt: float = 1.0):
        self.dt = dt
        self.k = 0.0
        self.mu = 0.0
        self.sigma = 0.0
        self.lam_up = 0.0
        self.lam_down = 0.0
        self.theta_up = 10.0   # mean positive jump (spikes)
        self.theta_down = 5.0  # mean negative jump (smaller)

    def fit(self, residuals: np.ndarray):
        ou = OrnsteinUhlenbeckProcess(dt=self.dt)
        ou.fit(residuals)
        self.k = ou.k
        self.mu = ou.mu
        self.sigma = ou.sigma
        med = np.median(residuals)
        pos_jumps = residuals[residuals > med + np.std(residuals) * 0.5] - med
        neg_jumps = med - residuals[residuals < med - np.std(residuals) * 0.5]
        n = len(residuals) * self.dt + 1e-10
        if len(pos_jumps) > 3:
            self.lam_up = min(len(pos_jumps) / n, 2.0)
            self.theta_up = float(max(np.mean(pos_jumps), 1.0))
        if len(neg_jumps) > 3:
            self.lam_down = min(len(neg_jumps) / n, 2.0)
            self.theta_down = float(max(np.mean(neg_jumps), 0.5))

    def simulate(self, current_x: np.ndarray, steps: int, n_paths: int = 1) -> np.ndarray:
        if np.ndim(current_x) == 0:
            current_x = np.array([current_x])
        n_samples = len(current_x)
        paths = np.zeros((n_samples, steps, n_paths))
        x_t = np.tile(current_x[:, np.newaxis], (1, n_paths))
        k, mu, dt, sigma = self.k, self.mu, self.dt, self.sigma
        lam_u, lam_d = self.lam_up, self.lam_down
        th_u, th_d = self.theta_up, self.theta_down

        if k > 1e-12:
            exp_neg_k_dt = np.exp(-k * dt)
            cond_var = (sigma**2 / (2.0 * k)) * (1.0 - np.exp(-2.0 * k * dt))
            cond_std = np.sqrt(max(cond_var, 0.0))
        else:
            exp_neg_k_dt = 1.0
            cond_std = sigma * np.sqrt(dt)

        for t in range(steps):
            cond_mean = mu + (x_t - mu) * exp_neg_k_dt
            noise = np.random.normal(0, 1, size=(n_samples, n_paths))
            x_t = cond_mean + cond_std * noise
            n_up = np.random.poisson(lam_u * dt, size=(n_samples, n_paths))
            n_dn = np.random.poisson(lam_d * dt, size=(n_samples, n_paths))
            for _ in range(int(max(np.max(n_up), np.max(n_dn)))):
                up_mask = n_up > 0
                dn_mask = n_dn > 0
                x_t = np.where(up_mask, x_t + np.random.exponential(th_u, size=(n_samples, n_paths)), x_t)
                x_t = np.where(dn_mask, x_t - np.random.exponential(th_d, size=(n_samples, n_paths)), x_t)
                n_up, n_dn = np.maximum(n_up - 1, 0), np.maximum(n_dn - 1, 0)
            paths[:, t, :] = x_t
        return paths

    def mean_path(self, current_x: np.ndarray, steps: int) -> np.ndarray:
        if np.ndim(current_x) == 0:
            current_x = np.array([current_x])
        n_samples = len(current_x)
        path = np.zeros((n_samples, steps))
        k, mu, dt, lam_u, lam_d, th_u, th_d = (
            self.k, self.mu, self.dt, self.lam_up, self.lam_down,
            self.theta_up, self.theta_down
        )
        for t in range(steps):
            elapsed = (t + 1) * dt
            if k > 1e-12:
                path[:, t] = mu + (current_x - mu) * np.exp(-k * elapsed)
            else:
                path[:, t] = current_x
            path[:, t] += (lam_u * th_u - lam_d * th_d) * dt
            current_x = path[:, t]
        return path


class HybridProbabilisticTransformerAsymmetricJump(HybridProbabilisticTransformer):
    # Hybrid Transformer + OU with asymmetric jumps (larger upward spikes)

    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        self.ou_process = AsymmetricJumpProcess(dt=1.0)

    def fit_ou(self, X_train, y_train):
        print("Fitting Asymmetric Jump process on training residuals...")
        y_pred_dist = self.keras_model.predict(X_train, batch_size=256, verbose=0)
        flat_params = y_pred_dist.reshape(-1, y_pred_dist.shape[-1])
        y_pred_means = self.head.mean(flat_params).reshape(y_train.shape)
        all_residuals = y_train - y_pred_means
        self.ou_process.fit(all_residuals.flatten())
        self._train_residuals = all_residuals
        print(f"  OU: k={self.ou_process.k:.4f}, mu={self.ou_process.mu:.4f}, sigma={self.ou_process.sigma:.4f}")
        print(f"  Jumps: lam_up={self.ou_process.lam_up:.4f} (θ_up={self.ou_process.theta_up:.2f}), "
              f"lam_down={self.ou_process.lam_down:.4f} (θ_down={self.ou_process.theta_down:.2f})")


# Hour-specific OU (different variance per hour of day)
class HourlyOUProcess:
    # 24 OU processes, one per hour of day. Captures higher variance at peak hours

    def __init__(self, dt: float = 1.0):
        self.dt = dt
        self.ou_per_hour = [OrnsteinUhlenbeckProcess(dt=dt) for _ in range(24)]

    def fit(self, residuals: np.ndarray, hour_indices: np.ndarray):
        # residuals: (n_windows, horizon), hour_indices: (n_windows, horizon) with values 0-23
        for h in range(24):
            mask = hour_indices == h
            r = residuals[mask]
            if len(r) > 10:
                self.ou_per_hour[h].fit(r)

    def simulate(self, current_x: np.ndarray, steps: int, n_paths: int,
                 hour_0: int) -> np.ndarray:
        # hour_0 = hour of first forecast step (0-23)
        if np.ndim(current_x) == 0:
            current_x = np.array([current_x])
        n_samples = len(current_x)
        paths = np.zeros((n_samples, steps, n_paths))
        x_t = np.tile(current_x[:, np.newaxis], (1, n_paths))
        for t in range(steps):
            h = (hour_0 + t) % 24
            ou = self.ou_per_hour[h]
            # Flatten to simulate each (sample, path) independently
            current_flat = x_t.flatten()
            step_sim = ou.simulate(current_flat, 1, 1)
            x_t = step_sim[:, 0, 0].reshape(n_samples, n_paths)
            paths[:, t, :] = x_t
        return paths

    def mean_path(self, current_x: np.ndarray, steps: int, hour_0: int) -> np.ndarray:
        if np.ndim(current_x) == 0:
            current_x = np.array([current_x])
        n_samples = len(current_x)
        path = np.zeros((n_samples, steps))
        x = current_x.copy()
        for t in range(steps):
            h = (hour_0 + t) % 24
            ou = self.ou_per_hour[h]
            step_mean = ou.mean_path(x, 1)
            x = step_mean[:, 0]
            path[:, t] = x
        return path


class HybridProbabilisticTransformerHourlyOU(HybridProbabilisticTransformer):
    # Hybrid Transformer + OU with different parameters per hour of day

    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        self.ou_process = HourlyOUProcess(dt=1.0)
        self._hour_col_idx = (config.head_params or {}).get("hour_col_idx", 1)

    def fit_ou(self, X_train, y_train):
        print("Fitting Hourly OU (24 processes)...")
        y_pred_dist = self.keras_model.predict(X_train, batch_size=256, verbose=0)
        flat_params = y_pred_dist.reshape(-1, y_pred_dist.shape[-1])
        y_pred_means = self.head.mean(flat_params).reshape(y_train.shape)
        all_residuals = y_train - y_pred_means
        n_windows, horizon = all_residuals.shape

        # Get hour for each (window, step): last input hour + 1 + step
        last_hours = X_train[:, -1, self._hour_col_idx].astype(int) % 24
        hour_indices = np.zeros_like(all_residuals, dtype=int)
        for t in range(horizon):
            hour_indices[:, t] = (last_hours + 1 + t) % 24

        self.ou_process.fit(all_residuals, hour_indices)
        self._train_residuals = all_residuals
        print("  Fitted 24 hourly OU processes.")

    def compute_last_residual(self, X) -> np.ndarray:
        return super().compute_last_residual(X)

    def predict_hybrid(self, X, last_residuals: np.ndarray = None):
        y_pred_dist = self.keras_model.predict(X, verbose=0)
        flat_params = y_pred_dist.reshape(-1, y_pred_dist.shape[-1])
        y_pred_means = self.head.mean(flat_params).reshape(X.shape[0], -1)
        if last_residuals is None:
            last_residuals = self.compute_last_residual(X)
        horizon = y_pred_means.shape[1]
        last_hours = (X[:, -1, self._hour_col_idx].astype(int) % 24)
        ou_correction = np.zeros_like(y_pred_means)
        for i in range(X.shape[0]):
            ou_correction[i] = self.ou_process.mean_path(
                np.array([last_residuals[i]]), horizon, int(last_hours[i])
            ).flatten()
        return y_pred_means + ou_correction

    def sample_hybrid(self, X, n_samples: int = 100, last_residuals: np.ndarray = None):
        y_pred_dist = self.keras_model.predict(X, verbose=0)
        flat_params = y_pred_dist.reshape(-1, y_pred_dist.shape[-1])
        tf_samples = self.head.sample(flat_params, n_samples)
        transformer_samples = tf_samples.T.reshape(X.shape[0], -1, n_samples)
        if last_residuals is None:
            last_residuals = self.compute_last_residual(X)
        horizon = transformer_samples.shape[1]
        last_hours = (X[:, -1, self._hour_col_idx].astype(int) % 24)
        ou_paths = np.zeros((X.shape[0], horizon, n_samples))
        for i in range(X.shape[0]):
            ou_paths[i] = self.ou_process.simulate(
                np.array([last_residuals[i]]), horizon, n_samples, int(last_hours[i])
            )
        return transformer_samples + ou_paths

