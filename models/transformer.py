import matplotlib
import matplotlib.artist
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Optional, Dict, List

from config import ModelConfig, ExperimentConfig
from .heads import DistributionHead, JohnsonSUHead, GaussianHead, QuantileHead, MixtureGaussianHead, MixtureJohnsonSUHead

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


class HybridProbabilisticTransformer(ProbabilisticTransformer):
    # Extends ProbabilisticTransformer with OU process for residuals

    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        self.ou_process = OrnsteinUhlenbeckProcess()

    def fit_ou(self, X_train, y_train):
        #Fit the OU process on training residuals

        print("Fitting OU Process on training residuals: ")
        
        # Predict in large batches
        y_pred_dist = self.keras_model.predict(X_train, batch_size=2048, verbose=0)
        
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

