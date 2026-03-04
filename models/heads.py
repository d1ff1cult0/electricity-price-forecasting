from abc import ABC, abstractmethod
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from scipy.stats import johnsonsu, norm, truncnorm
from typing import Dict, Any, List, Optional
from scipy.interpolate import interp1d


def _np_softplus(x):
    return np.where(x > 20, x, np.log1p(np.exp(np.clip(x, -500, 20))))


def _safe_positive_params(params):
    gamma = np.nan_to_num(params[:, 0], nan=0.0)
    delta = np.clip(_np_softplus(np.nan_to_num(params[:, 1], nan=1.0)) + 1e-6, 1e-6, 1e6)
    xi = np.nan_to_num(params[:, 2], nan=0.0)
    lam = np.clip(_np_softplus(np.nan_to_num(params[:, 3], nan=1.0)) + 1e-6, 1e-6, 1e6)
    return gamma, delta, xi, lam

class DistributionHead(ABC):
    # Abstract base class for probabilistic output heads

    @abstractmethod
    def build_output_layer(self, x: tf.Tensor, horizon: int) -> tf.Tensor:
        # Builds the Keras layers to go from last layer to distribution parameters
        pass

    @abstractmethod
    def loss(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        # Computes the loss function
        pass

    @abstractmethod
    def sample(self, params: np.ndarray, n_samples: int) -> np.ndarray:
        # Samples from the distribution
        pass
    
    @abstractmethod
    def mean(self, params: np.ndarray) -> np.ndarray:
        #Returns the mean forecast
        pass
        
    @abstractmethod
    def quantiles(self, params: np.ndarray, q_list: List[float]) -> Dict[float, np.ndarray]:
        # Returns dictionary of quantile forecasts
        pass
        

class JohnsonSUHead(DistributionHead):
    # ohnson SU distribution head
    def __init__(self):
        self.param_names = ["gamma", "delta", "xi", "lambda"]
        
    def build_output_layer(self, x: tf.Tensor, horizon: int) -> tf.Tensor:
        total_params = 4
        param_units = horizon * total_params
        x = layers.Dense(param_units, activation="linear", name="jsu_params")(x)
        return layers.Reshape((horizon, total_params), name="jsu_reshaped")(x)
        
    def _split_params(self, y_pred: tf.Tensor):
        gamma = y_pred[..., 0]
        delta = tf.nn.softplus(y_pred[..., 1]) + 1e-6
        xi = y_pred[..., 2]
        lam = tf.nn.softplus(y_pred[..., 3]) + 1e-6
        return gamma, delta, xi, lam

    def loss(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        if len(y_true.shape) == 3:
            y_true = tf.squeeze(y_true, axis=-1)
            
        gamma, delta, xi, lam = self._split_params(y_pred)
        
        z = (y_true - xi) / lam
        asinh_z = tf.asinh(z)
        log_pdf = (
            tf.math.log(delta)
            - tf.math.log(lam)
            - 0.5 * tf.math.log(2.0 * np.pi)
            - 0.5 * tf.square(gamma + delta * asinh_z)
            - 0.5 * tf.math.log(1.0 + tf.square(z) + 1e-8)
        )
        return -tf.reduce_mean(log_pdf)
        
    def sample(self, params: np.ndarray, n_samples: int) -> np.ndarray:
        horizon = params.shape[0]
        samples = np.zeros((n_samples, horizon))
        
        gamma, delta, xi, lam = _safe_positive_params(params)
        
        for t in range(horizon):
            dist = johnsonsu(gamma[t], delta[t], loc=xi[t], scale=lam[t])
            samples[:, t] = dist.rvs(size=n_samples)
        return samples

    def mean(self, params: np.ndarray) -> np.ndarray:
        horizon = params.shape[0]
        means = np.zeros(horizon)
        
        gamma, delta, xi, lam = _safe_positive_params(params)
        
        for t in range(horizon):
            dist = johnsonsu(gamma[t], delta[t], loc=xi[t], scale=lam[t])
            means[t] = dist.mean()
        return means

    def quantiles(self, params: np.ndarray, q_list: List[float]) -> Dict[float, np.ndarray]:
        horizon = params.shape[0]
        results = {}
        
        gamma, delta, xi, lam = _safe_positive_params(params)
        
        for q in q_list:
            q_vals = np.zeros(horizon)
            for t in range(horizon):
                dist = johnsonsu(gamma[t], delta[t], loc=xi[t], scale=lam[t])
                q_vals[t] = dist.ppf(q)
            results[q] = q_vals
        return results

    def log_pdf_np(self, params: np.ndarray, y: np.ndarray) -> np.ndarray:
        # Compute log PDF for NLL evaluation
        params_flat = params.reshape(-1, params.shape[-1])
        gamma, delta, xi, lam = _safe_positive_params(params_flat)
        y_flat = y.reshape(-1)
        z = (y_flat - xi) / (lam + 1e-10)
        asinh_z = np.arcsinh(np.clip(z, -1e10, 1e10))
        log_pdf = (
            np.log(delta + 1e-10)
            - np.log(lam + 1e-10)
            - 0.5 * np.log(2.0 * np.pi)
            - 0.5 * (gamma + delta * asinh_z) ** 2
            - 0.5 * np.log(1.0 + z ** 2 + 1e-10)
        )
        return log_pdf.reshape(y.shape)


class GaussianHead(DistributionHead):
    # Gaussian distribution head
    def build_output_layer(self, x: tf.Tensor, horizon: int) -> tf.Tensor:
        total_params = 2
        param_units = horizon * total_params
        x = layers.Dense(param_units, activation="linear", name="gaussian_params")(x)
        return layers.Reshape((horizon, total_params), name="gaussian_reshaped")(x)
        
    def _split_params(self, y_pred: tf.Tensor):
        mu = y_pred[..., 0]
        sigma = tf.nn.softplus(y_pred[..., 1]) + 1e-6
        return mu, sigma

    def loss(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        if len(y_true.shape) == 3:
            y_true = tf.squeeze(y_true, axis=-1)
            
        mu, sigma = self._split_params(y_pred)
        
        z = (y_true - mu) / sigma
        log_pdf = (
            -0.5 * tf.math.log(2.0 * np.pi)
            - tf.math.log(sigma)
            - 0.5 * tf.square(z)
        )
        return -tf.reduce_mean(log_pdf)
        
    def sample(self, params: np.ndarray, n_samples: int) -> np.ndarray:
        horizon = params.shape[0]
        samples = np.zeros((n_samples, horizon))
        
        mu = params[:, 0]
        sigma = np.log(np.exp(params[:, 1]) + 1) + 1e-6
        
        for t in range(horizon):
            samples[:, t] = np.random.normal(loc=mu[t], scale=sigma[t], size=n_samples)
        return samples

    def mean(self, params: np.ndarray) -> np.ndarray:
        return params[:, 0]

    def quantiles(self, params: np.ndarray, q_list: List[float]) -> Dict[float, np.ndarray]:
        horizon = params.shape[0]
        results = {}
        
        mu = params[:, 0]
        sigma = np.log(np.exp(params[:, 1]) + 1) + 1e-6
        
        for q in q_list:
            results[q] = norm.ppf(q, loc=mu, scale=sigma)
        return results

    def log_pdf_np(self, params: np.ndarray, y: np.ndarray) -> np.ndarray:
        # Compute log PDF for NLL evaluation
        mu = params[..., 0]
        sigma = np.log(np.exp(np.clip(params[..., 1], -20, 20)) + 1) + 1e-6
        z = (y - mu) / (sigma + 1e-10)
        log_pdf = (
            -0.5 * np.log(2.0 * np.pi)
            - np.log(sigma + 1e-10)
            - 0.5 * z ** 2
        )
        return log_pdf


class QuantileHead(DistributionHead):
    # Non-parametric quantile regression head (Outputs a set of quantiles directly which minimized pinball loss)

    def __init__(self, quantiles: List[float] = None):
        self.quantiles_list = quantiles or [0.1, 0.25, 0.5, 0.75, 0.9]
        self.quantiles_list.sort()
        
    def build_output_layer(self, x: tf.Tensor, horizon: int) -> tf.Tensor:
        n_quantiles = len(self.quantiles_list)
        total_outputs = horizon * n_quantiles
        
        x = layers.Dense(total_outputs, activation="linear", name="quantile_params")(x)
        outputs = layers.Reshape((horizon, n_quantiles), name="quantile_reshaped")(x)
        return outputs

    def loss(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        if len(y_true.shape) == 2:
            y_true = tf.expand_dims(y_true, axis=-1)
        
        q_tensor = tf.constant(self.quantiles_list, dtype=tf.float32)
        q_tensor = tf.reshape(q_tensor, (1, 1, -1))
        
        error = y_true - y_pred
        loss = tf.maximum(q_tensor * error, (q_tensor - 1.0) * error)
        
        total_loss = tf.reduce_mean(loss)
        return total_loss

    def sample(self, params: np.ndarray, n_samples: int) -> np.ndarray:
        # Sample from the implicit distribution defined by quantiles by approximating the inverse CDF using linear interpolation of the quantiles

        horizon = params.shape[0]
        n_quantiles = len(self.quantiles_list)
        samples = np.zeros((n_samples, horizon))
        q_arr = np.array(self.quantiles_list)
        
        for t in range(horizon):
            p_t = np.sort(params[t])
            
            inv_cdf = interp1d(q_arr, p_t, kind='linear', fill_value="extrapolate")
            
            u = np.random.uniform(0, 1, size=n_samples)
            u = np.clip(u, q_arr[0], q_arr[-1])
            
            samples[:, t] = inv_cdf(u)
            
        return samples

    def mean(self, params: np.ndarray) -> np.ndarray:
        q_target = 0.5
        idx = -1
        for i, q in enumerate(self.quantiles_list):
            if abs(q - 0.5) < 1e-6:
                idx = i
                break
        
        if idx != -1:
            return params[:, idx]
        else:
            # Interpolate
            horizon = params.shape[0]
            means = np.zeros(horizon)
            q_arr = np.array(self.quantiles_list)
            for t in range(horizon):
                p_t = np.sort(params[t])
                f = interp1d(q_arr, p_t, fill_value="extrapolate")
                means[t] = f(0.5)
            return means

    def quantiles(self, params: np.ndarray, q_list: List[float]) -> Dict[float, np.ndarray]:
        horizon = params.shape[0]
        results = {}
        q_arr = np.array(self.quantiles_list)
        
        for q in q_list:
            if q in self.quantiles_list:
                idx = self.quantiles_list.index(q)
                results[q] = params[:, idx]
            else:
                # Interpolate
                q_vals = np.zeros(horizon)
                for t in range(horizon):
                    p_t = np.sort(params[t])
                    f = interp1d(q_arr, p_t, fill_value="extrapolate")
                    q_vals[t] = f(q)
                results[q] = q_vals
        return results

    def log_pdf_np(self, params: np.ndarray, y: np.ndarray) -> np.ndarray:
        # Approximate log PDF from quantile predictions (piecewise uniform between quantiles)
        q_arr = np.array(self.quantiles_list)
        orig_shape = y.shape
        if params.ndim == 2:
            params = params.reshape(1, -1, len(q_arr))
            y = y.reshape(1, -1)
        n, horizon, n_q = params.shape
        log_pdf = np.zeros_like(y, dtype=np.float64)
        eps = 1e-10
        for i in range(n):
            for t in range(horizon):
                # Sort (q, p) by p so intervals are correct
                pairs = sorted(zip(q_arr, params[i, t]), key=lambda x: x[1])
                q_sorted = np.array([x[0] for x in pairs])
                p_sorted = np.array([x[1] for x in pairs])
                y_val = float(y[i, t])
                if y_val <= p_sorted[0]:
                    dq = q_sorted[1] - q_sorted[0] if len(q_sorted) > 1 else 0.1
                    dw = max(p_sorted[1] - p_sorted[0], eps) if len(p_sorted) > 1 else eps
                    log_pdf[i, t] = np.log(dq + eps) - np.log(dw)
                elif y_val >= p_sorted[-1]:
                    dq = q_sorted[-1] - q_sorted[-2] if len(q_sorted) > 1 else 0.1
                    dw = max(p_sorted[-1] - p_sorted[-2], eps) if len(p_sorted) > 1 else eps
                    log_pdf[i, t] = np.log(dq + eps) - np.log(dw)
                else:
                    idx = np.searchsorted(p_sorted, y_val, side="right") - 1
                    idx = min(idx, len(q_sorted) - 2)
                    dq = q_sorted[idx + 1] - q_sorted[idx]
                    dw = max(p_sorted[idx + 1] - p_sorted[idx], eps)
                    log_pdf[i, t] = np.log(dq + eps) - np.log(dw)
        return log_pdf.reshape(orig_shape)


class MixtureGaussianHead(DistributionHead):
    # Mixture of Gaussians heads

    def __init__(self, n_components: int = 3):
        self.n_components = n_components

    def build_output_layer(self, x: tf.Tensor, horizon: int) -> tf.Tensor:
        # 3 params per component (weight, mu, sigma)
        total_params = self.n_components * 3
        param_units = horizon * total_params
        x = layers.Dense(param_units, activation="linear", name="gmm_params")(x)
        return layers.Reshape((horizon, total_params), name="gmm_reshaped")(x)

    def _split_params(self, y_pred: tf.Tensor):
        shape = tf.shape(y_pred)
        reshaped = tf.reshape(y_pred, (shape[0], shape[1], self.n_components, 3))
        
        logits = reshaped[..., 0]
        mu = reshaped[..., 1]
        sigma = tf.nn.softplus(reshaped[..., 2]) + 1e-6
        
        return logits, mu, sigma

    def loss(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        if len(y_true.shape) == 3:
            y_true = tf.squeeze(y_true, axis=-1) # (batch, horizon)
            
        logits, mu, sigma = self._split_params(y_pred)
        
        y_true_ex = tf.expand_dims(y_true, axis=-1)
        
        const_term = -0.5 * tf.math.log(2.0 * np.pi)
        log_probs = const_term - tf.math.log(sigma) - 0.5 * tf.square((y_true_ex - mu) / sigma)
        
        # Total log-likelihood = log(sum(exp(logits - logsumexp(logits)) * exp(log_probs)))
        #   = log(sum(exp(log_weights + log_probs)))
        #   = logsumexp(log_weights + log_probs)
        
        log_weights = tf.nn.log_softmax(logits, axis=-1)
        weighted_log_probs = log_weights + log_probs
        
        log_likelihood = tf.reduce_logsumexp(weighted_log_probs, axis=-1)
        return -tf.reduce_mean(log_likelihood)

    def sample(self, params: np.ndarray, n_samples: int) -> np.ndarray:
        horizon = params.shape[0]
        samples = np.zeros((n_samples, horizon))
        
        params = params.reshape(horizon, self.n_components, 3)
        logits = params[..., 0]
        mu = params[..., 1]
        sigma = np.log(np.exp(params[..., 2]) + 1) + 1e-6
        
        # calculate weights from logits
        max_logits = np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits - max_logits)
        weights = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        for t in range(horizon):
            component_indices = np.random.choice(self.n_components, size=n_samples, p=weights[t])
            
            chosen_mu = mu[t, component_indices]
            chosen_sigma = sigma[t, component_indices]
            samples[:, t] = np.random.normal(loc=chosen_mu, scale=chosen_sigma)
            
        return samples

    def mean(self, params: np.ndarray) -> np.ndarray:
        horizon = params.shape[0]
        params = params.reshape(horizon, self.n_components, 3)
        logits = params[..., 0]
        mu = params[..., 1]
        
        max_logits = np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits - max_logits)
        weights = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        means = np.sum(weights * mu, axis=1)
        return means

    def quantiles(self, params: np.ndarray, q_list: List[float]) -> Dict[float, np.ndarray]:
        n_approx_samples = 10000
        samples = self.sample(params, n_approx_samples)
        
        results = {}
        for q in q_list:
            results[q] = np.quantile(samples, q, axis=0)
        return results


class MixtureJohnsonSUHead(DistributionHead):
    # Mixture of Johnson SU distributions head

    def __init__(self, n_components: int = 3):
        self.n_components = n_components
        
    def build_output_layer(self, x: tf.Tensor, horizon: int) -> tf.Tensor:
        total_params = self.n_components * 5
        param_units = horizon * total_params
        x = layers.Dense(param_units, activation="linear", name="jsu_mix_params")(x)
        return layers.Reshape((horizon, total_params), name="jsu_mix_reshaped")(x)
    
    def _split_params(self, y_pred: tf.Tensor):
        shape = tf.shape(y_pred)
        reshaped = tf.reshape(y_pred, (shape[0], shape[1], self.n_components, 5))
        
        logits = reshaped[..., 0]
        gamma = reshaped[..., 1]
        delta = tf.nn.softplus(reshaped[..., 2]) + 1e-6
        xi = reshaped[..., 3]
        lam = tf.nn.softplus(reshaped[..., 4]) + 1e-6
        
        return logits, gamma, delta, xi, lam

    def loss(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        if len(y_true.shape) == 3:
            y_true = tf.squeeze(y_true, axis=-1)
            
        logits, gamma, delta, xi, lam = self._split_params(y_pred)
        
        y_true_ex = tf.expand_dims(y_true, axis=-1)
        
        z = (y_true_ex - xi) / lam
        asinh_z = tf.asinh(z)
        
        log_pdf_k = (
            tf.math.log(delta)
            - tf.math.log(lam)
            - 0.5 * tf.math.log(2.0 * np.pi)
            - 0.5 * tf.square(gamma + delta * asinh_z)
            - 0.5 * tf.math.log(1.0 + tf.square(z) + 1e-8)
        )
        
        log_weights = tf.nn.log_softmax(logits, axis=-1)
        weighted_log_probs = log_weights + log_pdf_k
        
        log_likelihood = tf.reduce_logsumexp(weighted_log_probs, axis=-1)
        return -tf.reduce_mean(log_likelihood)

    def sample(self, params: np.ndarray, n_samples: int) -> np.ndarray:
        horizon = params.shape[0]
        samples = np.zeros((n_samples, horizon))
        
        params = params.reshape(horizon, self.n_components, 5)
        
        logits = params[..., 0]
        gamma = params[..., 1]
        delta = np.log(np.exp(params[..., 2]) + 1) + 1e-6
        xi = params[..., 3]
        lam = np.log(np.exp(params[..., 4]) + 1) + 1e-6
        
        max_logits = np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits - max_logits)
        weights = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        for t in range(horizon):
            component_indices = np.random.choice(self.n_components, size=n_samples, p=weights[t])
            
            g = gamma[t, component_indices]
            d = delta[t, component_indices]
            x = xi[t, component_indices]
            l = lam[t, component_indices]
            
            z = np.random.normal(size=n_samples)
            samples[:, t] = x + l * np.sinh((z - g) / d)
            
        return samples

    def mean(self, params: np.ndarray) -> np.ndarray:
        horizon = params.shape[0]
        params = params.reshape(horizon, self.n_components, 5)
        
        logits = params[..., 0]
        gamma = params[..., 1]
        delta = np.log(np.exp(params[..., 2]) + 1) + 1e-6
        xi = params[..., 3]
        lam = np.log(np.exp(params[..., 4]) + 1) + 1e-6
        
        max_logits = np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits - max_logits)
        weights = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        component_means = xi - lam * np.exp(1/(2 * delta**2)) * np.sinh(gamma / delta)
        
        return np.sum(weights * component_means, axis=1)

    def quantiles(self, params: np.ndarray, q_list: List[float]) -> Dict[float, np.ndarray]:
        # Approximate
        n_approx_samples = 10000
        samples = self.sample(params, n_approx_samples)
        results = {}
        for q in q_list:
            results[q] = np.quantile(samples, q, axis=0)
        return results


class JohnsonSUFloorHead(JohnsonSUHead):
    #Johnson SU with soft floor penalty and (optional) asymmetric loss
    # Penalizes predictions that would put mass below 0
    # Asymmetric weight: higher penalty when underpredicting spikes

    def __init__(self, floor_penalty_weight: float = 0.1, asymmetric_weight: float = 1.5):
        super().__init__()
        self.floor_penalty_weight = floor_penalty_weight
        self.asymmetric_weight = asymmetric_weight

    def _jsu_mean_tf(self, gamma, delta, xi, lam):
        # Johnson SU mean in TF: xi - lam * exp(1/(2*delta^2)) * sinh(gamma/delta)
        term = tf.exp(0.5 / (delta * delta + 1e-8)) * tf.sinh(gamma / (delta + 1e-8))
        return xi - lam * term

    def loss(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        if len(y_true.shape) == 3:
            y_true = tf.squeeze(y_true, axis=-1)

        gamma, delta, xi, lam = self._split_params(y_pred)

        z = (y_true - xi) / lam
        asinh_z = tf.asinh(z)
        log_pdf = (
            tf.math.log(delta)
            - tf.math.log(lam)
            - 0.5 * tf.math.log(2.0 * np.pi)
            - 0.5 * tf.square(gamma + delta * asinh_z)
            - 0.5 * tf.math.log(1.0 + tf.square(z) + 1e-8)
        )
        base_nll = -tf.reduce_mean(log_pdf)

        # Soft floor penalty to penalize when mean < 0
        mean_pred = self._jsu_mean_tf(gamma, delta, xi, lam)
        floor_penalty = tf.reduce_mean(tf.maximum(0.0, -mean_pred))

        # Asymmetric: weight underpredictions (y_true > mean) more for positive y_true
        err = y_true - mean_pred
        mask_under = tf.cast(err > 0, tf.float32)
        mask_pos = tf.cast(y_true > 0, tf.float32)
        asym_penalty = tf.reduce_mean(
            tf.maximum(0.0, err) * mask_under * mask_pos * (self.asymmetric_weight - 1.0)
        )

        return base_nll + self.floor_penalty_weight * floor_penalty + 0.1 * asym_penalty


class TruncatedNormalHead(DistributionHead):
    # Normal distribution truncated at 0

    def build_output_layer(self, x: tf.Tensor, horizon: int) -> tf.Tensor:
        total_params = 2
        param_units = horizon * total_params
        x = layers.Dense(param_units, activation="linear", name="truncnorm_params")(x)
        return layers.Reshape((horizon, total_params), name="truncnorm_reshaped")(x)

    def _split_params(self, y_pred: tf.Tensor):
        mu = y_pred[..., 0]
        sigma = tf.nn.softplus(y_pred[..., 1]) + 1e-6
        return mu, sigma

    def loss(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        if len(y_true.shape) == 3:
            y_true = tf.squeeze(y_true, axis=-1)

        mu, sigma = self._split_params(y_pred)

        a = (0.0 - mu) / sigma
        z = (y_true - mu) / sigma
        sqrt2 = np.sqrt(2.0)
        cdf_a = 0.5 * (1.0 + tf.math.erf(a / sqrt2))
        log_Z = tf.math.log(tf.maximum(1.0 - cdf_a, 1e-10))

        norm_log_pdf = (
            -0.5 * tf.math.log(2.0 * np.pi)
            - tf.math.log(sigma)
            - 0.5 * tf.square(z)
        )
        log_pdf = norm_log_pdf - log_Z
        return -tf.reduce_mean(log_pdf)

    def sample(self, params: np.ndarray, n_samples: int) -> np.ndarray:
        horizon = params.shape[0]
        samples = np.zeros((n_samples, horizon))

        mu = params[:, 0]
        sigma = np.log(np.exp(params[:, 1]) + 1) + 1e-6

        for t in range(horizon):
            a = (0.0 - mu[t]) / sigma[t]
            dist = truncnorm(a, np.inf, loc=mu[t], scale=sigma[t])
            samples[:, t] = dist.rvs(size=n_samples)
        return samples

    def mean(self, params: np.ndarray) -> np.ndarray:
        horizon = params.shape[0]
        means = np.zeros(horizon)

        mu = params[:, 0]
        sigma = np.log(np.exp(params[:, 1]) + 1) + 1e-6

        for t in range(horizon):
            a = (0.0 - mu[t]) / sigma[t]
            dist = truncnorm(a, np.inf, loc=mu[t], scale=sigma[t])
            means[t] = float(dist.mean())
        return means

    def quantiles(self, params: np.ndarray, q_list: List[float]) -> Dict[float, np.ndarray]:
        horizon = params.shape[0]
        results = {}

        mu = params[:, 0]
        sigma = np.log(np.exp(params[:, 1]) + 1) + 1e-6

        for q in q_list:
            q_vals = np.zeros(horizon)
            for t in range(horizon):
                a = (0.0 - mu[t]) / sigma[t]
                dist = truncnorm(a, np.inf, loc=mu[t], scale=sigma[t])
                q_vals[t] = dist.ppf(q)
            results[q] = q_vals
        return results
