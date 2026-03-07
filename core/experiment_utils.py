"""
standardized experiment utilities for electricity price forecasting.

Provides standardized hyperparameters, data loading, evaluation, and
result/model/prediction persistence so that all notebooks use
the same pipeline and produce comparable results
"""

import gc
import json
import time
from collections import namedtuple
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

from config import DataConfig, ExperimentConfig, ModelConfig, TrainingConfig
from core.evaluator import Evaluator
from core.trainer import Trainer
from data import DataPipeline
from transformations import StandardScalingTransformation

# Standard parameters
CANONICAL_MODEL_CONFIG = {
    "d_model": 224,
    "num_heads": 7,
    "num_layers": 3,
    "ff_dim": 256,
    "dropout": 0.15,
}

CANONICAL_TRAIN_CONFIG = {
    "batch_size": 32,
    "epochs": 30,
    "learning_rate": 7e-4,
    "patience": 5,
}

INPUT_WINDOW = 168
OUTPUT_HORIZON = 24
N_RUNS = 10

QUANTILE_LEVELS = [0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975]


DataBundle = namedtuple("DataBundle", [
    "X_train_s", "y_train_s",
    "X_val_s", "y_val_s",
    "X_test_s", "y_test_s",
    "y_train", "y_val", "y_test",
    "scaler", "pipeline", "data_config",
])


# Data loading
def load_data(
    input_window: int = INPUT_WINDOW,
    output_horizon: int = OUTPUT_HORIZON,
    dataset_name: str = "BE_ENTSOE",
) -> DataBundle:
    """load data through the standard pipeline"""
    data_config = DataConfig(
        dataset_name=dataset_name,
        input_window=input_window,
        output_horizon=output_horizon,
    )
    pipeline = DataPipeline(data_config)
    df_train, df_val, df_test = pipeline.get_data_splits()

    X_train, y_train = pipeline.create_sequences(df_train)
    X_val, y_val = pipeline.create_sequences(df_val)
    X_test, y_test = pipeline.create_sequences(df_test)

    scaler = StandardScalingTransformation()
    scaler.fit(X_train, y_train)
    X_train_s, y_train_s = scaler.transform(X_train, y_train)
    X_val_s, y_val_s = scaler.transform(X_val, y_val)
    X_test_s, y_test_s = scaler.transform(X_test, y_test)

    print(f"Data loaded  —  Train: {X_train_s.shape[0]}, "
          f"Val: {X_val_s.shape[0]}, Test: {X_test_s.shape[0]}")

    return DataBundle(
        X_train_s=X_train_s, y_train_s=y_train_s,
        X_val_s=X_val_s, y_val_s=y_val_s,
        X_test_s=X_test_s, y_test_s=y_test_s,
        y_train=y_train, y_val=y_val, y_test=y_test,
        scaler=scaler, pipeline=pipeline, data_config=data_config,
    )


# Evaluation helpers
def _is_hybrid(model) -> bool:
    return hasattr(model, "predict_hybrid") and hasattr(model, "fit_ou")


def evaluate_model(
    model,
    data: DataBundle,
    training_time: Optional[float] = None,
    quantile_levels: Optional[List[float]] = None,
    n_crps_samples: int = 200,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    # Evaluate a model (standard or hybrid) and return all metrics plus raw predictions suitable for saving

    quantile_levels = quantile_levels or QUANTILE_LEVELS
    scaler = data.scaler
    y_test = data.y_test
    X_test_s = data.X_test_s

    if _is_hybrid(model):
        metrics, predictions = _evaluate_hybrid(
            model, X_test_s, y_test, scaler, quantile_levels, n_crps_samples,
        )
    else:
        evaluator = Evaluator(model, scaler)
        metrics = evaluator.evaluate(X_test_s, y_test)
        y_pred_mean = evaluator.generate_forecasts(X_test_s)
        q_originals = evaluator.generate_quantile_forecasts(X_test_s, quantile_levels)
        predictions = {"y_pred_mean": y_pred_mean}
        for q, arr in q_originals.items():
            predictions[f"q_{q}"] = arr

    if training_time is not None:
        metrics["training_time"] = float(training_time)

    return metrics, predictions


def _evaluate_hybrid(
    model, X_test_s, y_test, scaler, quantile_levels, n_crps_samples,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    # Evaluate a HybridProbabilisticTransformer variant
    y_pred_scaled = model.predict_hybrid(X_test_s, last_residuals=None)
    _, y_pred_mean = scaler.inverse_transform(X=None, y=y_pred_scaled)

    q_scaled_dict = model.quantiles_hybrid(
        X_test_s, quantile_levels, n_samples=n_crps_samples, last_residuals=None,
    )
    q_originals = {}
    for q, val_scaled in q_scaled_dict.items():
        _, val_raw = scaler.inverse_transform(X=None, y=val_scaled)
        q_originals[q] = val_raw

    # Deterministic metrics
    mae = float(np.mean(np.abs(y_test - y_pred_mean)))
    mse = float(np.mean((y_test - y_pred_mean) ** 2))
    rmse = float(np.sqrt(mse))
    mask = y_test != 0
    mape = float(np.mean(np.abs((y_test[mask] - y_pred_mean[mask]) / y_test[mask])) * 100) if mask.any() else np.nan
    ss_res = np.sum((y_test - y_pred_mean) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    # Interval metrics (95% PI)
    q_low = q_originals.get(0.025, q_originals.get(min(quantile_levels)))
    q_high = q_originals.get(0.975, q_originals.get(max(quantile_levels)))
    covered = (y_test >= q_low) & (y_test <= q_high)
    picp = float(np.mean(covered))
    mpiw = float(np.mean(q_high - q_low))
    y_range = np.max(y_test) - np.min(y_test)
    pinaw = float(mpiw / y_range) if y_range > 0 else np.nan
    alpha = 0.05
    width = q_high - q_low
    lower_pen = (2 / alpha) * (q_low - y_test) * (y_test < q_low)
    upper_pen = (2 / alpha) * (y_test - q_high) * (y_test > q_high)
    interval_score = float(np.mean(width + lower_pen + upper_pen))

    # CRPS via sampling
    samples_scaled = model.sample_hybrid(X_test_s, n_samples=n_crps_samples, last_residuals=None)
    # samples_scaled shape: (batch, horizon, n_samples)
    n_samp = samples_scaled.shape[-1]
    batch_size_eval = samples_scaled.shape[0]
    horizon = samples_scaled.shape[1]
    # reshape to (n_samples, batch, horizon) for inverse transform
    samples_perm = np.transpose(samples_scaled, (2, 0, 1))
    samples_flat = samples_perm.reshape(-1, horizon)
    _, samples_flat_orig = scaler.inverse_transform(X=None, y=samples_flat)
    samples_orig = samples_flat_orig.reshape(n_samp, batch_size_eval, horizon)

    y_exp = np.expand_dims(y_test, axis=0)
    term1 = np.mean(np.abs(samples_orig - y_exp), axis=0)
    samples_sorted = np.sort(samples_orig, axis=0)
    weights = (2 * np.arange(n_samp) - n_samp + 1).reshape(-1, 1, 1)
    term2 = np.sum(weights * samples_sorted, axis=0) / (n_samp * n_samp)
    crps = float(np.mean(term1 - term2))

    # Pinball losses
    def _pinball(y_true, y_pred_q, q):
        err = y_true - y_pred_q
        return float(np.mean(np.maximum(q * err, (q - 1.0) * err)))

    pinball_10 = _pinball(y_test, q_originals.get(0.1, q_low), 0.1)
    pinball_50 = _pinball(y_test, q_originals.get(0.5, y_pred_mean), 0.5)
    pinball_90 = _pinball(y_test, q_originals.get(0.9, q_high), 0.9)
    avg_pinball = (pinball_10 + pinball_50 + pinball_90) / 3.0

    # NLL (only for standard head)
    nll = np.nan

    metrics = {
        "MAE": mae, "MSE": mse, "RMSE": rmse, "MAPE": mape, "R2": r2,
        "PICP": picp, "MPIW": mpiw, "PINAW": pinaw,
        "IntervalScore": interval_score, "CRPS": crps, "NLL": nll,
        "Pinball_10": pinball_10, "Pinball_50": pinball_50,
        "Pinball_90": pinball_90, "Avg_Pinball": avg_pinball,
    }

    predictions = {"y_pred_mean": y_pred_mean}
    for q, arr in q_originals.items():
        predictions[f"q_{q}"] = arr

    return metrics, predictions


# ---------------------------------------------------------------------------
# Result persistence
# ---------------------------------------------------------------------------

def save_run(
    results_dir: str,
    run_idx: int,
    model,
    metrics: Dict[str, float],
    predictions: Dict[str, np.ndarray],
    y_test: np.ndarray,
) -> None:
    # Save one run's metrics, predictions, and model weights
    run_dir = Path(results_dir) / f"run_{run_idx}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Metrics
    safe_metrics = {k: (float(v) if v is not None and not np.isnan(v) else None)
                    for k, v in metrics.items()}
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(safe_metrics, f, indent=2)

    # Predictions
    save_dict = {"y_test": y_test}
    for k, v in predictions.items():
        save_dict[k] = v
    np.savez_compressed(run_dir / "predictions.npz", **save_dict)

    # Model weights
    if hasattr(model, "keras_model") and model.keras_model is not None:
        try:
            model.keras_model.save_weights(str(run_dir / "model_weights.weights.h5"))
        except Exception:
            pass
    else:
        try:
            import joblib
            joblib.dump(model, str(run_dir / "model.joblib"))
        except Exception:
            pass


def save_summary(
    results_dir: str,
    all_run_metrics: List[Dict[str, float]],
    config_dict: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    # Compute and save averaged metrics + std across runs
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    avg, std = {}, {}
    for k in all_run_metrics[0]:
        vals = [m[k] for m in all_run_metrics
                if m.get(k) is not None and np.isfinite(m[k])]
        avg[k] = float(np.mean(vals)) if vals else None
        std[k] = float(np.std(vals)) if vals else None

    summary = {"avg": avg, "std": std, "n_runs": len(all_run_metrics)}
    if config_dict:
        summary["config"] = config_dict
    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    rows = [{"metric": k, "mean": avg[k], "std": std[k]} for k in avg]
    pd.DataFrame(rows).to_csv(results_dir / "summary.csv", index=False)

    return avg, std


def load_run(results_dir: str, run_idx: int) -> Tuple[Dict, Dict[str, np.ndarray]]:
    # Reload a single run's metrics and predictions
    run_dir = Path(results_dir) / f"run_{run_idx}"
    with open(run_dir / "metrics.json") as f:
        metrics = json.load(f)
    data = np.load(run_dir / "predictions.npz")
    predictions = {k: data[k] for k in data.files}
    return metrics, predictions


# cache helpers
def load_cache(cache_file) -> Dict[str, Any]:
    cache_file = Path(cache_file)
    if cache_file.exists():
        try:
            with open(cache_file, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}


def save_cache(cache_file, cache: Dict[str, Any]) -> None:
    cache_file = Path(cache_file)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(cache, f, indent=2)



def set_seeds(run_idx: int = 0, base_seed: int = 42) -> None:
    """Set Python / NumPy / TF random seeds for reproducibility."""
    seed = base_seed + run_idx
    np.random.seed(seed)
    tf.random.set_seed(seed)



def run_experiment(
    model_cls,
    exp_name: str,
    data: DataBundle,
    results_base_dir: str,
    n_runs: int = N_RUNS,
    model_config: Optional[ModelConfig] = None,
    train_config: Optional[TrainingConfig] = None,
    head_type: str = "johnson_su",
    head_params: Optional[Dict] = None,
    is_hybrid: bool = False,
    cache: Optional[Dict] = None,
) -> Optional[Dict[str, float]]:
    """
    Train a model n_runs times, evaluate, and persist everything

    Parameters
    ----------
    model_cls : class
        Model constructor (e.g. ProbabilisticTransformer)
    exp_name : str
        Human-readable experiment name (also used as result sub-directory)
    data : DataBundle
        Output of load_data()
    results_base_dir : str
        Top-level results folder; a sub-folder exp_name is created inside
    n_runs : int
    model_config, train_config : optional overrides
    head_type, head_params : distribution head settings
    is_hybrid : bool
        If True, call fit_ou / predict_hybrid path
    cache : dict or None
        If provided and exp_name already present, skip training

    Returns
    -------
    avg_metrics : dict
    """
    if cache is not None and exp_name in cache:
        print(f"[{exp_name}] found in cache — skipping.")
        return cache[exp_name]

    results_dir = Path(results_base_dir) / exp_name.replace(" ", "_")
    summary_file = results_dir / "summary.json"
    if summary_file.exists():
        try:
            with open(summary_file, "r") as f:
                summary = json.load(f)
            avg = summary.get("avg", {})
            if avg:
                print(f"[{exp_name}] found on disk ({summary_file}) — skipping.")
                if cache is not None:
                    cache[exp_name] = avg
                return avg
        except (json.JSONDecodeError, KeyError):
            pass

    model_config = model_config or ModelConfig(**CANONICAL_MODEL_CONFIG)
    train_config = train_config or TrainingConfig(**CANONICAL_TRAIN_CONFIG)
    head_params = head_params or {}

    results_dir.mkdir(parents=True, exist_ok=True)

    config_snapshot = {
        "model": {k: getattr(model_config, k) for k in model_config.__dataclass_fields__},
        "training": {k: getattr(train_config, k) for k in train_config.__dataclass_fields__},
        "head_type": head_type,
        "head_params": head_params,
        "n_runs": n_runs,
    }

    print(f"\n{'='*60}")
    print(f"  Experiment: {exp_name}")
    print(f"  Model: {model_cls.__name__}  |  head: {head_type}  |  runs: {n_runs}")
    print(f"{'='*60}")

    all_metrics: List[Dict[str, float]] = []

    for r in range(n_runs):
        print(f"  Run {r + 1}/{n_runs} ...")
        tf.keras.backend.clear_session()
        gc.collect()
        set_seeds(r)

        exp_conf = ExperimentConfig(
            name=f"{exp_name}_run{r}",
            data_config=data.data_config,
            model_config=model_config,
            training_config=train_config,
            head_type=head_type,
            head_params=head_params,
        )

        model = model_cls(exp_conf)
        trainer = Trainer(exp_conf)

        t0 = time.time()
        try:
            trainer.train(model, data.X_train_s, data.y_train_s,
                          data.X_val_s, data.y_val_s)
        except Exception as e:
            print(f"    Training error: {e}")
            continue
        training_time = time.time() - t0

        if is_hybrid:
            model.fit_ou(data.X_train_s, data.y_train_s)

        metrics, predictions = evaluate_model(model, data, training_time)
        save_run(results_dir, r, model, metrics, predictions, data.y_test)
        all_metrics.append(metrics)

    if not all_metrics:
        return None

    avg, std = save_summary(results_dir, all_metrics, config_snapshot)

    if cache is not None:
        cache[exp_name] = avg

    print(f"  MAE: {avg['MAE']:.4f} ± {std['MAE']:.4f}")
    return avg
