import os
import json
import gc
import logging
import numpy as np
import tensorflow as tf
from typing import Dict, Any, List
from pathlib import Path
from config import ExperimentConfig
from data import DataPipeline
from transformations import StandardScalingTransformation
from models import ProbabilisticTransformer
from .trainer import Trainer
from .evaluator import Evaluator

logger = logging.getLogger(__name__)

class ExperimentManager:
    """
    Runs experiments
    1. checks disk for existing results to avoid re-running
    2. runs N times to reduce statistical validation
    3. aggressive cleanup after each run for memory usage because before memory kept increasing causing server to crash
    """
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results_dir = Path(config.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.result_file = self.results_dir / f"{config.name}_results.json"

    def run(self) -> Dict[str, Any]:
        """
        runs the experiment or loads from disk.
        """
        # Check if it is already calculated
        if self.result_file.exists():
            logger.info(f"Found existing results at {self.result_file}, loading")
            with open(self.result_file, "r") as f:
                return json.load(f)

        logger.info(f"No existing results found, starting experiment '{self.config.name}' with {self.config.n_runs} runs")
        
        # Prepare Data
        pipeline = DataPipeline(self.config.data_config)
        train_df, val_df, test_df = pipeline.get_data_splits()
        X_train_raw, y_train_raw = pipeline.create_sequences(train_df)
        X_val_raw, y_val_raw = pipeline.create_sequences(val_df)
        X_test_raw, y_test_raw = pipeline.create_sequences(test_df)

        # transformation
        from transformations import (
            StandardScalingTransformation, 
            YeoJohnsonTransformation, 
            ArcsinhTransformation, 
            RobustScalerTransformation
        )

        logger.info(f"Transforming data using {self.config.transform_type}")
        
        if self.config.transform_type == "standard_scaling":
            transform = StandardScalingTransformation()
        elif self.config.transform_type == "yeo_johnson":
            transform = YeoJohnsonTransformation()
        elif self.config.transform_type == "arcsinh":
            transform = ArcsinhTransformation()
        elif self.config.transform_type == "robust_scaling":
            transform = RobustScalerTransformation()
        else:
            raise ValueError(f"Unknown transform_type: {self.config.transform_type}")

        transform.fit(X_train_raw, y_train_raw)
        
        X_train, y_train = transform.transform(X_train_raw, y_train_raw)
        X_val, y_val = transform.transform(X_val_raw, y_val_raw)
        X_test, y_test = transform.transform(X_test_raw, y_test_raw)
        
        all_metrics = []
        
        # Validation loop
        for run_id in range(self.config.n_runs):
            logger.info(f"Run {run_id + 1}/{self.config.n_runs} ")
            
            # memory cleanup
            self._cleanup()
            
            try:
                # model build
                model = ProbabilisticTransformer(self.config)
                
                # training
                trainer = Trainer(self.config)
                trainer.train(model, X_train, y_train, X_val, y_val)
                
                # evaluation
                evaluator = Evaluator(model, transform)
                metrics = evaluator.evaluate(X_test, y_test_raw)
                all_metrics.append(metrics)
                
                logger.info(f"Run {run_id + 1} Metrics: {metrics}")
                
            except Exception as e:
                logger.error(f"Run {run_id + 1} failed: {e}")
                raise e
                
            finally:
                del model
                del trainer
                del evaluator
                
                self._cleanup()
        
        # get metrics
        aggregated = self._aggregate_metrics(all_metrics)
        
        # save results
        final_results = {
            "config": str(self.config),
            "runs": all_metrics,
            "aggregated": aggregated
        }
        
        with open(self.result_file, "w") as f:
            json.dump(final_results, f, indent=2)
            
        logger.info(f"Experiment completed. Results saved to {self.result_file}")
        return final_results

    def _cleanup(self):
        # forceful garbage collection and backend clearing
        tf.keras.backend.clear_session()
        gc.collect()

    def _aggregate_metrics(self, metrics_list: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        # compute Mean and Std for all metrics
        keys = metrics_list[0].keys()
        agg = {}
        for k in keys:
            values = [m[k] for m in metrics_list]
            agg[k] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values))
            }
        return agg
