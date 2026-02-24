import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Optional, Dict

from config import ExperimentConfig
from .heads import DistributionHead, JohnsonSUHead, GaussianHead, QuantileHead

class ProbabilisticNHITS:
    # Probabilistic N-HiTS

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.model_config = config.model_config
        self.data_config = config.data_config
        
        self.head: DistributionHead
        if config.head_type == "johnson_su":
            self.head = JohnsonSUHead()
        elif config.head_type == "gaussian":
            self.head = GaussianHead()
        elif config.head_type == "quantile":
            self.head = QuantileHead()
        else:
            raise ValueError(f"Unknown head type: {config.head_type}")
            
        self.keras_model: Optional[keras.Model] = None

    def build_model(self, n_features: int):
        conf = self.model_config
        input_window = self.data_config.input_window
        horizon = self.data_config.output_horizon
        
        inputs = keras.Input(shape=(input_window, n_features))
        
        # Flatten input
        x_in = layers.Flatten()(inputs)
        
        # Configurations for stacks
        stacks = [
            {"pool_size": 8, "n_blocks": 1}, # Low freq
            {"pool_size": 4, "n_blocks": 1}, # Mid freq
            {"pool_size": 1, "n_blocks": 1}, # High freq
        ]
        width = conf.ff_dim
        
        forecast_accum = 0
        backcast_residual = x_in
        
        # Determine param_dim based on head type
        if isinstance(self.head, JohnsonSUHead):
            param_dim = 4
        elif isinstance(self.head, GaussianHead):
            param_dim = 2
        elif isinstance(self.head, QuantileHead):
            param_dim = len(self.head.quantiles_list)
        else:
            param_dim = 4  # fallback
        forecast_dim = horizon * param_dim
        
        for stack_id, stack_conf in enumerate(stacks):
            pool_size = stack_conf["pool_size"]
            
            for block_id in range(stack_conf["n_blocks"]):
                x = backcast_residual
                
                # FC Stack
                for _ in range(2):
                    x = layers.Dense(width, activation='relu')(x)
                    x = layers.Dropout(conf.dropout)(x)
                
                # Projections
                theta_b = layers.Dense(width, activation='linear')(x)
                theta_f = layers.Dense(width, activation='linear')(x)
                
                # Backcast (reconstruct input)
                backcast = layers.Dense(input_window * n_features, activation='linear')(theta_b)
                
                # Forecast
                forecast = layers.Dense(forecast_dim, activation='linear')(theta_f)
                
                # Residuals
                backcast_residual = layers.Subtract()([backcast_residual, backcast])
                
                # put together
                if isinstance(forecast_accum, int):
                    forecast_accum = forecast
                else:
                    forecast_accum = layers.Add()([forecast_accum, forecast])
        
        outputs = layers.Reshape((horizon, param_dim))(forecast_accum)
        
        self.keras_model = keras.Model(inputs=inputs, outputs=outputs)
        
        self.keras_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config.training_config.learning_rate),
            loss=lambda y_true, y_pred: self.head.loss(y_true, y_pred)
        )
        return self.keras_model
