import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Optional, Dict

from config import ExperimentConfig
from .heads import DistributionHead, JohnsonSUHead, GaussianHead, QuantileHead

class ProbabilisticNBEATS:
    # Probabilistic N-BEATS

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
        # Builds N-BEATS

        conf = self.model_config
        input_window = self.data_config.input_window
        horizon = self.data_config.output_horizon
        
        inputs = keras.Input(shape=(input_window, n_features))
        
        x_in = layers.Flatten()(inputs)
        
        # Configuration
        num_blocks = 3
        num_layers_per_block = 2
        width = conf.ff_dim
        
        
        forecast_accum = 0
        backcast_residual = x_in
        
        for i in range(num_blocks):
            x = backcast_residual
            
            # FC Stack
            for _ in range(num_layers_per_block):
                x = layers.Dense(width, activation='relu')(x)
                
            theta_b = layers.Dense(width, activation='linear', name=f"theta_b_{i}")(x)
            theta_f = layers.Dense(width, activation='linear', name=f"theta_f_{i}")(x)
            
            backcast = layers.Dense(input_window * n_features, activation='linear', name=f"backcast_{i}")(theta_b)
            
            if isinstance(self.head, JohnsonSUHead):
                param_dim = 4
            elif isinstance(self.head, GaussianHead):
                param_dim = 2
            elif isinstance(self.head, QuantileHead):
                param_dim = len(self.head.quantiles_list)
            else:
                param_dim = 4  # fallback
            forecast_dim = horizon * param_dim
            
            forecast = layers.Dense(forecast_dim, activation='linear', name=f"forecast_{i}")(theta_f)
            
            # Update residuals
            backcast_residual = layers.Subtract()([backcast_residual, backcast])
            
            # Accumulate forecast
            if isinstance(forecast_accum, int):
                forecast_accum = forecast
            else:
                forecast_accum = layers.Add()([forecast_accum, forecast])
                
        # Final output reshape
        outputs = layers.Reshape((horizon, param_dim))(forecast_accum)
        
        self.keras_model = keras.Model(inputs=inputs, outputs=outputs)
        
        self.keras_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config.training_config.learning_rate),
            loss=lambda y_true, y_pred: self.head.loss(y_true, y_pred)
        )
        return self.keras_model
