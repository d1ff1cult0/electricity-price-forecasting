import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Optional, Dict

from config import ExperimentConfig
from .heads import DistributionHead, JohnsonSUHead, GaussianHead, QuantileHead

class ProbabilisticLSTM:
    # Probabilistic LSTM Model

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
        # Builds the compiled Keras model

        conf = self.model_config
        
        inputs = keras.Input(shape=(self.data_config.input_window, n_features), name="input")
        
        x = inputs
        
        # LSTM Layers
        for i in range(conf.num_layers):
            return_sequences = (i < conf.num_layers - 1)
            
            x = layers.LSTM(
                units=int(conf.d_model), 
                return_sequences=return_sequences,
                dropout=conf.dropout
            )(x)
            
        # Head
        x = layers.Dense(conf.ff_dim, activation="relu")(x)
        x = layers.Dropout(conf.dropout)(x)
        
        # Output head
        outputs = self.head.build_output_layer(x, self.data_config.output_horizon)
        
        self.keras_model = keras.Model(inputs=inputs, outputs=outputs)
        
        # Compile
        self.keras_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config.training_config.learning_rate),
            loss=lambda y_true, y_pred: self.head.loss(y_true, y_pred)
        )
        return self.keras_model
