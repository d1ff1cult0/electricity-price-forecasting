import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Optional, Dict, Tuple

from config import ExperimentConfig
from .heads import DistributionHead, JohnsonSUHead, GaussianHead, QuantileHead

class ProbabilisticDeepAR:
    # Probabilistic DeepAR (Autoregressive LSTM) Model with Teacher Forcing as training method and autoregressive sampling as inference method
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
        # Builds the training model (Teacher Forcing)
        conf = self.model_config
        horizon = self.data_config.output_horizon
        
        # Encoder
        encoder_inputs = keras.Input(shape=(None, n_features), name="encoder_input")
        encoder_lstm = layers.LSTM(int(conf.d_model), return_state=True, dropout=conf.dropout, name="encoder_lstm")
        _, state_h, state_c = encoder_lstm(encoder_inputs)
        encoder_states = [state_h, state_c]
        
        repeated_context = layers.RepeatVector(horizon)(state_h)
        
        decoder_lstm = layers.LSTM(int(conf.d_model), return_sequences=True, dropout=conf.dropout, name="decoder_lstm")
        decoder_outputs = decoder_lstm(repeated_context, initial_state=encoder_states)
        
        if isinstance(self.head, JohnsonSUHead):
            param_dim = 4
        elif isinstance(self.head, GaussianHead):
            param_dim = 2
        elif isinstance(self.head, QuantileHead):
            param_dim = len(self.head.quantiles_list)
        else:
            param_dim = 4  # fallback
        
        x = layers.TimeDistributed(layers.Dense(conf.ff_dim, activation="relu"))(decoder_outputs)
        x = layers.TimeDistributed(layers.Dropout(conf.dropout))(x)
        
        outputs = layers.TimeDistributed(layers.Dense(param_dim, activation="linear"))(x)
        
        self.keras_model = keras.Model(inputs=encoder_inputs, outputs=outputs)
        
        self.keras_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config.training_config.learning_rate),
            loss=lambda y_true, y_pred: self.head.loss(y_true, y_pred)
        )
        return self.keras_model
