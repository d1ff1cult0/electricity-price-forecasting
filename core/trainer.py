import tensorflow as tf
from tensorflow import keras
import numpy as np
from typing import Optional, List
from config import ExperimentConfig
from models.transformer import ProbabilisticTransformer
from transformations.base import BaseTransformation
import logging

class Trainer:
    # Standardized training loop

    def __init__(self, config: ExperimentConfig):
        self.config = config
        
    def train(self, 
              model: ProbabilisticTransformer, 
              X_train: np.ndarray, 
              y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, 
              y_val: Optional[np.ndarray] = None,
              verbose: int = 1) -> keras.callbacks.History:
        
        # Build model if not built
        if model.keras_model is None:
            # Infer features
            n_features = X_train.shape[-1]
            model.build_model(n_features)
            
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config.training_config.patience,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]
        
        history = model.keras_model.fit(
            X_train, y_train,
            epochs=self.config.training_config.epochs,
            batch_size=self.config.training_config.batch_size,
            validation_data=(X_val, y_val) if X_val is not None else None,
            callbacks=callbacks,
            verbose=verbose
        )
        return history
