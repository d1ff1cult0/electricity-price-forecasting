import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
from config import DataConfig
from .loader import DataLoader

class DataPipeline:
    """
    Handles data splitting and sequence generation
    - Train start: config.train_start_date (default 2023-02-01)
    - Test: last config.test_duration_months (default 6 months)
    """
    def __init__(self, config: DataConfig):
        self.config = config
        self.loader = DataLoader(config)

    def get_data_splits(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        # returns (train_df, val_df, test_df).

        prices, exog = self.loader.load_raw_data()
        
        # Combine into single DataFrame
        df = prices.to_frame(name='Prices')
        if exog is not None:
            df = df.join(exog)
            
        df.index = pd.to_datetime(df.index)
        
        # Define test split (last N months)
        test_start_date = df.index.max() - pd.DateOffset(months=self.config.test_duration_months)
        
        # Define train start date
        train_start_date = pd.to_datetime(self.config.train_start_date)
        
        # Create splits
        # Train [train_start_date, test_start_date)
        train_mask = (df.index >= train_start_date) & (df.index < test_start_date)
        train_df = df[train_mask].copy()
        # Test [test_start_date, end]
        test_mask = (df.index >= test_start_date)
        test_df = df[test_mask].copy()
        # Validation: take last 10% of training data
        val_size = int(len(train_df) * 0.1)
        val_df = train_df.iloc[-val_size:].copy()
        train_df = train_df.iloc[:-val_size].copy()
        
        if len(train_df) == 0:
            raise ValueError(f"Training set is empty")

        return train_df, val_df, test_df

    def create_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        # Creates (X, y) sequences from a DataFrame
        data = df.values
        
        X_list, y_list = [], []
        
        input_win = self.config.input_window
        out_hor = self.config.output_horizon
        n_samples = len(data) - input_win - out_hor + 1
        
        if n_samples <= 0:
            return np.empty((0, input_win, df.shape[1])), np.empty((0, out_hor))

        for i in range(n_samples):
            window = data[i : i + input_win]
            target = data[i + input_win : i + input_win + out_hor, 0] # 0 is Prices
            
            if np.isnan(window).any() or np.isnan(target).any():
                continue
                
            X_list.append(window)
            y_list.append(target)
            
        return np.array(X_list), np.array(y_list)
