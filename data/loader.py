import pandas as pd
from typing import Tuple, List, Optional
from pathlib import Path
from config import DataConfig


class DataLoader:
    # Loads electricity price data and exogenous features.
    def __init__(self, config: DataConfig):
        self.config = config

    def load_raw_data(self) -> Tuple[pd.Series, Optional[pd.DataFrame]]:
        # Loads the raw prices and exogenous features (legacy code from before refactoring)
        return self._load_from_csv()
    
    def _load_from_csv(self) -> Tuple[pd.Series, Optional[pd.DataFrame]]:
        """
        Load data directly from CSV file.
        
        Expected CSV format:
        - First column: 'Date' (datetime)
        - Second column: 'Prices' (target variable)
        - Remaining columns: Exogenous features
        """

        base_dir = Path(__file__).parent
        possible_paths = [
            base_dir / "datasets" / f"{self.config.dataset_name}.csv",
            base_dir / f"{self.config.dataset_name}.csv",
            Path("data/datasets") / f"{self.config.dataset_name}.csv",
            Path("data") / f"{self.config.dataset_name}.csv",
            Path("datasets") / f"{self.config.dataset_name}.csv",
            Path(self.config.dataset_name + ".csv"),
        ]
        
        csv_path = None
        for path in possible_paths:
            if path.exists():
                csv_path = path
                break
        
        if csv_path is None:
            raise FileNotFoundError(
                f"Could not find dataset '{self.config.dataset_name}'.csv"
            )
        
        # Load CSV
        df = pd.read_csv(csv_path)
        
        # Ensure Date column exists and convert to datetime
        if 'Date' not in df.columns:
            raise ValueError(
                f"CSV file {csv_path} must have a 'Date' column "
                f"Found columns: {list(df.columns)}"
            )
        
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date').sort_index()
        
        # Extract prices
        if 'Prices' not in df.columns:
            raise ValueError(
                f"CSV file {csv_path} must have a 'Prices' column "
                f"Found columns: {list(df.columns)}"
            )
        
        prices = df['Prices']
        
        # Extract exogenous features
        exog = df.drop(columns=['Prices'])
        
        # Remove columns that are all NaN
        exog = exog.dropna(axis=1, how='all')
        
        # Return None if no exogenous features
        if exog.empty:
            exog = None
        
        return prices, exog
