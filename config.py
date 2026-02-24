from dataclasses import dataclass, field
from typing import List, Optional, Union, Dict, Any

@dataclass
class DataConfig:
    dataset_name: str = "BE_ENTSOE"
    features: List[str] = field(default_factory=lambda: ["Prices"])
    train_start_date: str = "2023-02-01" 
    test_duration_months: int = 6
    input_window: int = 168
    output_horizon: int = 24

@dataclass
class ModelConfig:
    d_model: int = 224
    num_heads: int = 7
    num_layers: int = 3
    ff_dim: int = 448
    dropout: float = 0.15

@dataclass
class TrainingConfig:
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 7e-4
    patience: int = 10
    validation_split: float = 0.1
    random_state: int = 42

@dataclass
class ExperimentConfig:
    name: str
    data_config: DataConfig = field(default_factory=DataConfig)
    model_config: ModelConfig = field(default_factory=ModelConfig)
    training_config: TrainingConfig = field(default_factory=TrainingConfig)
    transform_type: str = "standard_scaling"
    head_type: str = "johnson_su"
    head_params: Dict[str, Any] = field(default_factory=dict)
    n_runs: int = 10
    results_dir: str = "results"
