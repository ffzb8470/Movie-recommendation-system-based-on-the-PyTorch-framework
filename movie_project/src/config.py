from dataclasses import dataclass
from typing import List

@dataclass
class DataConfig:
    """数据配置"""
    dataset_path: str = "data/ml-latest-small"
    min_user_ratings: int = 50
    min_movie_ratings: int = 50
    test_size: float = 0.15
    val_size: float = 0.15
    random_state: int = 42

@dataclass
class ModelConfig:
    """模型配置"""
    embedding_dim: int = 64
    hidden_dims: List[int] = None
    dropout: float = 0.3
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [128, 64, 32]

@dataclass
class TrainingConfig:
    """训练配置"""
    batch_size: int = 512
    epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    patience: int = 10
    device: str = "auto"  # auto, cpu, cuda
    
    def __post_init__(self):
        if self.device == "auto":
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class Config:
    """总配置"""
    data: DataConfig = None
    model: ModelConfig = None
    training: TrainingConfig = None
    
    def __post_init__(self):
        self.data = DataConfig()
        self.model = ModelConfig()
        self.training = TrainingConfig()