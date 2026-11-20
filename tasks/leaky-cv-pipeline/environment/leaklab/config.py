from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    n_splits: int = 5
    seed: int = 0
    k_features: int = 6      # numeric features kept by the selector
    smoothing: float = 10.0  # target-encoding smoothing
    max_epochs: int = 200
    lr: float = 0.1
    patience: int = 10
