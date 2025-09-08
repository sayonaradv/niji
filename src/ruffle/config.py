import os

from pydantic.dataclasses import dataclass


@dataclass
class Config:
    cache_dir: str = "/data"
    log_dir: str = "/lightning_logs"
    seed: int = 1234


@dataclass
class DatasetConfig:
    data_dir: str = os.path.join(
        "data", "jigsaw-toxic-comment-classification-challenge"
    )
    batch_size: int = 32
    val_size: float = 0.2
