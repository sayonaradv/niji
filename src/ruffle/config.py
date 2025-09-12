import os
from dataclasses import dataclass

JIGSAW_DIR: str = "jigsaw-toxic-comment-classification-challenge"
ROOT_DIR: str = os.getcwd()


@dataclass
class Config:
    cache_dir: str = "./data"
    log_dir: str = "./runs"
    seed: int = 18


@dataclass
class DataConfig:
    batch_size: int = 64
    val_size: float = 0.2
    data_dir: str = f"./data/{JIGSAW_DIR}"


@dataclass
class ModuleConfig:
    num_labels: int = 6
    max_token_len: int = 256
    lr: float = 3e-5
    warmup_start_lr: float = 1e-5
    warmup_epochs: int = 5


@dataclass
class TrainerConfig:
    max_epochs: int = 20
    patience: int = 3
