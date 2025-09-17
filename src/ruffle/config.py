import os

from pydantic import PositiveFloat, PositiveInt
from pydantic.dataclasses import dataclass

JIGSAW_DIR: str = "jigsaw-toxic-comment-classification-challenge"
ROOT_DIR: str = os.getcwd()


@dataclass
class Config:
    cache_dir: str = "./data"
    log_dir: str = "./runs"
    seed: PositiveInt = 18


@dataclass
class DataConfig:
    batch_size: PositiveInt = 64
    val_size: float = 0.2
    data_dir: str = f"./data/{JIGSAW_DIR}"


@dataclass
class ModuleConfig:
    num_labels: PositiveInt = 6
    max_token_len: PositiveInt = 256
    lr: PositiveFloat = 3e-5
    warmup_start_lr: PositiveFloat = 1e-5
    warmup_epochs: PositiveInt = 5


@dataclass
class TrainerConfig:
    max_epochs: PositiveInt = 20
    patience: PositiveInt = 3
