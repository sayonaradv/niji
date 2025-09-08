import os
from typing import Annotated

from pydantic import Field, PositiveFloat, PositiveInt
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
    batch_size: PositiveInt = 64
    val_size: Annotated[PositiveFloat, Field(lt=1)] = 0.2


@dataclass
class ModelConfig:
    num_labels: PositiveInt = 6
    max_token_len: PositiveInt = 256
    lr: PositiveFloat = 3e-5
    warmup_start_lr: PositiveFloat = 3e-6
    warmup_epochs: PositiveInt = 5
    cache_dir: str = Config.cache_dir


@dataclass
class TrainingConfig:
    max_epochs: PositiveInt = 20
    accelerator: str = "auto"
    devices: str | int = "auto"
    precision: str = "16-mixed"
