import os

from pydantic import Field, NonNegativeInt, PositiveFloat, PositiveInt
from pydantic.dataclasses import dataclass

JIGSAW_DIR: str = "jigsaw-toxic-comment-classification-challenge"
ROOT_DIR: str = os.getcwd()


@dataclass
class Config:
    log_dir: str = "./runs"
    seed: NonNegativeInt = 18


@dataclass
class DatasetConfig:
    batch_size: PositiveInt = 64
    # pyrefly: ignore  # no-matching-overload
    val_size: float = Field(default=0.2, ge=0, le=1)
    data_dir: str = f"./data/{JIGSAW_DIR}"
    labels: list[str] | None = None


@dataclass
class ModuleConfig:
    num_labels: PositiveInt = 6
    label_names: list[str] | None = None
    max_token_len: PositiveInt = 256
    lr: PositiveFloat = 3e-5
    warmup_start_lr: PositiveFloat = 1e-5
    warmup_epochs: PositiveInt = 5
    cache_dir: str | None = "./data"


@dataclass
class TrainerConfig:
    max_epochs: PositiveInt = 20
    patience: PositiveInt = 3
