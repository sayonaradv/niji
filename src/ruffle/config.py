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
