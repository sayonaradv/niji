from dataclasses import dataclass
from os import cpu_count
from pathlib import Path
from typing import ClassVar


@dataclass
class Config:
    cache_dir: str | Path = "./data"
    log_dir: str | Path = "./mlruns"
    seed: int = 1234


@dataclass
class DataModuleConfig:
    dataset_name: str = "mat55555/jigsaw_toxic_comment"
    text_col: str = "text"
    label_cols: ClassVar[list[str]] = [
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate",
    ]
    val_size: float = 0.2
    max_token_len: int = 256
    batch_size: int = 64
    loader_columns: ClassVar[list[str]] = ["input_ids", "attention_mask", "labels"]
    train_split: str = "train"
    test_split: str = "test"
    num_workers: int | None = cpu_count()
    persistent_workers: bool = True


@dataclass
class ModuleConfig:
    model_name: str = "distilbert-base-cased"
    learning_rate: float = 3e-05


@dataclass
class TrainerConfig:
    accelerator: str = "auto"
    devices: int | str = "auto"
    strategy: str = "auto"
    precision: str | int = "auto"
    max_epochs: int = 5
    deterministic: bool = True
    log_model: str | bool = "all"
