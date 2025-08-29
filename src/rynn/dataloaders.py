import os
from dataclasses import dataclass
from enum import Enum

import lightning.pytorch as pl
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from rynn.types import Batch

JIGSAW_LABELS: list[str] = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]


@dataclass
class SplitConfig:
    inputs_path: str
    labels_path: str | None = None


class Split(Enum):
    TRAIN = SplitConfig(inputs_path="train.csv")
    TEST = SplitConfig(inputs_path="test.csv", labels_path="test_labels.csv")


class JigsawDataset(Dataset):
    def __init__(
        self,
        split: Split,
        data_dir: str,
        labels: list[str] = JIGSAW_LABELS,
    ) -> None:
        self.data_dir: str = data_dir
        self.labels: list[str] = labels
        self._check_data_dir()
        self.data: pd.DataFrame = self.load_data(split, data_dir=self.data_dir)
        self._check_labels()

    def _check_data_dir(self) -> None:
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(
                f"Data directory not found: '{self.data_dir}'. "
                f"Please ensure the directory exists and contains the Jigsaw dataset files."
            )

    def _check_labels(self) -> None:
        missing_labels: list[str] = [
            label for label in self.labels if label not in self.data.columns
        ]
        if missing_labels:
            available_labels: list[str] = [
                col for col in self.data.columns if col in JIGSAW_LABELS
            ]
            raise ValueError(
                f"Labels {missing_labels} not found in dataset. "
                f"Available labels: {available_labels}"
            )

    def load_data(self, split: Split, data_dir: str) -> pd.DataFrame:
        if split.value.labels_path is None:
            return pd.read_csv(os.path.join(data_dir, split.value.inputs_path))
        else:
            df1: pd.DataFrame = pd.read_csv(
                os.path.join(data_dir, split.value.inputs_path)
            )
            df2: pd.DataFrame = pd.read_csv(
                os.path.join(data_dir, split.value.labels_path)
            )
            return df1.merge(df2, on="id").query("toxic != -1").reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Batch:
        row: pd.Series = self.data.iloc[idx]
        return {
            "text": str(row["comment_text"]),
            "labels": torch.FloatTensor(row[self.labels].tolist()),
        }


class JigsawDataModule(pl.LightningDataModule):
    _DEFAULT_DATA_DIR: str = os.path.join(
        "data", "jigsaw-toxic-comment-classification-challenge"
    )

    def __init__(
        self,
        data_dir: str | None = None,
        labels: list[str] = JIGSAW_LABELS,
        batch_size: int = 64,
        val_size: float = 0.2,
    ) -> None:
        super().__init__()
        self.data_dir: str = (
            data_dir if data_dir is not None else self._DEFAULT_DATA_DIR
        )
        self.labels = labels
        self.batch_size = batch_size
        self.val_size = val_size

        self.train_ds: Dataset | None = None
        self.val_ds: Dataset | None = None
        self.test_ds: Dataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit" or stage is None:
            lengths: list[float] = [1 - self.val_size, self.val_size]
            full_train_ds: Dataset = JigsawDataset(
                Split.TRAIN, data_dir=self.data_dir, labels=self.labels
            )
            self.train_ds, self.val_ds = random_split(full_train_ds, lengths)
        if stage == "test" or stage is None:
            self.test_ds = JigsawDataset(
                Split.TEST, data_dir=self.data_dir, labels=self.labels
            )

    def train_dataloader(self) -> DataLoader:
        if self.train_ds is None:
            raise RuntimeError(
                "Train dataset has not been initialized. "
                "Did you forget to call `setup('fit')`?"
            )
        return DataLoader(
            self.train_ds,
            shuffle=True,
            batch_size=self.batch_size,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_ds is None:
            raise RuntimeError(
                "Validation dataset has not been initialized. "
                "Did you forget to call `setup('fit')`?"
            )
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            drop_last=True,
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_ds is None:
            raise RuntimeError(
                "Test dataset has not been initialized. "
                "Did you forget to call `setup('test')`?"
            )
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            drop_last=True,
        )
