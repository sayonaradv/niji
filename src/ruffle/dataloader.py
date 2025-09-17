import os
from enum import Enum

import lightning.pytorch as pl
import pandas as pd
import torch
from pydantic import ConfigDict, PositiveInt, validate_call
from torch.utils.data import DataLoader, Dataset, random_split

from ruffle.config import DataConfig
from ruffle.types import BATCH

JIGSAW_LABELS: list[str] = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]


class Split(Enum):
    TRAIN = "train"
    TEST = "test"

    @property
    def inputs_file(self) -> str:
        """Get the inputs CSV filename for this split."""
        return f"{self.value}.csv"

    @property
    def labels_file(self) -> str | None:
        """Get the labels CSV filename for this split, if separate."""
        return "test_labels.csv" if self == Split.TEST else None

    @validate_call(
        config=ConfigDict(arbitrary_types_allowed=True), validate_return=True
    )
    def load_data(self, data_dir: str) -> pd.DataFrame:
        """Load data for this split from the data directory."""
        inputs_path = os.path.join(data_dir, self.inputs_file)

        if self.labels_file is None:
            # Train data: everything in one file
            return pd.read_csv(inputs_path)
        else:
            # Test data: merge inputs and labels
            labels_path = os.path.join(data_dir, self.labels_file)
            inputs_df = pd.read_csv(inputs_path)
            labels_df = pd.read_csv(labels_path)
            return (
                inputs_df.merge(labels_df, on="id")
                .query("toxic != -1")
                .reset_index(drop=True)
            )


class JigsawDataset(Dataset):
    @validate_call(
        config=ConfigDict(validate_default=True, arbitrary_types_allowed=True)
    )
    def __init__(
        self,
        split: Split,
        data_dir: str = DataConfig.data_dir,
        labels: list[str] | None = None,
    ) -> None:
        self.data_dir: str = data_dir
        self._check_data_dir()

        self.data = split.load_data(self.data_dir)
        self.labels = labels or [
            col for col in self.data.columns if col in JIGSAW_LABELS
        ]
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

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> BATCH:
        row: pd.Series = self.data.iloc[idx]
        return {
            "text": str(row["comment_text"]),
            "labels": torch.FloatTensor(row[self.labels].tolist()),
        }


class JigsawDataModule(pl.LightningDataModule):
    @validate_call(config=ConfigDict(validate_default=True))
    def __init__(
        self,
        data_dir: str = DataConfig.data_dir,
        batch_size: PositiveInt = DataConfig.batch_size,
        val_size: float = DataConfig.val_size,
        labels: list[str] | None = None,
    ) -> None:
        super().__init__()

        self.data_dir = data_dir
        self.labels = labels or JIGSAW_LABELS
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
