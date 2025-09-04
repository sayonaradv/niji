"""Data loading utilities for the Jigsaw Toxic Comment Classification dataset.

This module provides PyTorch Dataset and PyTorch Lightning DataModule
implementations for loading and preprocessing the Jigsaw dataset for
multilabel toxicity classification tasks.
"""

import os
from dataclasses import dataclass
from enum import Enum

import lightning.pytorch as pl
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from ruffle.types import Batch

JIGSAW_LABELS: list[str] = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]
"""List of all available toxicity labels in the Jigsaw dataset."""


@dataclass
class SplitConfig:
    """Configuration for dataset split file paths.

    Attributes:
        inputs_path: Path to the input CSV file containing text data.
        labels_path: Optional path to the labels CSV file. None for training split
            where labels are included in the inputs file.
    """

    inputs_path: str
    labels_path: str | None = None


class Split(Enum):
    """Enumeration of available dataset splits.

    Attributes:
        TRAIN: Training split configuration with labels included in train.csv.
        TEST: Test split configuration with separate test.csv and test_labels.csv files.
    """

    TRAIN = SplitConfig(inputs_path="train.csv")
    TEST = SplitConfig(inputs_path="test.csv", labels_path="test_labels.csv")


class JigsawDataset(Dataset):
    """PyTorch Dataset for the Jigsaw Toxic Comment Classification dataset.

    Loads and preprocesses text comments with their corresponding toxicity labels
    for multilabel classification. Supports both training and test splits.

    Attributes:
        data_dir: Directory containing the dataset CSV files.
        labels: List of label names to include in the dataset.
        data: Pandas DataFrame containing the loaded and preprocessed data.
    """

    def __init__(
        self,
        split: Split,
        data_dir: str,
        labels: list[str] = JIGSAW_LABELS,
    ) -> None:
        """Initialize the JigsawDataset.

        Args:
            split: Dataset split to load (TRAIN or TEST).
            data_dir: Directory containing the Jigsaw dataset CSV files.
            labels: List of toxicity labels to include. Must be subset of JIGSAW_LABELS.

        Raises:
            FileNotFoundError: If data_dir doesn't exist.
            ValueError: If any labels are not found in the dataset.
        """
        self.data_dir: str = data_dir
        self.labels: list[str] = labels
        self._check_data_dir()
        self.data: pd.DataFrame = self.load_data(split, data_dir=self.data_dir)
        self._check_labels()

    def _check_data_dir(self) -> None:
        """Validate that the data directory exists.

        Raises:
            FileNotFoundError: If data_dir doesn't exist.
        """
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(
                f"Data directory not found: '{self.data_dir}'. "
                f"Please ensure the directory exists and contains the Jigsaw dataset files."
            )

    def _check_labels(self) -> None:
        """Validate that all requested labels exist in the dataset.

        Raises:
            ValueError: If any requested labels are not found in the data columns.
        """
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
        """Load and preprocess data for the specified split.

        For the training split, loads data directly from train.csv.
        For the test split, merges test.csv with test_labels.csv and filters
        out samples with missing labels (toxic == -1).

        Args:
            split: Dataset split to load (TRAIN or TEST).
            data_dir: Directory containing the dataset files.

        Returns:
            Pandas DataFrame with columns for 'comment_text' and label columns.
        """
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
        """Return the number of samples in the dataset.

        Returns:
            Number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> Batch:
        """Get a single sample from the dataset.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Dictionary containing:
                - 'text': Comment text as string.
                - 'labels': FloatTensor of binary labels for toxicity classification.
        """
        row: pd.Series = self.data.iloc[idx]
        return {
            "text": str(row["comment_text"]),
            "labels": torch.FloatTensor(row[self.labels].tolist()),
        }


class JigsawDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for the Jigsaw Toxic Comment Classification dataset.

    Handles data loading, preprocessing, and splitting for training, validation, and testing.
    Automatically splits the training data into train/validation sets and provides
    DataLoaders for all splits.

    Attributes:
        data_dir: Directory containing the dataset CSV files.
        labels: List of toxicity labels to include in the dataset.
        batch_size: Batch size for DataLoaders.
        val_size: Fraction of training data to use for validation.
        train_ds: Training dataset instance.
        val_ds: Validation dataset instance.
        test_ds: Test dataset instance.
    """

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
        """Initialize the JigsawDataModule.

        Args:
            data_dir: Directory containing dataset files. If None, uses default path
                'data/jigsaw-toxic-comment-classification-challenge'.
            labels: List of toxicity labels to include. Must be subset of JIGSAW_LABELS.
            batch_size: Batch size for all DataLoaders.
            val_size: Fraction of training data to use for validation (0.0 to 1.0).
        """
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
        """Set up datasets for the specified stage.

        Creates dataset instances for training/validation (stage='fit') or
        testing (stage='test'). For training, automatically splits the training
        data into train/validation sets.

        Args:
            stage: Stage to setup datasets for. Either 'fit', 'test', or None
                (which sets up both).
        """
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
        """Create DataLoader for training data.

        Returns:
            DataLoader for training with shuffling enabled and drop_last=True.

        Raises:
            RuntimeError: If train dataset hasn't been initialized via setup().
        """
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
        """Create DataLoader for validation data.

        Returns:
            DataLoader for validation without shuffling and drop_last=True.

        Raises:
            RuntimeError: If validation dataset hasn't been initialized via setup().
        """
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
        """Create DataLoader for test data.

        Returns:
            DataLoader for testing without shuffling and drop_last=True.

        Raises:
            RuntimeError: If test dataset hasn't been initialized via setup().
        """
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
