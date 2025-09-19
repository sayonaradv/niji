import os
from enum import Enum

import lightning.pytorch as pl
import pandas as pd
import torch
from pydantic import ConfigDict, Field, PositiveInt, validate_call
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
"""List of available Jigsaw toxicity classification labels.

This constant defines the standard set of toxicity labels used in the Jigsaw
dataset for multi-label classification tasks.
"""


class Split(Enum):
    """Enumeration for dataset splits.

    Defines the available dataset splits and provides methods to load
    data for each split with appropriate file handling.
    """

    TRAIN = "train"
    TEST = "test"

    @property
    def inputs_file(self) -> str:
        """Get the inputs CSV filename for this split.

        Returns:
            The CSV filename containing input data for this split.
        """
        return f"{self.value}.csv"

    @property
    def labels_file(self) -> str | None:
        """Get the labels CSV filename for this split, if separate.

        Returns:
            The CSV filename containing labels if they are stored separately
            from inputs (TEST split), otherwise None (TRAIN split).
        """
        return "test_labels.csv" if self == Split.TEST else None

    @validate_call(
        config=ConfigDict(arbitrary_types_allowed=True), validate_return=True
    )
    def load_data(self, data_dir: str) -> pd.DataFrame:
        """Load data for this split from the data directory.

        Loads and processes data according to the split type. For training data,
        loads from a single CSV file. For test data, merges inputs and labels
        from separate files and filters out invalid entries.

        Args:
            data_dir: Path to the directory containing the dataset files.

        Returns:
            A pandas DataFrame containing the loaded and processed data.

        Raises:
            FileNotFoundError: If required CSV files are not found in data_dir.
        """
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
    """PyTorch Dataset for the Jigsaw Toxicity Classification dataset.

    Provides access to text comments and their corresponding toxicity labels
    for multi-label classification tasks. Supports both training and test splits
    with configurable label subsets.

    Attributes:
        data_dir: Path to directory containing dataset files.
        data: Loaded pandas DataFrame containing the dataset.
        labels: List of label column names to include in outputs.
    """

    @validate_call(
        config=ConfigDict(validate_default=True, arbitrary_types_allowed=True)
    )
    def __init__(
        self,
        split: Split,
        data_dir: str = DataConfig.data_dir,
        labels: list[str] | None = None,
    ) -> None:
        """Initialize the Jigsaw dataset.

        Args:
            split: Which dataset split to load (TRAIN or TEST).
            data_dir: Path to directory containing dataset CSV files.
                Defaults to value from DataConfig.
            labels: List of label column names to include. If None, uses all
                available Jigsaw labels found in the dataset.

        Raises:
            FileNotFoundError: If the data directory doesn't exist.
            ValueError: If specified labels are not found in the dataset.
        """
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
        """Get the number of samples in the dataset.

        Returns:
            The total number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> BATCH:
        """Get a single sample from the dataset.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            A dictionary containing:
                - 'text': The comment text as a string.
                - 'labels': A FloatTensor containing the toxicity labels.
        """
        row: pd.Series = self.data.iloc[idx]
        return {
            "text": str(row["comment_text"]),
            "labels": torch.FloatTensor(row[self.labels].tolist()),
        }


class JigsawDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for the Jigsaw Toxicity dataset.

    Handles data loading, splitting, and DataLoader creation for training,
    validation, and testing. Automatically splits training data into train/val
    sets and provides configured DataLoaders for each phase.

    Attributes:
        data_dir: Path to directory containing dataset files.
        labels: List of label column names to include.
        batch_size: Batch size for DataLoaders.
        val_size: Fraction of training data to use for validation.
    """

    @validate_call(config=ConfigDict(validate_default=True))
    def __init__(
        self,
        data_dir: str = DataConfig.data_dir,
        batch_size: PositiveInt = DataConfig.batch_size,
        # pyrefly: ignore  # no-matching-overload
        val_size: float = Field(DataConfig.val_size, ge=0, le=1),
        labels: list[str] | None = None,
    ) -> None:
        """Initialize the Jigsaw DataModule.

        Args:
            data_dir: Path to directory containing dataset CSV files.
                Defaults to value from DataConfig.
            batch_size: Batch size for all DataLoaders. Must be positive.
                Defaults to value from DataConfig.
            val_size: Fraction of training data to use for validation.
                Must be between 0 and 1. Defaults to value from DataConfig.
            labels: List of label column names to include. If None, uses all
                available Jigsaw labels.
        """
        super().__init__()

        self.data_dir = data_dir
        self.labels = labels or JIGSAW_LABELS
        self.batch_size = batch_size
        self.val_size = val_size

        self.train_ds: Dataset | None = None
        self.val_ds: Dataset | None = None
        self.test_ds: Dataset | None = None

    def setup(self, stage: str | None = None) -> None:
        """Set up datasets for the specified stage.

        Creates train/validation datasets for 'fit' stage and test dataset
        for 'test' stage. If stage is None, sets up datasets for all stages.

        Args:
            stage: The training stage ('fit', 'test', or None for all stages).
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

    def train_dataloader(self) -> DataLoader | None:
        """Create the training DataLoader.

        Returns:
            A DataLoader configured for training with shuffling enabled
            and drop_last=True for consistent batch sizes, or None if
            the training dataset has not been initialized yet.
        """
        return (
            None
            if self.train_ds is None
            else DataLoader(
                self.train_ds,
                batch_size=self.batch_size,
                drop_last=True,
                shuffle=True,
            )
        )

    def val_dataloader(self) -> DataLoader | None:
        """Create the validation DataLoader.

        Returns:
            A DataLoader configured for validation with no shuffling
            and drop_last=True for consistent batch sizes, or None if
            the validation dataset has not been initialized yet.
        """
        return (
            None
            if self.val_ds is None
            else DataLoader(
                self.val_ds,
                batch_size=self.batch_size,
                drop_last=True,
                shuffle=False,
            )
        )

    def test_dataloader(self) -> DataLoader | None:
        """Create the test DataLoader.

        Returns:
            A DataLoader configured for testing with no shuffling
            and drop_last=True for consistent batch sizes, or None if
            the test dataset has not been initialized yet.
        """
        return (
            None
            if self.test_ds is None
            else DataLoader(
                self.test_ds,
                batch_size=self.batch_size,
                drop_last=True,
                shuffle=False,
            )
        )
