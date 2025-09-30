import os
from enum import Enum
from typing import Annotated

import lightning.pytorch as pl
import pandas as pd
import torch
from pydantic import ConfigDict, Field, NonNegativeInt, PositiveInt, validate_call
from torch.utils.data import DataLoader, Dataset, random_split

from blanki.types import BATCH

JIGSAW_LABELS: list[str] = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]

JIGSAW_HANDLE: str = "jigsaw-toxic-comment-classification-challenge"
DATA_DIR: str = f"./data/{JIGSAW_HANDLE}"


class Split(Enum):
    """Enumeration for dataset splits.

    Defines the available dataset splits and provides methods to load
    data for each split with appropriate file handling.

    Attributes:
        TRAIN: Training split identifier.
        TEST: Test split identifier.

    Example:
        >>> split = Split.TRAIN
        >>> data = split.load_data("/path/to/jigsaw/data")
    """

    TRAIN = "train"
    TEST = "test"

    @property
    def inputs_file(self) -> str:
        """Get the inputs CSV filename for this split.

        Returns:
            str: The CSV filename containing input data for this split.
                Returns "train.csv" for TRAIN split and "test.csv" for TEST split.
        """
        return f"{self.value}.csv"

    @property
    def labels_file(self) -> str | None:
        """Get the labels CSV filename for this split, if separate.

        Returns:
            str | None: The CSV filename containing labels if they are stored separately
                from inputs (TEST split returns "test_labels.csv"), otherwise None
                (TRAIN split where labels are in the same file as inputs).
        """
        return "test_labels.csv" if self == Split.TEST else None

    @validate_call(
        config=ConfigDict(arbitrary_types_allowed=True), validate_return=True
    )
    def load_data(self, data_dir: str) -> pd.DataFrame:
        """Load data for this split from the data directory.

        Loads and processes data according to the split type. For training data,
        loads from a single CSV file. For test data, merges inputs and labels
        from separate files and filters out invalid entries (where toxic == -1).

        Args:
            data_dir (str): Path to the directory containing the dataset files.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the loaded and processed data
                with columns including 'id', 'comment_text', and toxicity label columns.

        Raises:
            FileNotFoundError: If required CSV files are not found in data_dir.
        """
        inputs_path: str = os.path.join(data_dir, self.inputs_file)

        if self.labels_file is None:
            # Train data: everything in one file
            return pd.read_csv(inputs_path)
        else:
            # Test data: merge inputs and labels
            labels_path: str = os.path.join(data_dir, self.labels_file)
            inputs_df: pd.DataFrame = pd.read_csv(inputs_path)
            labels_df: pd.DataFrame = pd.read_csv(labels_path)
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

    The dataset expects CSV files with 'comment_text' column for inputs and
    toxicity label columns ('toxic', 'severe_toxic', etc.) for targets.

    Attributes:
        data_dir (str): Path to directory containing dataset files.
        data (pd.DataFrame): Loaded pandas DataFrame containing the dataset.
        labels (list[str]): List of label column names to include in outputs.

    Example:
        >>> dataset = JigsawDataset(
        ...     split=Split.TRAIN,
        ...     data_dir="/path/to/jigsaw/data",
        ...     labels=["toxic", "severe_toxic"]
        ... )
        >>> sample = dataset[0]
        >>> print(sample["text"])  # Comment text
        >>> print(sample["labels"])  # FloatTensor of toxicity labels
    """

    @validate_call(
        config=ConfigDict(validate_default=True, arbitrary_types_allowed=True)
    )
    def __init__(
        self,
        split: Split,
        data_dir: str,
        labels: list[str] | None = None,
    ) -> None:
        """Initialize the Jigsaw dataset.

        Args:
            split (Split): Which dataset split to load (Split.TRAIN or Split.TEST).
            data_dir (str): Path to directory containing dataset CSV files.
                Should contain files like "train.csv", "test.csv", and "test_labels.csv".
            labels (list[str] | None): List of label column names to include in outputs.
                If None, uses all available Jigsaw labels found in the dataset
                (toxic, severe_toxic, obscene, threat, insult, identity_hate).

        Raises:
            FileNotFoundError: If the data directory doesn't exist or required CSV files
                are missing.
            ValueError: If specified labels are not found in the dataset columns.
        """
        self.data_dir = data_dir
        self._check_data_dir()

        self.data = split.load_data(self.data_dir)
        self.labels = JIGSAW_LABELS if labels is None else labels
        self._check_labels()

    def _check_data_dir(self) -> None:
        """Validate that the data directory exists.

        Raises:
            FileNotFoundError: If the data directory doesn't exist.
        """
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(
                f"Data directory not found: '{self.data_dir}'. "
                f"Please ensure the directory exists and contains the Jigsaw dataset files."
            )

    def _check_labels(self) -> None:
        """Validate that all specified labels exist in the dataset.

        Raises:
            ValueError: If any specified labels are not found in the dataset columns.
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

    def __len__(self) -> int:
        """Get the number of samples in the dataset.

        Returns:
            int: The total number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> BATCH:
        """Get a single sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            BATCH: A dictionary containing:
                - 'text' (str): The comment text as a string.
                - 'labels' (torch.FloatTensor): A tensor containing the toxicity labels
                  with shape (num_labels,) and values of 0.0 or 1.0.

        Raises:
            IndexError: If idx is out of range for the dataset.
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
    sets using the specified validation fraction and provides configured
    DataLoaders for each phase.

    The DataModule expects the following file structure in data_dir:
        - train.csv: Training data with 'comment_text' and label columns
        - test.csv: Test inputs with 'comment_text' column
        - test_labels.csv: Test labels with 'id' and label columns

    Example:
        >>> dm = JigsawDataModule(
        ...     data_dir="/path/to/jigsaw/data",
        ...     batch_size=32,
        ...     val_size=0.2,
        ...     labels=["toxic", "severe_toxic"]
        ... )
        >>> dm.setup("fit")
        >>> train_loader = dm.train_dataloader()
        >>> val_loader = dm.val_dataloader()
    """

    @validate_call(config=ConfigDict(validate_default=True))
    def __init__(
        self,
        data_dir: str,
        batch_size: PositiveInt = 64,
        val_size: Annotated[float, Field(ge=0, le=1)] = 0.2,
        labels: list[str] | None = None,
        num_workers: NonNegativeInt | None = None,
    ) -> None:
        """Initialize the Jigsaw DataModule.

        Args:
            data_dir (str): Path to directory containing dataset CSV files.
                Should contain "train.csv", "test.csv", and "test_labels.csv".
            batch_size (PositiveInt): Batch size for all DataLoaders. Must be positive.
            val_size (float): Fraction of training data to use for validation.
                Must be between 0.0 and 1.0 (inclusive). For example, 0.2 means
                20% of training data will be used for validation.
            labels (list[str] | None): List of label column names to include in outputs.
                If None, uses all available Jigsaw labels (toxic, severe_toxic, obscene,
                threat, insult, identity_hate).
            num_workers (NonNegativeInt | None): Number of worker processes for data loading.
                If None, defaults to the number of CPU cores. If 0, uses single-threaded
                data loading. Must be non-negative.
        """
        super().__init__()

        self.data_dir = data_dir
        self.labels = JIGSAW_LABELS if labels is None else labels
        self.batch_size = batch_size
        self.val_size = val_size
        self.num_workers = (
            num_workers if num_workers is not None else os.cpu_count() or 1
        )

        self.train_ds: Dataset | None = None
        self.val_ds: Dataset | None = None
        self.test_ds: Dataset | None = None

    def setup(self, stage: str | None = None) -> None:
        """Set up datasets for the specified stage.

        Creates train/validation datasets for 'fit' stage and test dataset
        for 'test' stage. If stage is None, sets up datasets for all stages.
        The training dataset is automatically split into train and validation
        sets using the val_size fraction.

        Args:
            stage (str | None): The training stage to set up datasets for.
                Options are:
                - 'fit': Sets up train and validation datasets
                - 'test': Sets up test dataset
                - None: Sets up datasets for all stages
        """
        if stage == "fit" or stage is None:
            lengths: list[float] = [1 - self.val_size, self.val_size]
            full_train_ds: Dataset = JigsawDataset(
                Split.TRAIN, data_dir=self.data_dir, labels=self.labels
            )
            self.train_ds, self.val_ds = random_split(full_train_ds, lengths)

        if stage == "test" or stage is None:
            self.test_ds: Dataset = JigsawDataset(
                Split.TEST, data_dir=self.data_dir, labels=self.labels
            )

    def _make_dataloader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        """Make a DataLoader from a PyTorch Dataset.

        Args:
            dataset (Dataset): The PyTorch Dataset to make a DataLoader from.
            shuffle (bool): Whether to shuffle the dataset.

        Returns:
            DataLoader: A DataLoader configured for the dataset.
        """
        persistent_workers: bool = self.num_workers > 0
        return DataLoader(
            dataset,
            drop_last=True,
            shuffle=shuffle,
            persistent_workers=persistent_workers,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def train_dataloader(self) -> DataLoader | None:
        """Create the training DataLoader.

        Returns:
            DataLoader | None: A DataLoader configured for training with shuffling
                enabled and drop_last=True for consistent batch sizes. Returns None
                if the training dataset has not been initialized (setup() not called
                with 'fit' stage).
        """
        return (
            None
            if self.train_ds is None
            else self._make_dataloader(self.train_ds, shuffle=True)
        )

    def val_dataloader(self) -> DataLoader | None:
        """Create the validation DataLoader.

        Returns:
            DataLoader | None: A DataLoader configured for validation with no shuffling
                and drop_last=True for consistent batch sizes. Returns None if the
                validation dataset has not been initialized (setup() not called with
                'fit' stage).
        """
        return (
            None
            if self.val_ds is None
            else self._make_dataloader(self.val_ds, shuffle=False)
        )

    def test_dataloader(self) -> DataLoader | None:
        """Create the test DataLoader.

        Returns:
            DataLoader | None: A DataLoader configured for testing with no shuffling
                and drop_last=True for consistent batch sizes. Returns None if the
                test dataset has not been initialized (setup() not called with 'test'
                stage).
        """
        return (
            None
            if self.test_ds is None
            else self._make_dataloader(self.test_ds, shuffle=False)
        )
