"""PyTorch Lightning DataModule for HuggingFace datasets with AutoTokenizer support.

This module provides AutoTokenizerDataModule, a Lightning DataModule for managing
dataset splits and DataLoaders for any HuggingFace dataset used in sequence
classification tasks.

References:
    - https://lightning.ai/docs/pytorch/stable/data/datamodule.html
    - https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset
"""

import os
from typing import Any, cast

import lightning.pytorch as pl
import numpy as np
import torch
from datasets import Dataset, load_dataset
from pydantic import ValidationError
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from transformers import AutoTokenizer

from stormy.config import DataModuleConfig
from stormy.utils import combine_labels, get_num_workers


class AutoTokenizerDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for HuggingFace datasets with AutoTokenizer support.

    Provides a standardized interface for loading, preprocessing, and creating DataLoaders
    for HuggingFace datasets used in sequence classification tasks. Supports both single-label
    and multi-label classification with automatic tokenization and data splitting.

    Features:
        - Automatic dataset downloading and caching
        - Flexible train/validation/test split configuration
        - Built-in tokenization with customizable parameters
        - Multi-label classification support
        - Performance optimizations (multiprocessing, pin memory, persistent workers)
        - Comprehensive parameter validation using Pydantic

    Use cases:
        - Text classification (sentiment analysis, spam detection)
        - Multi-label classification (toxicity detection)
        - Fine-tuning transformer models on custom datasets
    """

    def __init__(
        self,
        dataset_name: str,
        model_name: str,
        train_split: str,
        test_split: str,
        text_column: str,
        label_columns: list[str],
        loader_columns: list[str] | None = None,
        max_token_len: int = 128,
        val_size: float = 0.2,
        batch_size: int = 32,
        cache_dir: str = "./data",
    ) -> None:
        """Initialize the AutoTokenizerDataModule.

        Args:
            dataset_name: Name of the HuggingFace dataset to load.
            model_name: Name of the pretrained HuggingFace model to use.
            train_split: Name of the dataset split to use for training.
            test_split: Name of the dataset split to use for testing.
            text_column: Name of the column containing input text data.
            label_columns: List of column names containing classification labels.
            loader_columns: List of dataset columns to include in DataLoaders.
                Defaults to ["input_ids", "attention_mask", "labels"].
            max_token_len: Maximum number of tokens per input sequence. Defaults to 128.
            val_size: Proportion of training data to use for validation. Defaults to 0.2.
            batch_size: Batch size for training and evaluation DataLoaders. Defaults to 32.
            cache_dir: Directory path where datasets and tokenizer files will be cached.
                Defaults to "./data".

        Raises:
            ValueError: If any parameter fails validation.
        """
        super().__init__()

        # Validate parameters using Pydantic
        try:
            config = DataModuleConfig(
                dataset_name=dataset_name,
                model_name=model_name,
                train_split=train_split,
                test_split=test_split,
                text_column=text_column,
                label_columns=label_columns,
                loader_columns=loader_columns
                if loader_columns is not None
                else ["input_ids", "attention_mask", "labels"],
                max_token_len=max_token_len,
                val_size=val_size,
                batch_size=batch_size,
                cache_dir=cache_dir,
            )
        except ValidationError as e:
            raise ValueError(
                f"Invalid configuration for AutoTokenizerDataModule: {e}"
            ) from e

        # Store validated configuration values
        self.dataset_name = config.dataset_name
        self.model_name = config.model_name
        self.train_split = config.train_split
        self.test_split = config.test_split
        self.text_column = config.text_column
        self.label_columns = config.label_columns
        self.loader_columns = config.loader_columns
        self.max_token_len = config.max_token_len
        self.val_size = config.val_size
        self.batch_size = config.batch_size
        self.cache_dir = config.cache_dir

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name, cache_dir=config.cache_dir
        )

        # Performance optimizations
        self.num_workers = get_num_workers()
        self.persistent_workers = True
        self.pin_memory = torch.cuda.is_available()

    def prepare_data(self) -> None:
        """Download and cache the dataset.

        Called only once per node to download the dataset to the cache directory.
        Disables tokenizer parallelism to avoid potential deadlocks during
        multiprocessing.

        Note:
            Called automatically by PyTorch Lightning trainer.
        """
        # Disable parallelism to avoid deadlocks during multiprocessing
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        load_dataset(self.dataset_name, cache_dir=self.cache_dir)

    def setup(self, stage: str | None) -> None:
        """Set up datasets for training, validation, and testing.

        Called on every GPU/process to load datasets, apply preprocessing,
        and set up data for the specified stage.

        Args:
            stage: Either 'fit' (for train/val), 'test', or None (for all stages).

        Raises:
            ValueError: If the loaded dataset is not a HuggingFace Dataset object.

        Note:
            Training data is automatically split into train/validation using val_size.
            All datasets are tokenized and formatted for PyTorch tensors.
        """
        if stage == "fit" or stage is None:
            dataset = load_dataset(
                self.dataset_name,
                cache_dir=self.cache_dir,
                split=self.train_split,
            )

            if not isinstance(dataset, Dataset):
                raise ValueError(f"Expected HuggingFace Dataset, got {type(dataset)}")

            dataset_dict = dataset.train_test_split(test_size=self.val_size)

            for split_name, split_dataset in dataset_dict.items():
                processed_dataset = split_dataset.map(
                    self.preprocess_data,
                    batched=True,
                    num_proc=self.num_workers,
                    desc=f"Tokenizing {split_name} split",
                )
                processed_dataset.set_format(type="torch", columns=self.loader_columns)

                if split_name == "train":
                    self.train_data = processed_dataset
                else:  # "test" becomes validation in this context
                    self.val_data = processed_dataset

        if stage == "test" or stage is None:
            test_dataset = load_dataset(
                self.dataset_name,
                cache_dir=self.cache_dir,
                split=self.test_split,
            )

            if not isinstance(test_dataset, Dataset):
                raise ValueError(
                    f"Expected HuggingFace Dataset, got {type(test_dataset)}"
                )

            self.test_data = test_dataset.map(
                self.preprocess_data,
                batched=True,
                num_proc=self.num_workers,
                desc="Tokenizing test split",
            )

            self.test_data.set_format(type="torch", columns=self.loader_columns)

    def preprocess_data(self, batch: dict[str, Any]) -> dict[str, list[Any]]:
        """Preprocess a batch of data by tokenizing text and combining labels.

        Args:
            batch: A batch dictionary from the HuggingFace dataset containing
                text data and labels according to the configured column names.

        Returns:
            Processed batch dictionary with tokenized inputs and combined labels.
            Contains keys like "input_ids", "attention_mask", and "labels".

        Note:
            Uses the configured max_token_len for truncation/padding and combines
            multiple label columns for multi-label classification scenarios.
        """
        inputs = self.tokenizer(
            batch[self.text_column],
            max_length=self.max_token_len,
            padding="max_length",
            truncation=True,
            return_tensors=None,
        )

        if len(self.label_columns) > 1:
            inputs["labels"] = combine_labels(batch, self.label_columns)
        else:
            inputs["labels"] = np.array(
                batch[self.label_columns[0]], dtype=float
            ).tolist()

        return inputs

    def _create_dataloader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        """Create a DataLoader with consistent configuration.

        Args:
            dataset: The HuggingFace dataset to wrap in a DataLoader.
            shuffle: Whether to shuffle the data. Defaults to False.

        Returns:
            Configured PyTorch DataLoader with performance optimizations enabled.
        """
        return DataLoader(
            cast(TorchDataset, dataset),
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            shuffle=shuffle,
        )

    def train_dataloader(self) -> DataLoader:
        """Return the training DataLoader.

        Returns:
            DataLoader for training data with shuffling enabled.

        Raises:
            RuntimeError: If called before setup('fit').
        """
        if not hasattr(self, "train_data") or self.train_data is None:
            raise RuntimeError("Training data not setup. Call setup('fit') first.")
        return self._create_dataloader(self.train_data, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """Return the validation DataLoader.

        Returns:
            DataLoader for validation data without shuffling.

        Raises:
            RuntimeError: If called before setup('fit').
        """
        if not hasattr(self, "val_data") or self.val_data is None:
            raise RuntimeError("Validation data not setup. Call setup('fit') first.")
        return self._create_dataloader(self.val_data, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        """Return the test DataLoader.

        Returns:
            DataLoader for test data without shuffling.

        Raises:
            RuntimeError: If called before setup('test').
        """
        if not hasattr(self, "test_data") or self.test_data is None:
            raise RuntimeError("Test data not setup. Call setup('test') first.")
        return self._create_dataloader(self.test_data, shuffle=False)


def create_jigsaw_datamodule(**kwargs) -> AutoTokenizerDataModule:
    """Create a Jigsaw toxic comment classification DataModule with sensible defaults.

    Convenience function providing pre-configured defaults for the Jigsaw Toxic Comment
    Classification dataset. The dataset contains Wikipedia comments labeled for various
    types of toxicity, making it ideal for multi-label classification experiments.

    Args:
        **kwargs: Parameters to override the defaults. Common overrides include:
            - model_name: Transformer model (default: "prajjwal1/bert-tiny")
            - max_token_len: Max sequence length
            - val_size: Validation split size
            - batch_size: Batch size
            - cache_dir: Cache directory

    Returns:
        AutoTokenizerDataModule configured for the Jigsaw dataset with all 6 toxicity
        labels (toxic, severe_toxic, obscene, threat, insult, identity_hate).

    Examples:
        Use with defaults:
        >>> dm = create_jigsaw_datamodule()

        Override parameters:
        >>> dm = create_jigsaw_datamodule(
        ...     model_name="distilbert-base-uncased",
        ...     max_token_len=256,
        ...     batch_size=16
        ... )

    Note:
        Uses bert-tiny model by default for fast experimentation.
    """
    default_config: dict[str, Any] = {
        "dataset_name": "mat55555/jigsaw_toxic_comment",
        "model_name": "prajjwal1/bert-tiny",
        "train_split": "train",
        "test_split": "test",
        "text_column": "text",
        "label_columns": [
            "toxic",
            "severe_toxic",
            "obscene",
            "threat",
            "insult",
            "identity_hate",
        ],
    }

    default_config.update(kwargs)
    return AutoTokenizerDataModule(**default_config)


if __name__ == "__main__":
    jigsaw_dm = create_jigsaw_datamodule()
    jigsaw_dm.prepare_data()
    jigsaw_dm.setup("fit")

    train_loader = jigsaw_dm.train_dataloader()
    batch = next(iter(train_loader))

    print(f"Loader columns: {jigsaw_dm.loader_columns}")

    print(f"DataModule: {jigsaw_dm}")
    print("Batch shapes:")
    for key, val in batch.items():
        if hasattr(val, "shape"):
            print(f"  {key}: {val.shape}")
        else:
            print(f"  {key}: {type(val)}")

    print(batch)
