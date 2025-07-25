"""
Custom PyTorch Lightning DataModule for HuggingFace datasets with AutoTokenizer support.

This module provides:
- AutoTokenizerDataModule: a pl.LightningDataModule for managing splits and DataLoaders for any HuggingFace dataset.

Notes:
    - https://lightning.ai/docs/pytorch/stable/data/datamodule.html
    - https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset
"""

import os
from typing import Any, cast

import lightning.pytorch as pl
import torch
from datasets import Dataset, load_dataset
from pydantic import ValidationError
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from transformers import AutoTokenizer

from stormy.config import DataModuleConfig
from stormy.utils import combine_labels, get_num_workers


class AutoTokenizerDataModule(pl.LightningDataModule):
    def __init__(self, **kwargs: Any) -> None:
        """Initializes the Lightning DataModule and validates configuration with Pydantic.

        Args:
            model_name (str): Name of the pretrained Hugging Face model to use (e.g., 'bert-base-uncased').
            dataset_name (str): Name of the Hugging Face dataset to load (e.g., 'imdb', 'ag_news').
            train_split (str): Name of the split to use for training.
            test_split (str): Name of the split to use for testing.
            text_column (str): Name of the column in the dataset that contains input text.
            label_columns (list[str]): List of column names containing the classification labels (must contain at least one)
            loader_columns (list[str], optional): List of dataset columns to be in the dataloaders. Defaults to ["input_ids", "attention_mask", "labels"].
            max_token_len (int, optional): Maximum number of tokens per input sequence (must be positive). Defaults to 128.
            val_size (float, optional): Proportion of training data to use for validation (must be between 0 and 1). Defaults to 0.2.
            batch_size (int, optional): Batch size to use for training and evaluation (must be positive)". Defaults to 32.
            cache_dir (str | Path, optional): Directory path to cache the dataset and tokenizer files. Defaults to `./data`.

        Notes:
            https://lightning.ai/docs/pytorch/stable/data/datamodule.html
        """
        super().__init__()
        try:
            config = DataModuleConfig(**kwargs)
        except ValidationError as e:
            raise ValueError(
                f"Invalid configuration for AutoTokenizerDataModule: {e}"
            ) from e

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

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name, cache_dir=config.cache_dir
        )

        # Performance optimizations
        self.num_workers = get_num_workers()
        self.persistent_workers = True
        self.pin_memory = torch.cuda.is_available()

    def prepare_data(self) -> None:
        """Download and cache the dataset.

        Notes:
            https://lightning.ai/docs/pytorch/stable/data/datamodule.html#prepare-data
        """
        # disable parrelism to avoid deadlocks
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        load_dataset(self.dataset_name, cache_dir=self.cache_dir)

    def setup(self, stage: str | None) -> None:
        """Set up datasets for training, validation, and testing.

        Args:
            stage: Either 'fit', 'test', or None (for both)

        Notes:
            https://lightning.ai/docs/pytorch/stable/data/datamodule.html#setup
        """
        if stage == "fit" or stage is None:
            dataset = load_dataset(
                self.dataset_name,
                cache_dir=self.cache_dir,
                split=self.train_split,
            )

            if not isinstance(dataset, Dataset):
                raise ValueError(f"Expected Dataset, got {type(dataset)}")

            # Split into train/validation
            dataset_dict = dataset.train_test_split(test_size=self.val_size)

            # Preprocess both splits
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
                else:  # "test" is validation in this context
                    self.val_data = processed_dataset

        if stage == "test" or stage is None:
            test_dataset = load_dataset(
                self.dataset_name,
                cache_dir=self.cache_dir,
                split=self.test_split,
            )

            if not isinstance(test_dataset, Dataset):
                raise ValueError(f"Expected Dataset, got {type(test_dataset)}")

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
            batch: A batch from the HuggingFace dataset

        Returns:
            Processed batch with tokenized inputs and combined labels
        """
        # Tokenize text
        inputs = self.tokenizer(
            batch[self.text_column],
            max_length=self.max_token_len,
            padding="max_length",
            truncation=True,
            return_tensors=None,  # Return as lists, not tensors
        )

        # Combine labels
        if len(self.label_columns) > 1:
            inputs["labels"] = combine_labels(batch, self.label_columns)
        else:
            inputs["labels"] = batch[self.label_columns[0]]

        return inputs

    def _create_dataloader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        """Create a DataLoader with consistent configuration

        Args:
            dataset: The dataset to create a DataLoader for
            shuffle: Whether to shuffle the data

        Returns:
            Configured DataLoader
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

        Notes:
            https://lightning.ai/docs/pytorch/stable/data/datamodule.html#train-dataloader
        """
        if self.train_data is None:
            raise RuntimeError("Training data not setup. Call setup('fit') first.")
        return self._create_dataloader(self.train_data, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """Return the validation DataLoader.

        Notes:
            https://lightning.ai/docs/pytorch/stable/data/datamodule.html#val-dataloader
        """
        if self.val_data is None:
            raise RuntimeError("Validation data not setup. Call setup('fit') first.")
        return self._create_dataloader(self.val_data, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        """Return the test DataLoader.

        Notes:
            https://lightning.ai/docs/pytorch/stable/data/datamodule.html#test-dataloader
        """
        if self.test_data is None:
            raise RuntimeError("Test data not setup. Call setup('test') first.")
        return self._create_dataloader(self.test_data, shuffle=False)


def create_jigsaw_datamodule(**kwargs) -> AutoTokenizerDataModule:
    """Create a Jigsaw toxic comment classification DataModule with sensible defaults.

    This function provides default configurations for the Jigsaw Toxic Comment dataset
    while allowing users to override any parameters they need to customize.

    Args:
        **kwargs: Any parameters to override the defaults. Common ones include:
            - model_name: The transformer model to use (default: "prajjwal1/bert-tiny")
            - max_token_len: Maximum token length (default: 256)
            - val_size: Validation split size (default: 0.2)
            - batch_size: Batch size for DataLoaders (default: 32)
            - seed: Random seed for reproducibility (default: 1234)
            - cache_dir: Directory to cache datasets (default: "data")

    Returns:
        AutoTokenizerDataModule: Configured data module for Jigsaw dataset

    Example:
        # Use all defaults
        dm = create_jigsaw_datamodule()

        # Override specific parameters
        dm = create_jigsaw_datamodule(
            model_name="distilbert-base-uncased",
            max_token_len=512,
            batch_size=16
        )
    """
    default_config = {
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

    # Update defaults with any user-provided overrides
    default_config.update(kwargs)
    return AutoTokenizerDataModule(**default_config)


if __name__ == "__main__":
    jigsaw_dm = create_jigsaw_datamodule()
    jigsaw_dm.prepare_data()
    jigsaw_dm.setup("fit")

    # Test train dataloader
    train_loader = jigsaw_dm.train_dataloader()
    batch = next(iter(train_loader))

    print(f"DataModule: {jigsaw_dm}")
    print("Batch shapes:")
    for key, val in batch.items():
        if hasattr(val, "shape"):
            print(f"  {key}: {val.shape}")
        else:
            print(f"  {key}: {type(val)}")

    print(batch)
