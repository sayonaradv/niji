"""
Custom PyTorch Lightning DataModule for HuggingFace datasets with AutoTokenizer support.

This module provides:
- AutoTokenizerDataModule: a pl.LightningDataModule for managing splits and DataLoaders for any HuggingFace dataset.

Notes:
    - https://lightning.ai/docs/pytorch/stable/data/datamodule.html
    - https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset
"""

import os
from pathlib import Path
from typing import Any, cast

import lightning.pytorch as pl
import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from transformers import AutoTokenizer

from stormy.utils import combine_labels, get_num_workers


class AutoTokenizerDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_name: str,
        model_name: str,
        train_split: str,
        test_split: str,
        text_column: str,
        label_columns: list[str],
        loader_columns: list[str] | None = None,
        max_token_len: int = 256,
        val_size: float = 0.2,
        batch_size: int = 32,
        seed: int = 1234,
        cache_dir: str | Path = "data",
    ) -> None:
        """Initialize the AutoTokenizerDataModule.

        Args:
            dataset_name: The name of the dataset as given on HF datasets
            model_name: The name of the model and accompanying tokenizer
            train_split: The name of the training split as given on HF Hub
            test_split: The name of the test split as given on HF Hub
            text_column: The name of the text column in the dataset
            label_columns: List of label column names for multi-label classification
            loader_columns: The list of column names to pass to the HF dataset's .set_format method
            max_token_len: Maximum token length for tokenization
            val_size: The size of the validation split to create from training data
            batch_size: The batch size to pass to the PyTorch DataLoaders
            seed: The seed used for data splitting
            cache_dir: Directory to cache datasets and tokenizers

        Raises:
            ValueError: If val_size is not between 0 and 1
            ValueError: If max_token_len is not positive
            ValueError: If batch_size is not positive

        Notes:
            https://lightning.ai/docs/pytorch/stable/data/datamodule.html
        """
        super().__init__()

        # Validation
        if not 0 < val_size < 1:
            raise ValueError(f"val_size must be between 0 and 1, got {val_size}")
        if max_token_len <= 0:
            raise ValueError(f"max_token_len must be positive, got {max_token_len}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if not label_columns:
            raise ValueError("label_columns cannot be empty")

        self.dataset_name = dataset_name
        self.cache_dir = str(cache_dir) if isinstance(cache_dir, Path) else cache_dir
        self.text_column = text_column
        self.label_columns = label_columns
        self.num_labels = len(label_columns)
        self.model_name = model_name
        self.val_size = val_size
        self.max_token_len = max_token_len
        self.batch_size = batch_size
        self.train_split = train_split
        self.test_split = test_split
        self.seed = seed

        self.loader_columns = (
            ["input_ids", "attention_mask", "labels"]
            if loader_columns is None
            else loader_columns
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

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
            dataset_dict = dataset.train_test_split(
                test_size=self.val_size, seed=self.seed
            )

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
        if self.num_labels > 1:
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
    """Create a Jigsaw toxic comment classification DataModule with sensible defaults."""
    default_config = {
        "dataset_name": "mat55555/jigsaw_toxic_comment",
        "model_name": "prajjwal1/bert-tiny",
        "train_split": "train",
        "test_split": "test",
        "text_column": "text",
        "label_columns": ["toxic", "severe_toxic", "threat"],
    }
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
