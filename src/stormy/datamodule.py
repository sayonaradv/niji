"""
Custom PyTorch Lightning DataModule for the Jigsaw Toxic Comment Classification dataset.

This module provides:
- JigsawDataModule: a pl.LightningDataModule for managing splits and DataLoaders for the Jigsaw dataset.

Notes:
    - https://lightning.ai/docs/pytorch/stable/data/datamodule.html
    - https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset
"""

import os
from typing import ClassVar

import lightning.pytorch as pl
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


class JigsawDataModule(pl.LightningDataModule):
    """
    Custom PyTorch LightningDataModule for the Jigsaw Toxic Comment Classification dataset.
    Handles data preparation, setup, and DataLoader creation for training, validation, and testing.

    Notes:
        https://lightning.ai/docs/pytorch/stable/data/datamodule.html
    """

    dataset_name: ClassVar[str] = "mat55555/jigsaw_toxic_comment"
    loader_columns: ClassVar[list[str]] = [
        "datasets_idx",
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "start_positions",
        "end_positions",
        "labels",
    ]

    def __init__(
        self,
        model_name_or_path: str = "distilbert-base-cased",
        cache_dir: str | None = "data",
        labels: list[str] = LABELS,
        val_size: float = 0.2,
        max_token_len: int = 128,
        batch_size: int = 32,
        seed: int = 18,
    ) -> None:
        """
        Args:
            model_name_or_path: Name or path of the pretrained model for tokenization.
            cache_dir: Path to the directory for caching datasets and tokenizers. If None, uses the default HuggingFace cache directories (usually ~/.cache/huggingface/datasets and ~/.cache/huggingface/transformers).
            labels: List of class/label names.
            val_size: Fraction of training data to use for validation (default: 0.2).
            max_token_len: Maximum token length for tokenization (default: 128).
            batch_size: Batch size for DataLoaders (default: 32).
            seed: Random seed for splitting the data (default: 18).
        """
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.cache_dir = cache_dir
        self.labels = labels
        self.num_labels = len(labels)
        self.val_size = val_size
        self.max_token_len = max_token_len
        self.batch_size = batch_size
        self.seed = seed
        self.num_workers = os.cpu_count()
        self.persistent_workers = True

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path, cache_dir=cache_dir, use_fast=True
        )

    def setup(self, stage: str | None = None) -> None:
        """
        Sets up datasets for different stages ('fit', 'validate', 'test').

        Loads the Jigsaw dataset, splits the training set into train/validation,
        applies tokenization and formatting, and prepares the datasets for use
        in DataLoaders.

        Args:
            stage: One of 'fit', 'validate', 'test', or None. If None, sets up all splits.

        Notes:
            https://lightning.ai/docs/pytorch/stable/data/datamodule.html#setup
        """
        if stage == "fit" or stage is None:
            dataset = load_dataset(
                self.dataset_name, cache_dir=self.cache_dir, split="train"
            )
            dataset = dataset.train_test_split(test_size=self.val_size, seed=self.seed)
            for split in dataset:
                dataset[split] = dataset[split].map(
                    self.convert_to_features,
                    remove_columns=["label"],
                    batched=True,
                    num_proc=self.num_workers,
                )
                columns = [
                    c for c in dataset[split].column_names if c in self.loader_columns
                ]
                dataset[split].set_format(type="torch", columns=columns)
            self.train_data = dataset["train"]
            self.val_data = dataset["test"]
            del dataset

        if stage == "test" or stage is None:
            self.test_data = load_dataset(
                self.dataset_name, cache_dir=self.cache_dir, split="test"
            )
            self.test_data.map(
                self.convert_to_features,
                remove_columns=["label"],
                batched=True,
                num_proc=self.num_workers,
            )
            columns = [
                c for c in self.test_data.column_names if c in self.loader_columns
            ]
            self.test_data.set_format(type="torch", columns=columns)

    def prepare_data(self) -> None:
        """
        Downloads and caches the Jigsaw dataset and tokenizer if not already present.

        Also disables parallelism for tokenizers to avoid deadlocks in some environments.

        If cache_dir is None, uses the default HuggingFace cache directories (usually ~/.cache/huggingface/datasets and ~/.cache/huggingface/transformers).
        """
        # disable parrelism to avoid deadlocks
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        load_dataset(self.dataset_name, cache_dir=self.cache_dir)
        AutoTokenizer.from_pretrained(
            self.model_name_or_path, cache_dir=self.cache_dir, use_fast=True
        )

    def convert_to_features(self, example_batch: dict[str, list]) -> dict[str, list]:
        """
        Tokenizes a batch of examples and converts labels to multi-label format if needed.

        Args:
            example_batch: A batch of examples from the dataset, with 'text' and label columns.

        Returns:
            A dictionary with tokenized features and targets.

        Notes:
            Labels are explicitly converted to float to ensure compatibility with BCEWithLogitsLoss,
            which requires target tensors to be of floating point type (float32), not integer/long.
        """
        features = self.tokenizer.batch_encode_plus(
            example_batch["text"],
            max_length=self.max_token_len,
            padding="max_length",
            truncation=True,
        )
        if self.num_labels > 1:
            # Vectorized label conversion
            features["labels"] = np.stack(
                [np.array(example_batch[col], dtype=np.float32) for col in self.labels],
                axis=1,
            )
        else:
            label_col = self.labels[0]
            features["labels"] = np.array(example_batch[label_col], dtype=np.float32)
        return features

    def train_dataloader(self) -> DataLoader:
        """
        Returns:
            DataLoader: DataLoader for the training split.

        Notes:
            https://lightning.ai/docs/pytorch/stable/data/datamodule.html#train-dataloader
        """
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Returns:
            DataLoader: DataLoader for the validation split.

        Notes:
            https://lightning.ai/docs/pytorch/stable/data/datamodule.html#val-dataloader
        """
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self) -> DataLoader:
        """
        Returns:
            DataLoader: DataLoader for the test split.

        Notes:
            https://lightning.ai/docs/pytorch/stable/data/datamodule.html#test-dataloader
        """
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )


if __name__ == "__main__":
    """
    Example usage for testing the JigsawDataModule.
    Instantiates the DataModule, prepares data, sets up splits, and prints a sample batch.
    """
    dm = JigsawDataModule()
    dm.prepare_data()
    dm.setup("fit")
    batch = next(iter(dm.train_dataloader()))
    for key, val in batch.items():
        print(f"{key}: {val.shape}")
    print(batch)
