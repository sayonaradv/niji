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
from typing import cast

import lightning.pytorch as pl
import numpy as np
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from transformers import AutoTokenizer

from stormy.config import Config, DataModuleConfig, ModuleConfig


class AutoTokenizerDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_name: str = DataModuleConfig.dataset_name,
        cache_dir: str | Path = Config.cache_dir,
        text_col: str = DataModuleConfig.text_col,
        label_cols: list[str] = DataModuleConfig.label_cols,
        model_name: str = ModuleConfig.model_name,
        val_size: float = DataModuleConfig.val_size,
        max_token_len: int = DataModuleConfig.max_token_len,
        batch_size: int = DataModuleConfig.batch_size,
        loader_columns: list[str] = DataModuleConfig.loader_columns,
        train_split: str = DataModuleConfig.train_split,
        test_split: str = DataModuleConfig.test_split,
        num_workers: int | None = DataModuleConfig.num_workers,
        persistent_workers: bool = DataModuleConfig.persistent_workers,
        seed: int = Config.seed,
    ) -> None:
        """ "A custom PyTorch Lightning DataModule for HuggingFace datasets with AutoTokenizer support."

        Args:
            dataset_name: the name of the dataset as given on HF datasets
            cache_dir: corresponds to HF datasets.load_dataset
            text_col: the name of the text column in the dataset
            label_cols: list of label column names for multi-label classification
            model_name: the name of the model and accompanying tokenizer
            val_size: the size of the validation split to create from training data
            max_token_len: maximum token length for tokenization
            batch_size: the batch size to pass to the PyTorch DataLoaders
            loader_columns: the list of column names to pass to the HF dataset's .set_format method
            train_split: the name of the training split as given on HF Hub
            test_split: the name of the test split as given on HF Hub
            num_workers: corresponds to torch.utils.data.DataLoader
            persistent_workers: whether to keep workers alive between epochs
            seed: the seed used for data splitting

        Notes:
            https://lightning.ai/docs/pytorch/stable/data/datamodule.html
        """
        super().__init__()
        self.dataset_name = dataset_name
        self.cache_dir = str(cache_dir) if isinstance(cache_dir, Path) else cache_dir
        self.text_col = text_col
        self.label_cols = label_cols
        self.num_labels = len(label_cols)
        self.model_name = model_name
        self.val_size = val_size
        self.max_token_len = max_token_len
        self.batch_size = batch_size
        self.loader_columns = loader_columns
        self.train_split = train_split
        self.test_split = test_split
        self.num_workers = num_workers if num_workers is not None else 0
        self.persistent_workers = persistent_workers
        self.seed = seed

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, cache_dir=cache_dir, use_fast=True
        )

    def prepare_data(self) -> None:
        """
        Notes:
            https://lightning.ai/docs/pytorch/stable/data/datamodule.html#prepare-data
        """
        # disable parrelism to avoid deadlocks
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        load_dataset(self.dataset_name, cache_dir=self.cache_dir)
        AutoTokenizer.from_pretrained(
            self.model_name, cache_dir=self.cache_dir, use_fast=True
        )

    def setup(self, stage: str) -> None:
        """
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
                raise ValueError()

            dataset = dataset.train_test_split(test_size=self.val_size, seed=self.seed)
            for split in dataset:
                dataset[split] = dataset[split].map(
                    self.convert_to_features,
                    batched=True,
                    num_proc=self.num_workers,
                )
                dataset[split].set_format(type="torch", columns=self.loader_columns)
            self.train_data = dataset["train"]
            self.val_data = dataset["test"]
            del dataset

        if stage == "test" or stage is None:
            self.test_data = load_dataset(
                self.dataset_name,
                cache_dir=self.cache_dir,
                split=self.test_split,
            )

            if not isinstance(self.test_data, Dataset):
                raise ValueError()

            self.test_data.map(
                self.convert_to_features,
                batched=True,
                num_proc=self.num_workers,
            )
            self.test_data.set_format(type="torch", columns=self.loader_columns)

    def convert_to_features(self, batch: dict[str, list]) -> dict[str, list]:
        """
        Labels are explicitly converted to float to ensure compatibility with BCEWithLogitsLoss,
        which requires target tensors to be of floating point type (float32), not integer/long.
        """
        features = self.tokenizer.batch_encode_plus(
            batch[self.text_col],
            max_length=self.max_token_len,
            padding="max_length",
            truncation=True,
        )
        if self.num_labels > 1:
            # Vectorized label conversion
            features["labels"] = np.stack(
                [np.array(batch[col], dtype=np.float32) for col in self.label_cols],
                axis=1,
            )
        else:
            label_col = self.label_cols[0]
            features["labels"] = np.array(batch[label_col], dtype=np.float32)
        return features

    def train_dataloader(self) -> DataLoader:
        """
        Notes:
            https://lightning.ai/docs/pytorch/stable/data/datamodule.html#train-dataloader
        """
        return DataLoader(
            cast(TorchDataset, self.train_data),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Notes:
            https://lightning.ai/docs/pytorch/stable/data/datamodule.html#val-dataloader
        """
        return DataLoader(
            cast(TorchDataset, self.val_data),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self) -> DataLoader:
        """
        Notes:
            https://lightning.ai/docs/pytorch/stable/data/datamodule.html#test-dataloader
        """
        return DataLoader(
            cast(TorchDataset, self.test_data),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )


if __name__ == "__main__":
    """
    Example usage for testing the AutoTokenizerDataModule.
    Instantiates the DataModule, prepares data, sets up splits, and prints a sample batch.
    """
    dm = AutoTokenizerDataModule()
    dm.prepare_data()
    dm.setup("fit")
    batch = next(iter(dm.train_dataloader()))
    for key, val in batch.items():
        print(f"{key}: {val.shape}")
    print(batch)
