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

    This DataModule provides a standardized interface for loading, preprocessing, and creating
    DataLoaders for any HuggingFace dataset that can be used for sequence classification tasks.
    It automatically handles tokenization using HuggingFace's AutoTokenizer, creates train/validation
    splits, and supports both single-label and multi-label classification scenarios.

    Key Features:
        - Automatic dataset downloading and caching
        - Flexible train/validation/test split configuration
        - Built-in tokenization with customizable parameters
        - Multi-label classification support
        - Performance optimizations (multiprocessing, pin memory, persistent workers)
        - Comprehensive parameter validation using Pydantic

    Common Use Cases:
        - Text classification (sentiment analysis, spam detection)
        - Multi-label classification (toxic comment detection)
        - Fine-tuning transformer models on custom datasets
        - Research experiments with different tokenization strategies

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
        """Initialize the AutoTokenizerDataModule with comprehensive parameter validation.

        All parameters are validated using Pydantic models to ensure type safety and
        provide clear error messages for invalid configurations.

        For detailed parameter specifications, validation rules, and examples,
        see the class docstring above and DataModuleConfig field definitions.
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

        This method is called only once per node and is responsible for downloading
        the dataset to the cache directory. It's safe to call multiple times.

        Notes:
            - Disables tokenizer parallelism to avoid potential deadlocks
            - Only downloads data, does not assign to instance variables
            - Called automatically by PyTorch Lightning trainer

        See Also:
            https://lightning.ai/docs/pytorch/stable/data/datamodule.html#prepare-data
        """
        # Disable parallelism to avoid deadlocks during multiprocessing
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        load_dataset(self.dataset_name, cache_dir=self.cache_dir)

    def setup(self, stage: str | None) -> None:
        """Set up datasets for training, validation, and testing.

        This method is called on every GPU/process and is responsible for:
        - Loading and splitting datasets
        - Applying preprocessing (tokenization)
        - Setting up data for the specified stage

        Args:
            stage: Either 'fit' (for train/val), 'test', or None (for all stages).
                - 'fit': Sets up training and validation datasets
                - 'test': Sets up test dataset
                - None: Sets up all datasets

        Raises:
            ValueError: If the loaded dataset is not a HuggingFace Dataset object

        Notes:
            - Training data is automatically split into train/validation using val_size
            - All datasets are tokenized and formatted for PyTorch tensors
            - Preprocessing is done with multiprocessing for efficiency

        See Also:
            https://lightning.ai/docs/pytorch/stable/data/datamodule.html#setup
        """
        if stage == "fit" or stage is None:
            dataset = load_dataset(
                self.dataset_name,
                cache_dir=self.cache_dir,
                split=self.train_split,
            )

            if not isinstance(dataset, Dataset):
                raise ValueError(f"Expected HuggingFace Dataset, got {type(dataset)}")

            # Split into train/validation using the configured ratio
            dataset_dict = dataset.train_test_split(test_size=self.val_size)

            # Preprocess both splits with multiprocessing
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

        This method is applied to each batch during dataset preprocessing and handles:
        - Tokenizing input text using the configured tokenizer
        - Combining multiple label columns for multi-label classification
        - Ensuring consistent output format for DataLoaders

        Args:
            batch: A batch dictionary from the HuggingFace dataset containing
                text data and labels according to the configured column names.

        Returns:
            Processed batch dictionary with tokenized inputs and combined labels.
            Contains keys like "input_ids", "attention_mask", and "labels".

        Notes:
            - Uses the configured max_token_len for truncation/padding
            - Combines multiple label columns using the combine_labels utility
            - Returns lists (not tensors) for compatibility with HuggingFace datasets
        """
        # Tokenize text with the configured parameters
        inputs = self.tokenizer(
            batch[self.text_column],
            max_length=self.max_token_len,
            padding="max_length",
            truncation=True,
            return_tensors=None,  # Return as lists for dataset compatibility
        )

        # Handle single-label vs multi-label scenarios
        if len(self.label_columns) > 1:
            # Multi-label: combine all label columns (already converted to float)
            inputs["labels"] = combine_labels(batch, self.label_columns)
        else:
            # Single-label: use the single label column directly and convert to float
            inputs["labels"] = np.array(
                batch[self.label_columns[0]], dtype=float
            ).tolist()

        return inputs

    def _create_dataloader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        """Create a DataLoader with consistent configuration.

        This internal method ensures all DataLoaders use the same configuration
        for performance optimization and consistency.

        Args:
            dataset: The HuggingFace dataset to wrap in a DataLoader
            shuffle: Whether to shuffle the data (True for training, False for eval)

        Returns:
            Configured PyTorch DataLoader with performance optimizations enabled
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
            DataLoader for training data with shuffling enabled

        Raises:
            RuntimeError: If called before setup('fit')

        Notes:
            https://lightning.ai/docs/pytorch/stable/data/datamodule.html#train-dataloader
        """
        if not hasattr(self, "train_data") or self.train_data is None:
            raise RuntimeError("Training data not setup. Call setup('fit') first.")
        return self._create_dataloader(self.train_data, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """Return the validation DataLoader.

        Returns:
            DataLoader for validation data without shuffling

        Raises:
            RuntimeError: If called before setup('fit')

        Notes:
            https://lightning.ai/docs/pytorch/stable/data/datamodule.html#val-dataloader
        """
        if not hasattr(self, "val_data") or self.val_data is None:
            raise RuntimeError("Validation data not setup. Call setup('fit') first.")
        return self._create_dataloader(self.val_data, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        """Return the test DataLoader.

        Returns:
            DataLoader for test data without shuffling

        Raises:
            RuntimeError: If called before setup('test')

        Notes:
            https://lightning.ai/docs/pytorch/stable/data/datamodule.html#test-dataloader
        """
        if not hasattr(self, "test_data") or self.test_data is None:
            raise RuntimeError("Test data not setup. Call setup('test') first.")
        return self._create_dataloader(self.test_data, shuffle=False)


def create_jigsaw_datamodule(**kwargs) -> AutoTokenizerDataModule:
    """Create a Jigsaw toxic comment classification DataModule with sensible defaults.

    This convenience function provides pre-configured defaults for the Jigsaw Toxic Comment
    Classification dataset, making it easy to get started with toxic comment detection tasks.
    All parameters can be overridden by passing them as keyword arguments.

    The Jigsaw dataset contains Wikipedia comments labeled for various types of toxicity
    including toxic, severe_toxic, obscene, threat, insult, and identity_hate. This makes
    it ideal for multi-label classification experiments.

    Args:
        **kwargs: Any parameters to override the defaults. Common overrides include:
            - model_name: Change the transformer model (default: "prajjwal1/bert-tiny")
            - max_token_len: Adjust max sequence length (default: uses DataModule default)
            - val_size: Change validation split size (default: uses DataModule default)
            - batch_size: Adjust batch size (default: uses DataModule default)
            - cache_dir: Change cache directory (default: uses DataModule default)

    Returns:
        AutoTokenizerDataModule: Fully configured data module for Jigsaw dataset
        ready for training toxic comment classifiers.

    Examples:
        Use with all defaults:

        >>> dm = create_jigsaw_datamodule()
        >>> dm.prepare_data()
        >>> dm.setup("fit")

        Override specific parameters:

        >>> dm = create_jigsaw_datamodule(
        ...     model_name="distilbert-base-uncased",
        ...     max_token_len=256,
        ...     batch_size=16,
        ...     val_size=0.15
        ... )

        Use with custom cache directory:

        >>> dm = create_jigsaw_datamodule(
        ...     cache_dir="/fast_storage/huggingface_cache"
        ... )

    Notes:
        - Uses the bert-tiny model by default for fast experimentation
        - Configured for all 6 toxicity labels (multi-label classification)
        - Perfect for research and experimentation with toxic comment detection
        - All AutoTokenizerDataModule features are available
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

    # Update defaults with any user-provided overrides
    default_config.update(kwargs)
    return AutoTokenizerDataModule(**default_config)


if __name__ == "__main__":
    # Example usage and testing
    jigsaw_dm = create_jigsaw_datamodule()
    jigsaw_dm.prepare_data()
    jigsaw_dm.setup("fit")

    # Test train dataloader
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
