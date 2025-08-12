import os
from typing import Any, cast

import lightning.pytorch as pl
import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from stormy.utils import combine_labels, get_num_workers


class AutoTokenizerDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_name: str,
        train_split: str,
        test_split: str,
        text_column: str,
        label_columns: list[str],
        val_size: float = 0.2,
        batch_size: int = 32,
        cache_dir: str = "./data",
    ) -> None:
        super().__init__()

        # # Validate parameters using Pydantic
        # try:
        #     config = DataModuleConfig(
        #         dataset_name=dataset_name,
        #         model_name=model_name,
        #         train_split=train_split,
        #         test_split=test_split,
        #         text_column=text_column,
        #         label_columns=label_columns,
        #         max_token_len=max_token_len,
        #         val_size=val_size,
        #         batch_size=batch_size,
        #         cache_dir=cache_dir,
        #     )
        # except ValidationError as e:
        #     raise ValueError(
        #         f"Invalid configuration for AutoTokenizerDataModule: {e}"
        #     ) from e

        # Store validated configuration values
        self.dataset_name = dataset_name
        self.train_split = train_split
        self.test_split = test_split
        self.text_column = text_column
        self.label_columns = label_columns
        self.val_size = val_size
        self.batch_size = batch_size
        self.cache_dir = cache_dir

        # Performance optimizations
        self.num_workers = get_num_workers()
        self.persistent_workers = True
        self.pin_memory = torch.cuda.is_available()

    def prepare_data(self) -> None:
        # Disable parallelism to avoid deadlocks during multiprocessing
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        load_dataset(self.dataset_name, cache_dir=self.cache_dir)

    def setup(self, stage: str | None) -> None:
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
        processed_batch = {}
        processed_batch["text"] = batch[self.text_column]
        if len(self.label_columns) > 1:
            processed_batch["labels"] = combine_labels(batch, self.label_columns)
        else:
            processed_batch["labels"] = torch.tensor(
                batch[self.label_columns[0]],
                dtype=torch.float,
            )

        return processed_batch

    def _create_dataloader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            cast(TorchDataset, dataset),
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            shuffle=shuffle,
        )

    def train_dataloader(self) -> DataLoader:
        if not hasattr(self, "train_data") or self.train_data is None:
            raise RuntimeError("Training data not setup. Call setup('fit') first.")
        return self._create_dataloader(self.train_data, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        if not hasattr(self, "val_data") or self.val_data is None:
            raise RuntimeError("Validation data not setup. Call setup('fit') first.")
        return self._create_dataloader(self.val_data, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        if not hasattr(self, "test_data") or self.test_data is None:
            raise RuntimeError("Test data not setup. Call setup('test') first.")
        return self._create_dataloader(self.test_data, shuffle=False)
