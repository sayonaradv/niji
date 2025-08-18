import os
from typing import cast

import lightning.pytorch as pl
import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset


class AutoTokenizerDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_name: str,
        *,
        train_split: str,
        test_split: str,
        text_column: str,
        label_columns: list[str],
        val_size: float = 0.2,
        batch_size: int = 32,
        cache_dir: str | None = "data",
    ) -> None:
        super().__init__()

        self.dataset_name = dataset_name
        self.train_split = train_split
        self.test_split = test_split
        self.text_column = text_column
        self.label_columns = label_columns
        self.val_size = val_size
        self.batch_size = batch_size
        self.cache_dir = cache_dir

        cpu_count = os.cpu_count()
        self.num_workers = cpu_count if cpu_count is not None else 0

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
            val_ds = dataset_dict.pop("test")
            dataset_dict["val"] = val_ds

            for split, ds in dataset_dict.items():
                ds = ds.map(
                    self.preprocess_data,
                    batched=True,
                    num_proc=self.num_workers,
                    remove_columns=ds.column_names,
                    desc=f"Preprocessing {split} split",
                )
                ds.set_format(type="torch", columns=["text", "labels"])
                if split == "train":
                    self.train_data = ds
                else:
                    self.val_data = ds

        if stage == "test" or stage is None:
            ds = load_dataset(
                self.dataset_name,
                cache_dir=self.cache_dir,
                split=self.test_split,
            )

            if not isinstance(ds, Dataset):
                raise ValueError(f"Expected HuggingFace Dataset, got {type(ds)}")

            self.test_data = ds.map(
                self.preprocess_data,
                batched=True,
                num_proc=self.num_workers,
                remove_columns=ds.column_names,
                desc="Tokenizing test split",
            )

            self.test_data.set_format(type="torch", columns=["text", "labels"])

    def preprocess_data(self, batch: dict) -> dict:
        # Combine the binary label columns into one tensor per example
        labels = torch.tensor(
            list(zip(*[batch[col] for col in self.label_columns], strict=False)),
            dtype=torch.float,
        )
        return {
            "text": batch[self.text_column],
            "labels": labels,
        }

    def _create_dataloader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            cast(TorchDataset, dataset),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True,
            drop_last=True,
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


def get_jigsaw_datamodule(
    val_size: float = 0.2, batch_size: int = 8, cache_dir: str = "data"
) -> AutoTokenizerDataModule:
    return AutoTokenizerDataModule(
        "mat55555/jigsaw_toxic_comment",
        train_split="train[:1000]",
        test_split="test[:1000]",
        text_column="text",
        label_columns=[
            "toxic",
            "severe_toxic",
            "obscene",
            "threat",
            "insult",
            "identity_hate",
        ],
        val_size=val_size,
        batch_size=batch_size,
        cache_dir=cache_dir,
    )


if __name__ == "__main__":
    jigsaw_dm = get_jigsaw_datamodule()
    jigsaw_dm.prepare_data()
    jigsaw_dm.setup(stage="fit")

    train_ds = jigsaw_dm.train_data
    val_ds = jigsaw_dm.val_data

    train_dl = jigsaw_dm.train_dataloader()
    val_dl = jigsaw_dm.val_dataloader()

    batch = next(iter(train_dl))
    print(batch)
