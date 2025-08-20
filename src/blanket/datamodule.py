import os

import lightning.pytorch as pl
import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader


class HFDataModule(pl.LightningDataModule):
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

        self.train_data: Dataset | None = None
        self.val_data: Dataset | None = None
        self.test_data: Dataset | None = None

    def prepare_data(self) -> None:
        # Disable parallelism to avoid deadlocks during multiprocessing
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        _ = load_dataset(self.dataset_name, cache_dir=self.cache_dir)

    def setup(self, stage: str | None) -> None:
        if stage == "fit" or stage is None:
            ds = load_dataset(
                self.dataset_name, cache_dir=self.cache_dir, split=self.train_split
            )

            ds = ds.map(
                self.combine_labels,
                remove_columns=ds.column_names,
                batched=True,
            )
            ds.set_format("torch")
            ds = ds.train_test_split(test_size=self.val_size)

            self.train_data, self.val_data = ds["train"], ds["test"]

            del ds

        if stage == "test" or stage is None:
            ds = load_dataset(
                self.dataset_name, cache_dir=self.cache_dir, split=self.test_split
            )

            ds = ds.map(
                self.combine_labels,
                remove_columns=ds.column_names,
                batched=True,
            )

            ds.set_format("torch")
            self.test_data = ds

            del ds

    def combine_labels(self, batch: dict) -> dict:
        # Combine the binary label columns into one tensor per example
        labels = torch.tensor(
            list(zip(*[batch[col] for col in self.label_columns], strict=False)),
            dtype=torch.int32,
        )
        return {
            "text": batch[self.text_column],
            "labels": labels.tolist(),
        }

    def _create_dataloader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            shuffle=shuffle,
            batch_size=self.batch_size,
            drop_last=True,
        )

    def train_dataloader(self) -> DataLoader:
        if self.train_data is None:
            raise RuntimeError("Training data not setup. Call setup('fit') first.")
        return self._create_dataloader(self.train_data, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        if self.val_data is None:
            raise RuntimeError("Validation data not setup. Call setup('fit') first.")
        return self._create_dataloader(self.val_data, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        if self.test_data is None:
            raise RuntimeError("Test data not setup. Call setup('test') first.")
        return self._create_dataloader(self.test_data, shuffle=False)


def get_jigsaw_datamodule(
    val_size: float = 0.2, batch_size: int = 6, cache_dir: str = "data"
) -> HFDataModule:
    return HFDataModule(
        "mat55555/jigsaw_toxic_comment",
        train_split="train",
        test_split="test",
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

    train_dl = jigsaw_dm.train_dataloader()
    val_dl = jigsaw_dm.val_dataloader()
    batch = next(iter(train_dl))
    print(batch)
