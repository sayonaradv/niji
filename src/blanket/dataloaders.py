import os
from typing import Literal

import lightning.pytorch as pl
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split

JIGSAW_DATA_DIR: str = os.path.join(
    "data", "jigsaw-toxic-comment-classification-challenge"
)


class JigsawDataset(Dataset):
    def __init__(self, split: Literal["train", "test"], data_dir: str) -> None:
        """
        Initialize Jigsaw dataset.

        Args:
            split: Either "train" or "test"
            data_dir: Directory containing the Jigsaw dataset files
        """
        self.data = self.load_data(split=split, data_dir=data_dir)

    def load_data(self, split: Literal["train", "test"], data_dir: str) -> pd.DataFrame:
        """Load data from the specified directory and split."""
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory {data_dir} does not exist.")

        train_path: str = os.path.join(data_dir, "train.csv")
        test_text_path: str = os.path.join(data_dir, "test.csv")
        test_labels_path: str = os.path.join(data_dir, "test_labels.csv")

        if split == "test":
            df: pd.DataFrame = (
                pd.read_csv(test_text_path)
                .merge(pd.read_csv(test_labels_path))
                .query("toxic != -1")
                .reset_index(drop=True)
            )

        elif split == "train":
            df = pd.read_csv(train_path)
        else:
            raise ValueError(f"Invalid split '{split}'. Must be 'train' or 'test'.")

        return df.rename(columns={"comment_text": "text"}).drop(columns=["id"])

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, str | Tensor]:
        row: dict[str, str | int] = self.data.iloc[idx].to_dict()
        text: str = str(row.pop("text"))
        labels: Tensor = torch.IntTensor(list(row.values()))
        return {"text": text, "labels": labels}


class JigsawDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = JIGSAW_DATA_DIR,
        batch_size: int = 64,
        val_size: float = 0.2,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_size = val_size

        self.train_ds: Dataset | None = None
        self.val_ds: Dataset | None = None
        self.test_ds: Dataset | None = None

    def setup(self, stage: str | None) -> None:
        if stage == "fit" or stage is None:
            lengths: list[float] = [1 - self.val_size, self.val_size]
            full_train_ds = JigsawDataset(split="train", data_dir=self.data_dir)
            self.train_ds, self.val_ds = random_split(full_train_ds, lengths)
        elif stage == "test" or stage is None:
            self.test_ds = JigsawDataset(split="test", data_dir=self.data_dir)

    def train_dataloader(self) -> DataLoader | None:
        if self.train_ds:
            return DataLoader(
                self.train_ds, shuffle=True, batch_size=self.batch_size, drop_last=True
            )
        else:
            return None

    def val_dataloader(self) -> DataLoader | None:
        if self.val_ds:
            return DataLoader(self.val_ds, batch_size=self.batch_size, drop_last=True)
        else:
            return None

    def test_dataloader(self) -> DataLoader | None:
        if self.test_ds:
            return DataLoader(self.test_ds, batch_size=self.batch_size, drop_last=True)
        else:
            return None


if __name__ == "__main__":
    dm = JigsawDataModule()
    dm.setup(stage="fit")
    train_dl = dm.train_dataloader()
    val_dl = dm.val_dataloader()
    test_dl = dm.test_dataloader()

    print(f"Train dataloader: {len(train_dl)}")
    print(f"Val dataloader: {len(val_dl)}")
    print(f"Test dataloader: {test_dl}")
