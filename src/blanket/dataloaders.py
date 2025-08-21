import os
from typing import Literal

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data.dataset import Dataset

JIGSAW_DATA_DIR: str = os.path.join(
    "data", "jigsaw-toxic-comment-classification-challenge"
)


class JigsawDataset(Dataset):
    def __init__(
        self, split: Literal["train", "test"] = "train", data_dir: str = JIGSAW_DATA_DIR
    ) -> None:
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

        return (
            df.astype(
                {
                    "comment_text": "str",
                    "toxic": "int",
                    "severe_toxic": "int",
                    "obscene": "int",
                    "threat": "int",
                    "insult": "int",
                    "identity_hate": "int",
                }
            )
            .rename(columns={"comment_text": "text"})
            .drop(columns=["id"])
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, str | Tensor]:
        row: dict[str, str | int] = self.data.iloc[idx].to_dict()
        text: str = str(row.pop("text"))
        labels: Tensor = torch.IntTensor(list(row.values()))
        return {"text": text, "labels": labels}


if __name__ == "__main__":
    train_ds = JigsawDataset(split="train")
    test_ds = JigsawDataset(split="test")

    print(f"Train dataset size: {len(train_ds)}")
    print(f"Test dataset size: {len(test_ds)}")
    print("\nSample from train dataset:")
    print(train_ds[0])
