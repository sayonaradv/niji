import os

import pandas as pd
from torch.utils.data.dataset import Dataset

JIGSAW_DATA_DIR: str = os.path.join(
    "data", "jigsaw-toxic-comment-classification-challenge"
)


class JigsawDataset(Dataset):
    def __init__(self, split: str = "train", data_dir: str | None = None) -> None:
        self.data = self.load_data(split=split, data_dir=data_dir)

    def load_data(self, split: str, data_dir: str | None) -> pd.DataFrame:
        if data_dir is None:
            data_dir = os.path.join(
                "data", "jigsaw-toxic-comment-classification-challenge"
            )

        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"{data_dir} does not exist. ")

        train_path: str = os.path.join(data_dir, "train.csv")
        test_text_path: str = os.path.join(data_dir, "test.csv")
        test_labels_path: str = os.path.join(data_dir, "test_labels.csv")

        if split == "test":
            df1 = pd.read_csv(test_text_path)
            df2 = pd.read_csv(test_labels_path)
            df3 = df1.merge(df2)
            df4 = df3[df3["toxic"] != -1].reset_index(drop=True)

        elif split == "train":
            df4 = pd.read_csv(train_path)

        df5 = df4.astype(
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

        df6 = df5.rename(columns={"comment_text": "text"}).drop(columns=["id"])

        return df6


if __name__ == "__main__":
    train_ds = JigsawDataset(split="train")
    test_ds = JigsawDataset(split="test")

    print(train_ds.data)
    print(test_ds.data)
