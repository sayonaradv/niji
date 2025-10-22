import pandas as pd
import pytest
import torch
from pydantic import ValidationError
from torch.utils.data import DataLoader, Dataset

from niji.dataloader import JIGSAW_LABELS, JigsawDataModule, JigsawDataset, Split
from niji.exceptions import DataNotFoundError


@pytest.fixture
def mock_data_dir(tmp_path) -> str:
    """Create mock  CSVs (10 rows) mimicking the Jigsaw dataset format.

    train: 4 toxic samples.
    test: 2 toxic samples, 2 invalid samples.
    """
    train = pd.DataFrame(
        {
            "id": range(1, 11),
            "comment_text": [f"comment {i}" for i in range(1, 11)],
            "toxic": [0, 1, 0, 1, 0, 1, 0, 0, 1, 0],
            "severe_toxic": [0, 1, 0, 1, 0, 1, 0, 0, 1, 0],
            "obscene": [0, 1, 0, 1, 0, 1, 0, 0, 1, 0],
            "threat": [0, 1, 0, 1, 0, 1, 0, 0, 1, 0],
            "insult": [0, 1, 0, 1, 0, 1, 0, 0, 1, 0],
            "identity_hate": [0, 1, 0, 1, 0, 1, 0, 0, 1, 0],
        }
    )
    train.to_csv(tmp_path / "train.csv", index=False)

    test = pd.DataFrame(
        {
            "id": range(11, 21),
            "comment_text": [f"test comment {i}" for i in range(11, 21)],
        }
    )
    test.to_csv(tmp_path / "test.csv", index=False)

    test_labels = pd.DataFrame(
        {
            "id": range(11, 21),
            "toxic": [0, -1, 0, -1, 0, 1, 0, 0, 1, 0],
            "severe_toxic": [0, -1, 0, -1, 0, 1, 0, 0, 1, 0],
            "obscene": [0, -1, 0, -1, 0, 1, 0, 0, 1, 0],
            "threat": [0, -1, 0, -1, 0, 1, 0, 0, 1, 0],
            "insult": [0, -1, 0, -1, 0, 1, 0, 0, 1, 0],
            "identity_hate": [0, -1, 0, -1, 0, 1, 0, 0, 1, 0],
        }
    )
    test_labels.to_csv(tmp_path / "test_labels.csv", index=False)

    return str(tmp_path)


class TestSplit:
    def test_train_files(self) -> None:
        assert Split.TRAIN.inputs_file == "train.csv"
        assert Split.TRAIN.labels_file is None

    def test_test_files(self) -> None:
        assert Split.TEST.inputs_file == "test.csv"
        assert Split.TEST.labels_file == "test_labels.csv"


def test_train_load_data(mock_data_dir) -> None:
    df = Split.TRAIN.load_data(mock_data_dir)

    assert len(df) == 10
    assert len(df.columns.tolist()) == 8


def test_test_load_data(mock_data_dir) -> None:
    df = Split.TEST.load_data(mock_data_dir)

    # Should have 8 rows (2 filtered out due to toxic=-1)
    assert len(df) == 8
    assert "comment_text" in df.columns
    assert len(df.columns.tolist()) == 8


class TestJigsawDataset:
    def test_init_with_defaults(self, mock_data_dir) -> None:
        dataset = JigsawDataset(Split.TRAIN, data_dir=mock_data_dir)

        assert len(dataset.data) == 10  # from fixture
        assert dataset.labels == JIGSAW_LABELS  # defaults to all 6 labels

    def test_init_with_custom_labels(self, mock_data_dir) -> None:
        custom_labels = ["toxic", "insult"]
        dataset = JigsawDataset(
            split=Split.TRAIN, data_dir=mock_data_dir, labels=custom_labels
        )

        assert dataset.labels == custom_labels
        assert len(dataset.labels) == 2

    def test_len(self, mock_data_dir) -> None:
        train_ds = JigsawDataset(Split.TRAIN, data_dir=mock_data_dir)
        test_ds = JigsawDataset(Split.TEST, data_dir=mock_data_dir)

        assert len(train_ds) == 10
        assert len(test_ds) == 8  # after filtering

    def test_getitem(self, mock_data_dir) -> None:
        dataset = JigsawDataset(Split.TRAIN, data_dir=mock_data_dir)
        item = dataset[0]

        # Check item structure
        assert isinstance(item, dict)
        assert set(item.keys()) == {"text", "labels"}

        # Check text format
        assert isinstance(item["text"], str)
        assert item["text"] == "comment 1"  # from fixture data

        # Check labels format
        assert isinstance(item["labels"], torch.FloatTensor)
        assert item["labels"].shape == (6,)  # 6 JIGSAW_LABELS
        assert item["labels"].dtype == torch.float32

    def test_getitem_with_custom_labels(self, mock_data_dir) -> None:
        custom_labels = ["toxic", "insult"]
        dataset = JigsawDataset(
            Split.TRAIN, data_dir=mock_data_dir, labels=custom_labels
        )
        item = dataset[0]

        assert isinstance(item["labels"], torch.FloatTensor)
        assert item["labels"].shape == (len(custom_labels),)
        assert item["labels"].dtype == torch.float32

    def test_check_labels(self, mock_data_dir) -> None:
        missing_labels = ["missing_1", "missing_2"]
        with pytest.raises(ValueError, match="not found in dataset"):
            JigsawDataset(Split.TRAIN, data_dir=mock_data_dir, labels=missing_labels)

    def test_check_data_dir(self) -> None:
        data_dir = "nonexistant/data_dir"
        with pytest.raises(DataNotFoundError, match="Data directory not found"):
            JigsawDataset(Split.TRAIN, data_dir=data_dir)


class TestJigsawDataModule:
    def test_invalid_batch_size(self, mock_data_dir) -> None:
        with pytest.raises(ValidationError, match="batch_size"):
            JigsawDataModule(mock_data_dir, batch_size=-10)

    def test_invalid_val_size(self, mock_data_dir) -> None:
        for invalid_val_size in [-0.2, 1.2]:
            with pytest.raises(ValidationError, match="val_size"):
                JigsawDataModule(mock_data_dir, val_size=invalid_val_size)

    def test_default_labels(self, mock_data_dir) -> None:
        dm = JigsawDataModule(mock_data_dir)
        assert dm.labels == JIGSAW_LABELS

    def test_fit_stage(self, mock_data_dir) -> None:
        batch_size = 4
        val_size = 0.2

        dm = JigsawDataModule(mock_data_dir, batch_size=batch_size, val_size=val_size)
        dm.setup("fit")

        # Only train and val datasets/dataloaders should be created
        assert isinstance(dm.train_ds, Dataset)
        assert isinstance(dm.val_ds, Dataset)
        assert isinstance(dm.train_dataloader(), DataLoader)
        assert isinstance(dm.val_dataloader(), DataLoader)
        assert dm.test_ds is None
        assert dm.test_dataloader() is None

        # Check dataloader sizes
        assert len(dm.train_dataloader()) == len(dm.train_ds) // batch_size  # type: ignore[arg-type]
        assert len(dm.val_dataloader()) == len(dm.val_ds) // batch_size  # type: ignore[arg-type]

        # Check validation data size
        assert len(dm.val_dataloader()) == int(len(dm.train_dataloader()) * val_size)  # type: ignore[arg-type]

    def test_test_stage(self, mock_data_dir) -> None:
        batch_size = 4
        dm = JigsawDataModule(mock_data_dir, batch_size=batch_size)
        dm.setup("test")

        # Only test dataset/dataloader should be created
        assert dm.train_ds is None
        assert dm.val_ds is None
        assert dm.train_dataloader() is None
        assert dm.val_dataloader() is None

        assert isinstance(dm.test_ds, Dataset)
        assert isinstance(dm.test_dataloader(), DataLoader)

        assert len(dm.test_dataloader()) == len(dm.test_ds) // batch_size  # type: ignore[arg-type]
