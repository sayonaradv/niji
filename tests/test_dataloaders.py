import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from rynn.dataloaders import JigsawDataModule


@pytest.fixture
def tiny_jigsaw_dir(tmp_path) -> str:
    """Create synthetic CSVs (20 rows) mimicking the Jigsaw dataset format."""

    # train.csv (20 rows)
    train = pd.DataFrame(
        {
            "id": range(1, 21),
            "comment_text": [f"comment {i}" for i in range(1, 21)],
            "toxic": [i % 2 for i in range(20)],
            "severe_toxic": [0 for _ in range(20)],
            "obscene": [i % 3 == 0 for i in range(20)],
            "threat": [0 for _ in range(20)],
            "insult": [i % 4 == 0 for i in range(20)],
            "identity_hate": [0 for _ in range(20)],
        }
    )
    train.to_csv(tmp_path / "train.csv", index=False)

    # test.csv (20 rows)
    test = pd.DataFrame(
        {
            "id": range(21, 41),
            "comment_text": [f"test comment {i}" for i in range(21, 41)],
        }
    )
    test.to_csv(tmp_path / "test.csv", index=False)

    # test_labels.csv (20 rows)
    test_labels = pd.DataFrame(
        {
            "id": range(21, 41),
            "toxic": [i % 2 for i in range(20)],
            "severe_toxic": [0 for _ in range(20)],
            "obscene": [i % 3 == 0 for i in range(20)],
            "threat": [0 for _ in range(20)],
            "insult": [i % 4 == 0 for i in range(20)],
            "identity_hate": [0 for _ in range(20)],
        }
    )
    test_labels.to_csv(tmp_path / "test_labels.csv", index=False)

    return str(tmp_path)


@pytest.fixture
def dm(tiny_jigsaw_dir) -> JigsawDataModule:
    return JigsawDataModule(data_dir=tiny_jigsaw_dir, batch_size=1)


def test_train_dataloader_raises_without_setup(dm) -> None:
    with pytest.raises(RuntimeError, match="setup\\('fit'\\)"):
        _ = dm.train_dataloader()


def test_val_dataloader_raises_without_setup(dm) -> None:
    with pytest.raises(RuntimeError, match="setup\\('fit'\\)"):
        _ = dm.val_dataloader()


def test_test_dataloader_raises_without_setup(dm) -> None:
    with pytest.raises(RuntimeError, match="setup\\('test'\\)"):
        _ = dm.test_dataloader()


def test_fit_dataloaders_after_setup(dm) -> None:
    dm.setup(stage="fit")

    train_dl = dm.train_dataloader()
    val_dl = dm.val_dataloader()

    # Assert dataloader types
    assert isinstance(train_dl, DataLoader)
    assert isinstance(val_dl, DataLoader)

    # Check batch counts
    assert len(train_dl) == len(dm.train_ds) // dm.batch_size  # 16 // 4 = 4 batches
    assert len(val_dl) == len(dm.val_ds) // dm.batch_size  # 4 // 4 = 1 batch

    # Check batch schema
    batch = next(iter(train_dl))
    assert "text" in batch and "labels" in batch
    assert isinstance(batch["text"][0], str)
    assert isinstance(batch["labels"], torch.Tensor)
    assert batch["labels"].ndim == 2  # multi-label


def test_test_dataloader_after_setup(dm) -> None:
    dm.setup(stage="test")
    test_dl = dm.test_dataloader()

    # 20 rows, batch_size=4, drop_last=True → 5 batches
    assert len(test_dl) == len(dm.test_ds) // dm.batch_size

    batch = next(iter(test_dl))
    assert "text" in batch and "labels" in batch
    assert isinstance(batch["text"][0], str)
    assert isinstance(batch["labels"], torch.Tensor)
    assert batch["labels"].ndim == 2
