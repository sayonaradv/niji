import lightning.pytorch as pl
import torch
from jsonargparse import auto_cli

from ruffle.dataset import JIGSAW_DATA_DIR, JigsawDataModule
from ruffle.model import RuffleModel

# See https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
torch.set_float32_matmul_precision("medium")


def test(
    ckpt_path: str,
    data_dir: str = JIGSAW_DATA_DIR,
    batch_size: int = 64,
) -> None:
    """Test a trained toxicity classification model using PyTorch Lightning.

    Loads a trained checkpoint and evaluates it on the test set from the Jigsaw
    Toxic Comment Classification dataset.

    Args:
        ckpt_path: Path to the trained model checkpoint file (.ckpt).
        data_dir: Path to dataset directory containing train and test files.
        batch_size: Test batch size.

    Examples:
        Basic testing:
            test("checkpoints/epoch=10-val_loss=0.1234.ckpt")
    """
    model = RuffleModel.load_from_checkpoint(ckpt_path)
    model.eval()

    datamodule = JigsawDataModule(
        data_dir=data_dir,
        batch_size=batch_size,
        labels=model.hparams["label_names"],
    )

    trainer = pl.Trainer(logger=True)
    trainer.test(model, datamodule=datamodule)


def main() -> None:
    auto_cli(test)


if __name__ == "__main__":
    main()
