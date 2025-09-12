from time import perf_counter

import lightning.pytorch as pl
import torch
from jsonargparse import auto_cli

from ruffle.config import DataConfig
from ruffle.dataloader import JigsawDataModule
from ruffle.module import RuffleModel
from ruffle.utils import log_perf

# See https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
torch.set_float32_matmul_precision("medium")


def test(
    ckpt_path: str,
    data_dir: str = DataConfig.data_dir,
    batch_size: int = DataConfig.batch_size,
    perf: bool = True,
) -> None:
    model = RuffleModel.load_from_checkpoint(ckpt_path)
    model.eval()

    datamodule = JigsawDataModule(
        data_dir=data_dir,
        batch_size=batch_size,
        labels=model.hparams["label_names"],
    )

    trainer = pl.Trainer()

    start = perf_counter()
    trainer.test(model, datamodule=datamodule)
    stop = perf_counter()

    if perf:
        log_perf(start, stop, trainer)


def main() -> None:
    auto_cli(test)


if __name__ == "__main__":
    main()
