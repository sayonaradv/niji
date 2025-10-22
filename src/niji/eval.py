from collections.abc import Mapping
from time import perf_counter

import lightning.pytorch as pl
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.loggers import TensorBoardLogger

from niji.dataloader import JigsawDataModule
from niji.inference import load_checkpoint
from niji.training import DATA_DIR
from niji.utils import log_perf


def test(
    model_name: str | None = None,
    ckpt_path: str | None = None,
    data_dir: str = DATA_DIR,
    batch_size: int = 64,
    num_workers: int | None = None,
    perf: bool = True,
    run_name: str | None = None,
) -> Mapping[str, float]:
    """Evaluate a trained model on the Jigsaw test dataset.

    Args:
        model_name (str, optional): Name of available model to download remotely.
        ckpt_path (str, optional): Path to local checkpoint file.
        data_dir (str): Directory containing the test data. Defaults to DataConfig.data_dir.
        batch_size (int): Batch size for evaluation. Defaults to DataConfig.batch_size.
        num_workers (int, optional): Number of worker processes for data loading.
            If None, defaults to the number of CPU cores. If 0, uses single-threaded
            data loading. Must be non-negative. Defaults to None.
        run_name (str, optional): Name of the experiment for logging.
        perf (bool): Whether to log performance metrics. Defaults to True.

    Returns:
        Mapping[str, float]: Dictionary with metrics logged during the test phase.

    Raises:
        ValueError: If neither model_name nor ckpt_path is provided.
        ModelNotFoundError: If checkpoint file doesn't exist.
    """
    model = load_checkpoint(model_name, ckpt_path)
    model.eval()

    datamodule = JigsawDataModule(
        data_dir,
        batch_size=batch_size,
        labels=model.hparams["label_names"],
        num_workers=num_workers,
    )

    logger = TensorBoardLogger(save_dir="runs", name="test_runs", version=run_name)
    trainer = pl.Trainer(logger=logger, callbacks=[RichProgressBar()])

    if perf:
        start_time: float = perf_counter()
        metrics: Mapping[str, float] = trainer.test(model, datamodule=datamodule)[0]
        end_time: float = perf_counter()
        log_perf(start_time, end_time, trainer)
    else:
        metrics: Mapping[str, float] = trainer.test(model, datamodule=datamodule)[0]

    return metrics
