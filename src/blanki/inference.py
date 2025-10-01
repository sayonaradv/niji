from collections.abc import Mapping
from pathlib import Path
from time import perf_counter

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.loggers import TensorBoardLogger

from blanki.dataloader import JigsawDataModule
from blanki.exceptions import ModelNotFoundError
from blanki.module import Classifier
from blanki.training import DATA_DIR
from blanki.utils import log_perf

# See https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
torch.set_float32_matmul_precision("medium")

DOWNLOAD_BASE_URL = "https://github.com/sudojayder/blanki/releases/download/"
AVAILABLE_MODELS = {
    "bert-tiny": f"{DOWNLOAD_BASE_URL}v0.0.1alpha4/finetuned-bert-tiny.ckpt",
    "distilbert": f"{DOWNLOAD_BASE_URL}v0.0.1alpha4/finetuned-distilbert.ckpt",
}

TEST_CKPT_PATH: str = (
    "runs/training_runs/bert-tiny/checkpoints/epoch=01-val_loss=0.0726.ckpt"
)


def load_checkpoint(
    model_name: str | None = None,
    ckpt_path: str | None = None,
    device: str | None = None,
) -> Classifier:
    """Load a Classifier from checkpoint, either locally or from remote location.

    Args:
        model_name (str, optional): Name of available model to download remotely.
        ckpt_path (str, optional): Path to local checkpoint file.
        device (str, optional): Device to load model on (e.g., 'cuda', 'cpu').

    Returns:
        Classifier: The loaded model instance.

    Raises:
        ValueError: If neither model_name nor ckpt_path is provided, or if
            model_name is not in AVAILABLE_MODELS.
        ModelNotFoundError: If the local checkpoint file doesn't exist.
    """
    if not (model_name or ckpt_path):
        raise ValueError("Must provide either 'model_name' or 'ckpt_path'.")

    if ckpt_path:
        # Use local checkpoint - validate it exists
        checkpoint_path = Path(ckpt_path)
        if not checkpoint_path.is_file():
            raise ModelNotFoundError(f"Checkpoint file does not exist: {ckpt_path}")
        final_path = str(checkpoint_path)
    else:
        # Use remote model - validate model_name
        if model_name is None or model_name not in AVAILABLE_MODELS:
            available_models: str = ", ".join(AVAILABLE_MODELS.keys())
            raise ValueError(
                f"Unknown model '{model_name}'. Available models: {available_models}"
            )
        final_path: str = AVAILABLE_MODELS[model_name]

    return Classifier.load_from_checkpoint(final_path, map_location=device)


def test(
    model_name: str | None = None,
    ckpt_path: str | None = TEST_CKPT_PATH,
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
    model: Classifier = load_checkpoint(model_name, ckpt_path)
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
