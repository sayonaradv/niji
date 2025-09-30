from time import perf_counter
from typing import Annotated

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from lightning.pytorch.loggers import TensorBoardLogger
from pydantic import (
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    validate_call,
)

from blanki.dataloader import JIGSAW_HANDLE, JigsawDataModule
from blanki.module import Classifier
from blanki.utils import log_perf

# See https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
torch.set_float32_matmul_precision("medium")

CACHE_DIR: str = "./data"
DATA_DIR: str = f"{CACHE_DIR}/{JIGSAW_HANDLE}"
LOG_DIR: str = "./runs"


def _validate_warmup_epochs(warmup_epochs: int, max_epochs: int) -> None:
    if warmup_epochs >= max_epochs:
        raise ValueError(
            f"warmup_epochs ({warmup_epochs}) must be less than max_epochs ({max_epochs})"
        )


def _validate_warmup_lr(warmup_start_lr: float, peak_lr: float) -> None:
    if warmup_start_lr >= peak_lr:
        raise ValueError(
            f"warmup_start_lr ({warmup_start_lr}) must be less than the peak lr ({peak_lr})"
        )


@validate_call(config=ConfigDict(validate_default=True))
def train(
    model_name: str,
    data_dir: str = DATA_DIR,
    labels: list[str] | None = None,
    batch_size: PositiveInt = 64,
    val_size: Annotated[float, Field(ge=0, le=1)] = 0.2,
    max_token_len: PositiveInt = 256,
    lr: PositiveFloat = 3e-5,
    warmup_start_lr: PositiveFloat = 1e-5,
    warmup_epochs: PositiveInt = 5,
    max_epochs: PositiveInt = 20,
    patience: PositiveInt = 3,
    run_name: str | None = None,
    perf: bool = False,
    fast_dev_run: bool = False,
    cache_dir: str | None = CACHE_DIR,
    log_dir: str = LOG_DIR,
    num_workers: NonNegativeInt | None = None,
    seed: NonNegativeInt = 18,
) -> None:
    """Train a transformer model for toxicity classification on the Jigsaw dataset.


    Sets up and trains a Classifier on the Jigsaw toxicity dataset with configurable
    hyperparameters, data splitting, and training options. Uses PyTorch Lightning for
    training orchestration with automatic checkpointing, early stopping, and logging.

    Args:
        model_name (str): Hugging Face model identifier (e.g., "bert-base-uncased",
            "distilbert-base-uncased"). The model will be fine-tuned for multi-label
            classification.
        data_dir (str): Path to directory containing dataset CSV files. Should contain
            "train.csv", "test.csv", and "test_labels.csv" files from the Jigsaw dataset.
            Defaults to "./data/jigsaw-toxic-comment-classification-challenge".
        labels (list[str] | None): List of label column names to include in training.
            If None, uses all available Jigsaw labels (toxic, severe_toxic, obscene,
            threat, insult, identity_hate).
        batch_size (PositiveInt): Batch size for training and validation DataLoaders.
            Must be positive. Defaults to 64.
        val_size (float): Fraction of training data to use for validation. Must be
            between 0.0 and 1.0 (inclusive). For example, 0.2 means 20% of training
            data will be used for validation. Defaults to 0.2.
        max_token_len (PositiveInt): Maximum sequence length for tokenized inputs.
            Longer sequences will be truncated. Must be positive. Defaults to 256.
        lr (PositiveFloat): Peak learning rate for the Adam optimizer. Must be positive.
            Defaults to 3e-5.
        warmup_start_lr (PositiveFloat): Starting learning rate for the warmup phase.
            Should be smaller than `lr`. Must be positive. Defaults to 1e-5.
        warmup_epochs (PositiveInt): Number of epochs for learning rate warmup before
            reaching peak `lr`. Must be positive. Defaults to 5.
        max_epochs (PositiveInt): Maximum number of training epochs. Must be positive.
            Defaults to 20.
        patience (PositiveInt): Number of epochs with no improvement in validation loss
            before early stopping is triggered. Must be positive. Ignored when `perf`
            is True. Defaults to 3.
        run_name (str | None): Name of the experiment run for logging and checkpoint
            organization. If None, PyTorch Lightning will auto-generate a version name.
        perf (bool): Whether to enable performance benchmarking mode. When True,
            disables early stopping and logs performance metrics to "perf.json".
            Defaults to False.
        fast_dev_run (bool): Whether to run in fast development mode. When True,
            runs a single batch through training and validation for debugging.
            Defaults to False.
        cache_dir (str | None): Directory to cache pretrained models and tokenizers.
            If None, uses the default transformers cache directory. Defaults to "./data".
        log_dir (str): Directory for saving TensorBoard logs and checkpoints.
            Defaults to "./runs".
        num_workers (NonNegativeInt | None): Number of worker processes for data loading.
            If None, defaults to the number of CPU cores. If 0, uses single-threaded
            data loading. Must be non-negative. Defaults to None.
        seed (NonNegativeInt): Random seed for reproducibility across PyTorch,
            NumPy, and Python random number generators. Must be non-negative.
            Defaults to 18.

    Raises:
        DataNotFoundError: If the data directory doesn't exist or required CSV files
            are missing.
        ValueError: If specified labels are not found in the dataset columns, or if
            validation parameters are invalid.

    Example:
        >>> # Basic training with default parameters
        >>> train("bert-base-uncased")

        >>> # Custom training with specific labels and hyperparameters
        >>> train(
        ...     model_name="distilbert-base-uncased",
        ...     labels=["toxic", "severe_toxic"],
        ...     batch_size=32,
        ...     lr=2e-5,
        ...     max_epochs=10,
        ...     run_name="distilbert_experiment"
        ... )

        >>> # Fast development run for debugging
        >>> train("bert-tiny", fast_dev_run=True)
    """
    _validate_warmup_epochs(warmup_epochs, max_epochs)
    _validate_warmup_lr(warmup_start_lr, lr)

    pl.seed_everything(seed, workers=True)

    datamodule = JigsawDataModule(data_dir, batch_size, val_size, labels, num_workers)

    # If no labels provided, datamodule will automatically use JIGSAW_LABELS
    labels = datamodule.labels
    num_labels: int = len(labels)

    model = Classifier(
        model_name,
        num_labels,
        labels,
        max_token_len,
        lr,
        warmup_start_lr,
        warmup_epochs,
        cache_dir,
    )

    logger = TensorBoardLogger(save_dir=log_dir, name="training_runs", version=run_name)

    callbacks: list[Callback] = [
        ModelCheckpoint(filename="{epoch:02d}-{val_loss:.4f}"),
        LearningRateMonitor(logging_interval="epoch"),
        RichProgressBar(),
    ]

    # do not use EarlyStopping if getting perf benchmark
    if not perf:
        callbacks.append(
            EarlyStopping(monitor="val_loss", patience=patience),
        )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        logger=logger,
        callbacks=callbacks,
        fast_dev_run=fast_dev_run,
        deterministic=True,
    )

    start: float = perf_counter()
    trainer.fit(model, datamodule=datamodule)
    stop: float = perf_counter()

    if perf:
        log_perf(start, stop, trainer)
