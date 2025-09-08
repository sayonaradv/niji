import lightning.pytorch as pl
import torch
from jsonargparse import auto_cli
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)

from ruffle.config import DatasetConfig, ModelConfig, TrainerConfig
from ruffle.dataset import JigsawDataModule
from ruffle.model import Classifier

_DEFAULT_DATASET_CONFIG = DatasetConfig()
_DEFAULT_MODEL_CONFIG = ModelConfig()
_DEFAULT_TRAINER_CONFIG = TrainerConfig()

# See https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
torch.set_float32_matmul_precision("medium")


def train(
    model_name: str,
    data_dir: str = _DEFAULT_DATASET_CONFIG.data_dir,
    batch_size: int = _DEFAULT_DATASET_CONFIG.batch_size,
    val_size: float = _DEFAULT_DATASET_CONFIG.val_size,
    num_labels: int = _DEFAULT_MODEL_CONFIG.num_labels,
    label_names: list[str] | None = None,
    max_token_len: int = _DEFAULT_MODEL_CONFIG.max_token_len,
    lr: float = _DEFAULT_MODEL_CONFIG.lr,
    warmup_start_lr: float = _DEFAULT_MODEL_CONFIG.warmup_start_lr,
    warmup_epochs: int = _DEFAULT_MODEL_CONFIG.warmup_epochs,
    cache_dir: str | None = _DEFAULT_MODEL_CONFIG.cache_dir,
    max_epochs: int = _DEFAULT_TRAINER_CONFIG.max_epochs,
    seed: int = _DEFAULT_TRAINER_CONFIG.seed,
    fast_dev_run: bool = False,
) -> None:
    """Train a toxicity classification model using PyTorch Lightning.

    Trains a transformer-based multilabel classifier on the Jigsaw Toxic Comment
    Classification dataset with configurable parameters for dataset loading,
    model architecture, and training hyperparameters.

    Args:
        model_name: HuggingFace model name or local path (e.g., "distilbert-base-uncased").
        data_dir: Path to dataset directory containing train and test files.
        batch_size: Training and validation batch size.
        val_size: Validation split ratio between 0.0 and 1.0.
        num_labels: Number of output labels (auto-set if label_names provided).
        label_names: Toxicity labels to train on. Controls both dataset loading
            and model label names. If None, uses all JIGSAW_LABELS.
        max_token_len: Maximum input sequence length for tokenization.
        lr: Peak learning rate for Adam optimizer.
        warmup_start_lr: Initial learning rate for linear warmup.
        warmup_epochs: Number of warmup epochs before cosine decay.
        cache_dir: Model cache directory. If None, uses HuggingFace default.
        max_epochs: Maximum training epochs.
        seed: Random seed for reproducible training.
        fast_dev_run: Enable fast development run for debugging.

    Examples:
        Basic training with defaults:
            train("distilbert-base-uncased")

        Custom dataset and training config:
            train("bert-base-uncased", batch_size=128, max_epochs=50, lr=2e-5)

        Train on specific toxicity types:
            train("roberta-base", label_names=["toxic", "severe_toxic"])
    """
    # Set seed for reproducibility
    pl.seed_everything(seed, workers=True)

    # Adjust num_labels based on label_names if provided
    if label_names is not None:
        num_labels = len(label_names)

    datamodule = JigsawDataModule(
        data_dir=data_dir,
        batch_size=batch_size,
        val_size=val_size,
        labels=label_names,
    )

    model = Classifier(
        model_name=model_name,
        num_labels=num_labels,
        label_names=label_names,
        max_token_len=max_token_len,
        lr=lr,
        warmup_start_lr=warmup_start_lr,
        warmup_epochs=warmup_epochs,
        cache_dir=cache_dir,
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=3,
            verbose=True,
        ),
        ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            filename="{epoch:02d}-{val_loss:.4f}",
            save_top_k=1,
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        RichProgressBar(),
    ]

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        fast_dev_run=fast_dev_run,
        deterministic=True,
        logger=True,
    )

    # Train the model
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    auto_cli(train)
