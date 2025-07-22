"""
Trainer module for running the StormyTransformer with PyTorch Lightning CLI.

This module provides a CLI entry point for training and evaluating the StormyTransformer model using the JigsawDataModule.
It supports configuration of early stopping and model checkpointing callbacks, as well as experiment logging with MLFlowLogger.

Features:
    - PyTorch Lightning CLI integration: https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html
    - EarlyStopping callback: https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.EarlyStopping.html
    - ModelCheckpoint callback: https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html
    - MLFlowLogger for experiment tracking: https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.MLFlowLogger.html
"""

import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.cli import LightningCLI

from .datamodule import JigsawDataModule
from .module import StormyTransformer

# see https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
torch.set_float32_matmul_precision("medium")


class StormyCLI(LightningCLI):
    """Custom LightningCLI with support for EarlyStopping, ModelCheckpoint, and argument linking.

    This CLI allows configuration of EarlyStopping and ModelCheckpoint callbacks via command-line arguments. It also links certain arguments between the data and model modules for consistency.

    - EarlyStopping and ModelCheckpoint are added as configurable forced callbacks.
    - Argument linking ensures num_labels and model_name_or_path are consistent between modules.
    - See: https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_expert.html#configure-forced-callbacks
    """

    def add_arguments_to_parser(self, parser):
        parser.add_lightning_class_args(EarlyStopping, "early_stopping")
        parser.set_defaults(
            {
                "early_stopping.monitor": "val_loss",
                "early_stopping.patience": 3,
                "early_stopping.mode": "min",
                "early_stopping.verbose": True,
            }
        )

        parser.add_lightning_class_args(ModelCheckpoint, "checkpoint")
        parser.set_defaults(
            {
                "checkpoint.monitor": "val_loss",
                "checkpoint.filename": "jigsaw-{epoch:02d}-{val_loss:.3f}",
                "checkpoint.verbose": True,
                "checkpoint.save_top_k": 3,
            }
        )

        # Force num_labels to always be linked to len(labels)
        parser.link_arguments(
            "data.labels", "model.num_labels", compute_fn=lambda labels: len(labels)
        )
        parser.link_arguments("model.model_name_or_path", "data.model_name_or_path")


def cli_main():
    """Entry point for training or evaluating StormyTransformer with LightningCLI.

    Sets up the CLI with StormyTransformer and JigsawDataModule, configures MLFlowLogger for experiment tracking, and applies default trainer settings.

    Notes:
        https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.MLFlowLogger.html
    """

    mlflow = {
        "class_path": "lightning.pytorch.loggers.MLFlowLogger",
        "init_args": {
            "experiment_name": "lightning_logs",
            "tracking_uri": "file:./ml-runs",
            "log_model": "all",
        },
    }
    StormyCLI(
        StormyTransformer,
        JigsawDataModule,
        trainer_defaults={
            "max_epochs": 10,
            "deterministic": True,
            "logger": mlflow,
        },
        seed_everything_default=18,
    )


if __name__ == "__main__":
    cli_main()
