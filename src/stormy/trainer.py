"""PyTorch Lightning CLI for training sequence classification models.

This module provides a command-line interface for training and evaluating sequence
classification models using PyTorch Lightning. Includes automatic parameter linking
between data and model configurations, and pre-configured defaults optimized for
toxicity detection tasks.

Features:
    - Automatic parameter linking between data and model configurations
    - Rich progress bars and model summaries
    - Early stopping and model checkpointing
    - Mixed precision training for memory efficiency
    - Deterministic training for reproducible results
    - TensorBoard logging for experiment tracking

Examples:
    Train with configuration file:
    ```bash
    uv run trainer fit --config configs/jigsaw-config.yaml
    ```

    Train with command line arguments:
    ```bash
    uv run trainer fit \
        --model.model_name bert-base-uncased \
        --model.learning_rate 3e-5 \
        --data.dataset_name mat55555/jigsaw_toxic_comment \
        --trainer.max_epochs 5
    ```

    Test a trained model:
    ```bash
    uv run trainer test --ckpt_path lightning_logs/version_0/checkpoints/best.ckpt
    ```

References:
    - PyTorch Lightning CLI: https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html
"""

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from lightning.pytorch.cli import ArgsType, LightningArgumentParser, LightningCLI

from stormy.datamodule import AutoTokenizerDataModule
from stormy.module import SequenceClassificationModule

# Optimize tensor operations for better performance on modern hardware
# See https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
torch.set_float32_matmul_precision("medium")


class MyLightningCLI(LightningCLI):
    """Custom Lightning CLI with automatic parameter linking.

    Extends the standard PyTorch Lightning CLI with automatic parameter linking
    to reduce configuration duplication. Ensures the number of labels in the model
    matches the number of label columns in the data module, and both use the same
    transformer model.

    Parameter links:
        - data.label_columns → model.num_labels: Sets number of output labels
          based on the number of label columns in the dataset
        - model.model_name → data.model_name: Ensures both components use the
          same pretrained transformer model

    Examples:
        With parameter linking, you only need:
        ```yaml
        model:
          model_name: bert-base-uncased
          learning_rate: 3e-5
        data:
          dataset_name: mat55555/jigsaw_toxic_comment
          label_columns: [toxic, severe_toxic, obscene, threat, insult, identity_hate]
          # model_name and num_labels are automatically linked
        ```

    Note:
        Parameter linking happens automatically during CLI initialization and
        prevents common configuration mistakes.
    """

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        """Add automatic parameter linking to the CLI parser.

        Sets up automatic parameter linking between data module and model
        configurations to ensure related parameters stay synchronized.

        Args:
            parser: The Lightning argument parser to configure with parameter links.

        Note:
            Called automatically by the Lightning CLI framework during initialization.
        """
        parser.link_arguments(
            "data.label_columns",
            "model.num_labels",
            compute_fn=lambda label_columns: len(label_columns),
        )

        parser.link_arguments("model.model_name", "data.model_name")


def cli_main(args: ArgsType = None) -> None:
    """Main entry point for the training CLI.

    Creates and runs the Lightning CLI with pre-configured defaults optimized
    for sequence classification training. Sets up the complete training pipeline
    including data loading, model training, callbacks, and logging.

    Args:
        args: Optional command line arguments. If None, arguments are parsed
            from sys.argv. Mainly used for testing or programmatic invocation.

    Trainer defaults:
        - max_epochs: 10 (prevents overfitting)
        - deterministic: True (ensures reproducible results)
        - precision: "16-mixed" (memory efficient mixed precision)
        - seed_everything_default: 1234 (fixed seed for reproducibility)
        - callbacks: Early stopping, model checkpointing, rich UI components
        - logger: TensorBoard logging enabled

    Examples:
        Train with configuration file:
        ```bash
        uv run trainer fit --config configs/bert-base.yaml
        ```

        Train with command line arguments:
        ```bash
        uv run trainer fit \
            --model.model_name prajjwal1/bert-tiny \
            --model.learning_rate 5e-5 \
            --data.dataset_name mat55555/jigsaw_toxic_comment \
            --trainer.max_epochs 5
        ```

        Test a trained model:
        ```bash
        uv run trainer test \
            --ckpt_path lightning_logs/version_0/checkpoints/epoch=02-val_loss=0.1234.ckpt
        ```

    File outputs:
        - Model checkpoints: lightning_logs/version_X/checkpoints/
        - TensorBoard logs: lightning_logs/version_X/
        - Training logs: console output and tensorboard scalars

    Raises:
        SystemExit: If invalid command line arguments are provided.
        RuntimeError: If CUDA is requested but not available.
        ValidationError: If model or data configuration is invalid.
    """
    MyLightningCLI(
        model_class=SequenceClassificationModule,
        datamodule_class=AutoTokenizerDataModule,
        trainer_class=pl.Trainer,
        seed_everything_default=1234,
        args=args,
        trainer_defaults={
            "max_epochs": 10,
            "deterministic": True,
            "precision": "16-mixed",
            "callbacks": [
                EarlyStopping(monitor="val_loss", mode="min", patience=3, verbose=True),
                ModelCheckpoint(
                    monitor="val_loss",
                    mode="min",
                    filename="{epoch:02d}-{val_loss:.4f}",
                    save_top_k=1,
                    verbose=True,
                ),
                RichModelSummary(max_depth=-1),
                RichProgressBar(),
            ],
            "logger": True,
        },
    )


if __name__ == "__main__":
    cli_main()
