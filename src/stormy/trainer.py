"""
PyTorch Lightning CLI implementation for training sequence classification models.

This module provides:
- MyLightningCLI: Custom Lightning CLI with MLFlow logging and automatic config management
- MLFlowSaveConfigCallback: Custom callback for saving configs in MLFlow run directories
- cli_main: Entry point function for command-line training

Key Features:
    - MLFlow experiment tracking with automatic model logging
    - Early stopping and model checkpointing with sensible defaults
    - Rich progress bars for training visualization
    - Automatic config saving in run-specific directories
    - Deterministic training with mixed precision support

Notes:
    - Uses medium precision for float32 matrix multiplication for performance
    - All configurations are automatically linked between data and model components
    - Compatible with both local and remote MLFlow tracking servers
    - Supports resuming from checkpoints and hyperparameter optimization

See Also:
    - Lightning CLI: https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html
    - MLFlow: https://mlflow.org/docs/latest/tracking.html
    - PyTorch Lightning Callbacks: https://lightning.ai/docs/pytorch/stable/extensions/callbacks.html
"""

import os

import torch
from jsonargparse import lazy_instance
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar
from lightning.pytorch.cli import ArgsType, LightningCLI, SaveConfigCallback
from lightning.pytorch.core import LightningModule
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.trainer import Trainer

from stormy.datamodule import AutoTokenizerDataModule
from stormy.module import SequenceClassificationModule

# see https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
torch.set_float32_matmul_precision("medium")


class MLFlowSaveConfigCallback(SaveConfigCallback):
    """Custom SaveConfigCallback that saves config files in MLFlow run directories."""

    def __init__(self, *args, **kwargs):
        # Disable saving to log_dir to prevent config.yaml in root directory
        super().__init__(*args, save_to_log_dir=False, **kwargs)

    def save_config(
        self, trainer: Trainer, pl_module: LightningModule, stage: str
    ) -> None:
        if isinstance(trainer.logger, MLFlowLogger):
            # Get relevant MLFlow directories
            save_dir = str(trainer.logger.save_dir)
            experiment_id = str(trainer.logger.experiment_id)
            run_id = str(trainer.logger.run_id)

            # Construct the run-specific directory path
            run_dir = os.path.join(save_dir, experiment_id, run_id)

            # Ensure the directory exists
            os.makedirs(run_dir, exist_ok=True)

            # Save the config in the run directory
            config_path = os.path.join(run_dir, self.config_filename)
            self.parser.save(self.config, config_path, skip_none=False, overwrite=True)

            # Also log the config as an artifact to MLFlow
            trainer.logger.experiment.log_artifact(run_id, config_path)
        else:
            # Fall back to default behavior for other loggers
            super().save_config(trainer, pl_module, stage)


class MyLightningCLI(LightningCLI):
    """Custom Lightning CLI with MLFlow logging and automatic callback configuration.

    This CLI extends the base LightningCLI to provide a complete training setup with:
        - Early stopping based on validation loss
        - Model checkpointing with descriptive filenames
        - Automatic argument linking between data and model components
        - MLFlow integration for experiment tracking

    The CLI automatically configures:
        - EarlyStopping: Monitors validation loss with patience of 3 epochs
        - ModelCheckpoint: Saves top 3 models with epoch and loss in filename
        - Argument linking: Connects label columns count to model num_labels
        - Model name consistency: Ensures data and model use same tokenizer

    Example Usage:
        ```bash
        # Train with default config
        python -m stormy.trainer fit

        # Train with custom config file
        python -m stormy.trainer fit --config configs/my-config.yaml

        # Override specific parameters
        python -m stormy.trainer fit --trainer.max_epochs 20 --model.learning_rate 1e-4

        # Test a trained model
        python -m stormy.trainer test --ckpt_path path/to/checkpoint.ckpt
        ```

    Configuration Features:
        - All parameters can be specified via YAML config files
        - Command line arguments override config file values
        - Automatic validation of parameter combinations and types
        - Support for nested configuration structures

    See Also:
        - Lightning CLI: https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html
        - CLI Advanced: https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_advanced.html
        - MLFlow: https://mlflow.org/docs/latest/tracking.html
    """

    def add_arguments_to_parser(self, parser):
        """Add and configure CLI arguments with sensible defaults.

        Configures early stopping, model checkpointing, and automatic argument linking
        to provide a complete training setup out of the box. All defaults are chosen
        based on best practices for transformer fine-tuning.

        Args:
            parser: The argument parser to which Lightning class arguments are added.
                This is provided automatically by the Lightning CLI framework.

        Configuration Details:
            Early Stopping:
                - Monitors validation loss for model improvement
                - Stops training after 3 epochs without improvement
                - Uses minimum mode (lower loss is better)
                - Provides verbose logging of stopping decisions

            Model Checkpointing:
                - Monitors validation loss for best model selection
                - Saves top 3 checkpoints to prevent overfitting
                - Uses descriptive filenames with epoch and loss values
                - Enables verbose logging of checkpoint saves

            Argument Linking:
                - Automatically sets model.num_labels from data.label_columns length
                - Ensures data.model_name matches model.model_name for tokenizer consistency
                - Prevents configuration mismatches between components

        Notes:
            - All defaults can be overridden via command line or config file
            - Parser configuration happens before training begins
            - Links are computed dynamically based on provided configurations
            - Validation occurs automatically before training starts

        See Also:
            - EarlyStopping: https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.EarlyStopping.html
            - ModelCheckpoint: https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html
        """
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
                "checkpoint.filename": "finetuned-{epoch:02d}-{val_loss:.4f}",
                "checkpoint.verbose": True,
                "checkpoint.save_top_k": 3,
            }
        )

        parser.link_arguments(
            "data.label_columns",
            "model.num_labels",
            compute_fn=lambda label_columns: len(label_columns),
        )
        parser.link_arguments("model.model_name", "data.model_name")


def cli_main(args: ArgsType = None):
    """Main entry point for the Lightning CLI training interface.

    Creates and runs a custom Lightning CLI configured for sequence classification
    with MLFlow experiment tracking, automatic checkpointing, and comprehensive
    logging. This function serves as the primary interface for training models
    from the command line or programmatically.

    The CLI is preconfigured with:
        - SequenceClassificationModule for transformer-based text classification
        - AutoTokenizerDataModule for automatic text preprocessing and tokenization
        - MLFlow logger with experiment tracking and model logging
        - Rich progress bars for enhanced training visualization
        - Deterministic training for reproducible results
        - Custom config callback for proper MLFlow integration

    Example Usage:
        ```python
        # Run with default arguments
        cli_main()

        # Run with custom arguments
        cli_main(["fit", "--trainer.max_epochs", "20"])

        # Run from command line
        # python -m stormy.trainer fit --config config.yaml
        ```

    Args:
        args: Optional command line arguments to pass to the CLI. If None,
            arguments are automatically parsed from sys.argv. Can be:
            - None: Parse from command line automatically
            - List[str]: Explicit command line arguments
            - ArgumentParser namespace: Parsed arguments object

    Configuration Features:
        - Seed: Fixed at 1234 for reproducible experiments
        - Max Epochs: Default 10 epochs with early stopping
        - Precision: Automatic mixed precision for memory efficiency
        - Deterministic: Ensures reproducible results across runs
        - Logging: MLFlow with automatic model and artifact logging
        - Callbacks: Rich progress bars and custom config saving

    Integration Details:
        - Uses MLFlowSaveConfigCallback for proper config management
        - Automatically logs models to MLFlow after training
        - Supports distributed training across multiple GPUs
        - Compatible with hyperparameter optimization frameworks
        - Integrates with Lightning's automatic logging and metrics

    Notes:
        - The CLI automatically validates all configuration parameters
        - Model and data module classes are linked via argument parsing
        - All Lightning trainer features are available via command line
        - Checkpoints and logs are saved to configurable directories
        - The function doesn't return values but runs training to completion

    See Also:
        - LightningCLI: https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html
        - MLFlow Logger: https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.MLFlowLogger.html
        - Trainer: https://lightning.ai/docs/pytorch/stable/common/trainer.html
    """
    MyLightningCLI(
        SequenceClassificationModule,
        AutoTokenizerDataModule,
        args=args,
        seed_everything_default=1234,
        save_config_callback=MLFlowSaveConfigCallback,
        trainer_defaults={
            "max_epochs": 10,
            "deterministic": True,
            "logger": lazy_instance(
                MLFlowLogger,
                experiment_name="lightning_logs",
                log_model="all",
            ),
            "callbacks": [lazy_instance(RichProgressBar)],
        },
    )


if __name__ == "__main__":
    cli_main()
