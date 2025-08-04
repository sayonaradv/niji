"""Stormy: Open-source Python library for toxicity detection using BERT models.

Stormy is a machine learning library that provides:
- Fine-tuned BERT models for accurate toxicity detection
- Real-time text analysis capabilities
- PyTorch Lightning integration for training and inference
- Support for multi-label classification tasks
- Built specifically for the Jigsaw Toxic Comment Classification challenge

The library is designed to help maintain healthy online environments by identifying
and handling inappropriate content with state-of-the-art natural language processing.

Key Components:
    - AutoTokenizerDataModule: Data loading and preprocessing for HuggingFace datasets
    - SequenceClassificationModule: Lightning module for transformer-based classification
    - Configuration models: Pydantic-based parameter validation and documentation
    - Utility functions: Helper functions for tensor operations and data handling

Example:
    Basic usage for toxicity detection:

    >>> from stormy.datamodule import create_jigsaw_datamodule
    >>> from stormy.module import SequenceClassificationModule
    >>> import lightning.pytorch as pl
    >>>
    >>> # Create data module for Jigsaw dataset
    >>> dm = create_jigsaw_datamodule()
    >>>
    >>> # Create classification module
    >>> module = SequenceClassificationModule(
    ...     model_name="bert-base-uncased",
    ...     num_labels=6,  # Jigsaw has 6 toxicity categories
    ...     learning_rate=3e-5
    ... )
    >>>
    >>> # Train the model
    >>> trainer = pl.Trainer(max_epochs=3, accelerator="auto")
    >>> trainer.fit(module, dm)

See Also:
    - Documentation: https://github.com/dbozbay/stormy-ai
    - Jigsaw Challenge: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
    - PyTorch Lightning: https://lightning.ai/docs/pytorch/stable/
    - HuggingFace Transformers: https://huggingface.co/docs/transformers/
"""

__version__ = "1.0.1a9"
__author__ = "Cozmo"
__email__ = "107803920+Cozmo18@users.noreply.github.com"

# Public API exports
from stormy.datamodule import AutoTokenizerDataModule, create_jigsaw_datamodule
from stormy.module import SequenceClassificationModule
from stormy.config import DataModuleConfig, ModuleConfig

__all__ = [
    "AutoTokenizerDataModule",
    "SequenceClassificationModule",
    "DataModuleConfig",
    "ModuleConfig",
    "create_jigsaw_datamodule",
    "main",
]


def main() -> None:
    """Entry point for the Stormy command-line interface.

    Prints a welcome message with the Stormy logo. This function serves as the
    main entry point when the package is run as `stormy` from the command line.

    In future versions, this will be expanded to provide CLI functionality for:
    - Running toxicity detection on text files
    - Training models with custom datasets
    - Evaluating model performance metrics
    - Converting models to different formats (ONNX, TensorRT, etc.)

    Returns:
        None

    Example:
        Run from command line:
        ```bash
        stormy
        ```

        Or programmatically:
        ```python
        from stormy import main
        main()
        ```

    Notes:
        The storm emoji (⛈️) represents the library's ability to detect and handle
        the "stormy weather" of toxic online content.
    """
    print("Hello from Stormy! ⛈️")
