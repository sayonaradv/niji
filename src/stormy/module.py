"""
Custom PyTorch Lightning Module for sequence classification using HuggingFace transformers.

This module provides:
- SequenceClassificationModule: a pl.LightningModule for multi-label text classification tasks.

Notes:
    - https://lightning.ai/docs/pytorch/stable/common/lightning_module.html
    - https://huggingface.co/docs/transformers/tasks/sequence_classification
"""

import lightning.pytorch as pl
import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from pydantic import ValidationError
from torch import Tensor
from torchmetrics.functional.classification import multilabel_accuracy
from transformers import AutoModelForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput

from stormy.config import ModuleConfig


class SequenceClassificationModule(pl.LightningModule):
    """A PyTorch Lightning module for multi-label text classification using transformer models.

        >>> # With AutoTokenizerDataModule
        >>> dm = AutoTokenizerDataModule(...)
        >>> module = SequenceClassificationModule(...)
        >>> trainer = pl.Trainer(
        ...     max_epochs=5,
        ...     accelerator="gpu",
        ...     devices=1,
        ...     precision="16-mixed"
        ... )
        >>> trainer.fit(module, dm)
        >>> trainer.test(module, dm)

    Metrics Logged:
        - train_loss: Cross-entropy loss during training
        - val_loss: Validation loss (logged with progress bar)
        - val_acc: Multi-label accuracy on validation set (logged with progress bar)
        - test_loss: Test loss (logged with progress bar)
        - test_acc: Multi-label accuracy on test set (logged with progress bar)

    Notes:
        - All parameters are validated using Pydantic models for type safety and clear errors
        - The module automatically configures the model for multi-label classification
        - Uses sigmoid activation internally for multi-label outputs
        - Compatible with mixed precision training for memory efficiency
        - Supports distributed training across multiple GPUs
        - Model weights are automatically saved in Lightning checkpoints
        - For detailed parameter specifications, see ModuleConfig documentation

    See Also:
        - ModuleConfig: For detailed parameter validation specifications
        - AutoTokenizerDataModule: Compatible DataModule for text data
        - PyTorch Lightning Module: https://lightning.ai/docs/pytorch/stable/common/lightning_module.html
        - HuggingFace Transformers: https://huggingface.co/docs/transformers/tasks/sequence_classification
    """

    def __init__(
        self,
        model_name: str,
        num_labels: int,
        learning_rate: float = 3e-5,
    ) -> None:
        """Initialize the SequenceClassificationModule with comprehensive parameter validation.

        Creates a new Lightning module for multi-label sequence classification by loading
        a pretrained transformer model and configuring it for the specified number of labels.
        All parameters are validated using Pydantic to ensure type safety and provide
        detailed error messages for invalid configurations.

        The module is automatically configured for multi-label classification with sigmoid
        activation and BCEWithLogitsLoss. The model weights are initialized from the
        pretrained checkpoint, with only the classification head being randomly initialized.

        For detailed parameter specifications, validation rules, and examples,
        see the class docstring above and ModuleConfig field definitions.

        Args:
            model_name: Name of the pretrained HuggingFace model (see class docstring for details)
            num_labels: Number of target labels (see class docstring for details)
            learning_rate: Learning rate for optimizer (see class docstring for details)
        """
        super().__init__()

        # Validate parameters using Pydantic model
        try:
            config = ModuleConfig(
                model_name=model_name,
                num_labels=num_labels,
                learning_rate=learning_rate,
            )
        except ValidationError as e:
            raise ValueError(
                f"Invalid configuration for SequenceClassificationModule: {e}"
            ) from e

        # Save hyperparameters for Lightning checkpointing and logging
        self.save_hyperparameters()

        # Store validated configuration values
        self.model_name_or_path = config.model_name
        self.num_labels = config.num_labels
        self.learning_rate = config.learning_rate

        # Load pretrained model with classification head
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name,
            num_labels=config.num_labels,
            problem_type="multi_label_classification",  # Configure for multi-label
        )

        # Ensure model is in training mode
        self.model.train()

    def forward(self, **inputs) -> SequenceClassifierOutput:
        """Forward pass through the transformer model.

        Performs a forward pass through the underlying transformer model with the provided
        inputs. This method is called during training, validation, and testing steps.

        Args:
            **inputs: Keyword arguments containing model inputs. Typically includes:
                - input_ids: Tokenized input sequences (Tensor of shape [batch_size, seq_len])
                - attention_mask: Attention mask for padding tokens (Tensor of shape [batch_size, seq_len])
                - labels: Target labels for loss computation (Tensor of shape [batch_size, num_labels])
                - Additional inputs depending on the specific transformer architecture

        Returns:
            SequenceClassifierOutput containing:
                - loss: Computed loss if labels are provided (Tensor)
                - logits: Raw model outputs before activation (Tensor of shape [batch_size, num_labels])
                - hidden_states: Hidden states from all layers (if output_hidden_states=True)
                - attentions: Attention weights from all layers (if output_attentions=True)

        Notes:
            - Loss is automatically computed when labels are provided in inputs
            - Logits are raw outputs; apply sigmoid for probabilities in multi-label tasks
            - This method delegates to the underlying HuggingFace model's forward method
        """
        return self.model(**inputs)

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        """Execute a single training step and log metrics.

        Performs forward pass, computes loss, and logs training metrics for one batch.
        This method is called automatically by the PyTorch Lightning trainer during
        the training loop.

        Args:
            batch: Dictionary containing batch data with keys:
                - input_ids: Tokenized input sequences
                - attention_mask: Attention mask for padding
                - labels: Target labels for loss computation
                - Additional keys depending on the DataModule configuration
            batch_idx: Index of the current batch within the epoch (used by Lightning)

        Returns:
            Training loss tensor that will be used by Lightning for backpropagation.
            The loss is automatically computed by the model when labels are provided.

        Logged Metrics:
            - train_loss: Cross-entropy loss for the current batch

        Notes:
            - Loss computation and backpropagation are handled automatically by Lightning

        See Also:
            https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#training
        """
        outputs = self(**batch)
        self.log("train_loss", outputs.loss)
        return outputs.loss

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        """Execute a single validation step and log metrics.

        Performs forward pass and logs validation metrics for one batch. This method
        is called automatically by the PyTorch Lightning trainer during validation.
        No gradients are computed during validation steps.

        Args:
            batch: Dictionary containing batch data (same format as training_step)
            batch_idx: Index of the current batch within the validation set

        Logged Metrics:
            - val_loss: Validation loss for the current batch (shown in progress bar)
            - val_acc: Multi-label accuracy for the current batch (shown in progress bar)

        Notes:
            - Model is automatically set to eval() mode during validation
            - No gradients are computed or accumulated during this step
            - Metrics are aggregated across all validation batches by Lightning
            - This method delegates to _shared_eval_step for consistency

        See Also:
            https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#validation
        """
        self._shared_eval_step(batch, stage="val")

    def test_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        """Execute a single test step and log metrics.

        Performs forward pass and logs test metrics for one batch. This method
        is called automatically by the PyTorch Lightning trainer during testing.
        No gradients are computed during test steps.

        Args:
            batch: Dictionary containing batch data (same format as training_step)
            batch_idx: Index of the current batch within the test set

        Logged Metrics:
            - test_loss: Test loss for the current batch (shown in progress bar)
            - test_acc: Multi-label accuracy for the current batch (shown in progress bar)

        Notes:
            - Model is automatically set to eval() mode during testing
            - No gradients are computed or accumulated during this step
            - Metrics are aggregated across all test batches by Lightning
            - This method delegates to _shared_eval_step for consistency
            - Used for final model evaluation after training completion

        See Also:
            https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#testing
        """
        self._shared_eval_step(batch, stage="test")

    def _shared_eval_step(self, batch: dict[str, Tensor], stage: str) -> None:
        """Shared logic for validation and test steps to ensure consistency.

        This internal method contains the common evaluation logic used by both
        validation_step and test_step methods. It computes loss and accuracy
        metrics and logs them with appropriate stage prefixes.

        Args:
            batch: Dictionary containing batch data with model inputs and labels
            stage: Either "val" or "test" to indicate which evaluation stage this is.
                Used for metric logging prefixes (e.g., "val_loss" vs "test_loss").

        Logged Metrics:
            - {stage}_loss: Loss for the current batch (with progress bar)
            - {stage}_acc: Multi-label accuracy for the current batch (with progress bar)

        Implementation Details:
            - Uses multilabel_accuracy from torchmetrics for consistent accuracy computation
            - Applies sigmoid activation internally for multi-label probability calculation
            - Both metrics are logged with prog_bar=True for real-time monitoring
            - Loss and logits are extracted from the model's SequenceClassifierOutput

        Notes:
            - This method ensures validation and test evaluation use identical logic
            - Multi-label accuracy computes the percentage of exactly correct label sets
            - The accuracy metric accounts for the multi-label nature of the task
            - All metrics are automatically aggregated by Lightning across batches
        """
        outputs = self(**batch)
        loss, logits = outputs.loss, outputs.logits

        # Compute multi-label accuracy using torchmetrics
        acc = multilabel_accuracy(logits, batch["labels"], num_labels=self.num_labels)

        # Log metrics with progress bar for real-time monitoring
        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_acc", acc, prog_bar=True)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Configure the optimizer for training.

        Sets up the Adam optimizer with the specified learning rate for training
        the transformer model. This method is called automatically by PyTorch
        Lightning during trainer setup.

        The Adam optimizer is chosen for its robust performance on transformer
        models and good handling of sparse gradients. It uses the default
        parameters (beta1=0.9, beta2=0.999, eps=1e-8) which work well for
        most transformer fine-tuning scenarios.

        Returns:
            Configured Adam optimizer with the specified learning rate applied
            to all model parameters. The optimizer will be used by Lightning
            for gradient-based parameter updates during training.

        Notes:
            - Lightning automatically handles optimizer.step() and optimizer.zero_grad()
            - The optimizer operates on all model parameters including pretrained weights
            - Different learning rates for different parameter groups can be configured

        See Also:
            https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.configure_optimizers
        """
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
