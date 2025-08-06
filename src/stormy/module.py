"""PyTorch Lightning Module for sequence classification using HuggingFace transformers.

This module provides SequenceClassificationModule, a Lightning module for multi-label
text classification tasks using pretrained transformer models.

References:
    - https://lightning.ai/docs/pytorch/stable/common/lightning_module.html
    - https://huggingface.co/docs/transformers/tasks/sequence_classification
"""

import lightning.pytorch as pl
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from pydantic import ValidationError
from torch import Tensor
from torchmetrics.functional.classification import multilabel_accuracy
from transformers import AutoModelForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput

from stormy.config import ModuleConfig


class SequenceClassificationModule(pl.LightningModule):
    """PyTorch Lightning module for multi-label text classification using transformers.

    Automatically configures pretrained transformer models for multi-label classification
    with sigmoid activation and BCEWithLogitsLoss. Supports mixed precision training
    and distributed training across multiple GPUs.

    Metrics logged:
        - train_loss: Cross-entropy loss during training
        - val_loss/val_acc: Validation loss and multi-label accuracy (with progress bar)
        - test_loss/test_acc: Test loss and multi-label accuracy (with progress bar)

    Examples:
        >>> module = SequenceClassificationModule(
        ...     model_name="bert-base-uncased",
        ...     num_labels=6,
        ...     learning_rate=3e-5
        ... )
        >>> trainer = pl.Trainer(max_epochs=5, accelerator="gpu")
        >>> trainer.fit(module, datamodule)

    Note:
        All parameters are validated using Pydantic models for type safety.
    """

    def __init__(self, model_name: str, num_labels: int) -> None:
        """Initialize the SequenceClassificationModule.

        Loads a pretrained transformer model and configures it for multi-label
        sequence classification with the specified number of labels.

        Args:
            model_name: Name of the pretrained HuggingFace model to use.
            num_labels: Number of target labels in the classification task.

        Raises:
            ValueError: If any parameter fails validation.
        """
        super().__init__()

        try:
            config = ModuleConfig(model_name=model_name, num_labels=num_labels)
        except ValidationError as e:
            raise ValueError(
                f"Invalid configuration for SequenceClassificationModule: {e}"
            ) from e

        self.save_hyperparameters()

        self.model_name_or_path = config.model_name
        self.num_labels = config.num_labels

        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name,
            num_labels=config.num_labels,
            problem_type="multi_label_classification",
        )

        self.model.train()

    def forward(self, **inputs) -> SequenceClassifierOutput:
        """Forward pass through the transformer model.

        Args:
            **inputs: Model inputs including input_ids, attention_mask, and optionally labels.

        Returns:
            SequenceClassifierOutput containing loss (if labels provided) and logits.

        Note:
            Loss is automatically computed when labels are provided in inputs.
        """
        return self.model(**inputs)

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        """Execute a single training step and log metrics.

        Args:
            batch: Dictionary containing input_ids, attention_mask, and labels.
            batch_idx: Index of the current batch within the epoch.

        Returns:
            Training loss tensor for backpropagation.
        """
        outputs = self(**batch)
        self.log("train_loss", outputs.loss, prog_bar=True, on_step=True, on_epoch=True)
        return outputs.loss

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        """Execute a single validation step and log metrics.

        Args:
            batch: Dictionary containing input_ids, attention_mask, and labels.
            batch_idx: Index of the current batch within the validation set.

        Note:
            Logs val_loss and val_acc metrics with progress bar.
        """
        self._shared_eval_step(batch, stage="val")

    def test_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        """Execute a single test step and log metrics.

        Args:
            batch: Dictionary containing input_ids, attention_mask, and labels.
            batch_idx: Index of the current batch within the test set.

        Note:
            Logs test_loss and test_acc metrics with progress bar.
        """
        self._shared_eval_step(batch, stage="test")

    def _shared_eval_step(self, batch: dict[str, Tensor], stage: str) -> None:
        """Shared logic for validation and test steps.

        Computes loss and multi-label accuracy metrics and logs them with
        appropriate stage prefixes.

        Args:
            batch: Dictionary containing batch data with model inputs and labels.
            stage: Either "val" or "test" for metric logging prefixes.

        Note:
            Uses torchmetrics multilabel_accuracy for consistent computation.
        """
        outputs = self(**batch)
        loss, logits = outputs.loss, outputs.logits
        acc = multilabel_accuracy(
            logits, batch["labels"], num_labels=self.num_labels, threshold=0.5
        )

        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{stage}_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Configure the optimizer for training.

        Returns:
            Adam optimizer with the specified learning rate applied to all
            model parameters.

        Note:
            Lightning automatically handles optimizer.step() and optimizer.zero_grad().
        """
        optimizer = self.optimizer(self.model.parameters())
        scheduler = self.scheduler(optimizer)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
