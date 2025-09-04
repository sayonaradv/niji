"""PyTorch Lightning module for multilabel text classification.

This module provides a Classifier class that wraps transformer models for
multilabel text classification tasks, specifically designed for toxicity
detection and content moderation applications.
"""

from typing import cast

import lightning.pytorch as pl
import torch
from lightning.pytorch.utilities.types import (
    STEP_OUTPUT,
    LRScheduler,
    OptimizerLRSchedulerConfig,
)
from torch import Tensor
from torch.nn import functional as F
from torch.optim import Optimizer
from torchmetrics.functional.classification import multilabel_accuracy

from ruffle.schedulers import LinearWarmupCosineAnnealingLR
from ruffle.types import Batch, TensorDict, TextInput
from ruffle.utils import get_model_and_tokenizer


class Classifier(pl.LightningModule):
    """PyTorch Lightning module for multilabel text classification.

    A wrapper around transformer models that provides training, validation,
    and testing capabilities for multilabel classification tasks. Uses binary
    cross-entropy loss with sigmoid activation for multi-label prediction.

    Attributes:
        model: The underlying transformer model for classification.
        tokenizer: Tokenizer associated with the model.
    """

    def __init__(
        self,
        model_name: str,
        num_labels: int = 6,
        label_names: list[str] | None = None,
        max_token_len: int = 256,
        lr: float = 3e-5,
        warmup_start_lr: float = 3e-6,
        warmup_epochs: int = 5,
        cache_dir: str | None = "data",
    ) -> None:
        """Initialize the Classifier module.

        Args:
            model_name: Name or path of the pre-trained transformer model.
            num_labels: Number of labels for multilabel classification.
            label_names: Optional list of human-readable label names. If provided,
                must have the same length as num_labels.
            max_token_len: Maximum sequence length for tokenization.
            lr: Peak learning rate for the optimizer.
            warmup_start_lr: Initial learning rate for warmup phase.
            warmup_epochs: Number of epochs for learning rate warmup.
            cache_dir: Directory to cache downloaded models. If None, uses default
                transformers cache directory.

        Raises:
            ValueError: If label_names length doesn't match num_labels.
        """
        super().__init__()
        self.save_hyperparameters()
        self._validate_labels()
        self.model, self.tokenizer = get_model_and_tokenizer(
            self.hparams["model_name"],
            cache_dir=self.hparams["cache_dir"],
            num_labels=self.hparams["num_labels"],
        )
        self.model.train()

    def _validate_labels(self) -> None:
        """Validate that label_names length matches num_labels if provided.

        Raises:
            ValueError: If label_names is provided but has different length than num_labels.
        """
        num_labels = self.hparams["num_labels"]
        label_names = self.hparams["label_names"]

        if label_names is not None and len(label_names) != num_labels:
            raise ValueError(
                f"Length of label_names ({len(label_names)}) must match num_labels ({num_labels})."
            )

    def configure_model(self) -> None:
        """Configure the model for optimized training.

        Compiles the model using PyTorch's compile functionality to improve
        training speed and efficiency.
        """
        self.model.compile()  # improves training speed

    def forward(self, text: TextInput, labels: Tensor | None = None) -> TensorDict:  # type: ignore[override]
        """Forward pass through the model.

        Tokenizes input text, runs it through the model, and applies sigmoid
        activation to get probabilities. Optionally computes loss if labels are provided.

        Args:
            text: Input text(s) as string or list of strings.
            labels: Optional ground truth labels for computing loss. Should be
                FloatTensor with shape (batch_size, num_labels) containing binary values.

        Returns:
            Dictionary containing:
                - 'outputs': Model predictions after sigmoid activation with shape
                  (batch_size, num_labels).
                - 'loss': Binary cross-entropy loss if labels provided, otherwise not included.
        """
        inputs: TensorDict = self.tokenizer(
            text,
            max_length=self.hparams["max_token_len"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs: Tensor = self.model(**inputs).logits
        outputs = torch.sigmoid(outputs)

        if labels is not None:
            labels = labels.to(self.device)
            loss: Tensor = F.binary_cross_entropy(outputs, labels)
            return {"outputs": outputs, "loss": loss}
        else:
            return {"outputs": outputs}

    def training_step(self, batch: Batch, batch_idx: int) -> STEP_OUTPUT:  # type: ignore[override]
        """Execute a single training step.

        Args:
            batch: Training batch containing 'text' and 'labels' keys.
            batch_idx: Index of the current batch.

        Returns:
            Training loss tensor for backpropagation.
        """
        loss: Tensor = self(**batch)["loss"]
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: Batch, batch_idx: int) -> STEP_OUTPUT:  # type: ignore[override]
        """Execute a single validation step.

        Args:
            batch: Validation batch containing 'text' and 'labels' keys.
            batch_idx: Index of the current batch.

        Returns:
            None.
        """
        self._shared_eval_step(batch, stage="val")
        return None

    def test_step(self, batch: Batch, batch_idx: int) -> STEP_OUTPUT:  # type: ignore[override]
        """Execute a single test step.

        Args:
            batch: Test batch containing 'text' and 'labels' keys.
            batch_idx: Index of the current batch.

        Returns:
            None.
        """
        self._shared_eval_step(batch, stage="test")
        return None

    def _shared_eval_step(self, batch: Batch, stage: str) -> STEP_OUTPUT:
        """Shared logic for validation and test steps.

        Computes predictions, loss, and multilabel accuracy for evaluation.
        Logs metrics with the specified stage prefix.

        Args:
            batch: Evaluation batch containing 'text' and 'labels' keys.
            stage: Stage name ("val" or "test") for metric logging.

        Returns:
            None.
        """
        preds: Tensor
        loss: Tensor
        labels: Tensor
        acc: Tensor

        preds, loss = self(**batch).values()
        labels = cast(Tensor, batch["labels"])
        acc = multilabel_accuracy(preds, labels, num_labels=self.hparams["num_labels"])

        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{stage}_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        """Configure optimizer and learning rate scheduler.

        Sets up Adam optimizer with linear warmup followed by cosine annealing
        learning rate schedule for stable training.

        Returns:
            Dictionary containing optimizer and lr_scheduler configurations
            for PyTorch Lightning.
        """
        optimizer: Optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.hparams["lr"]
        )
        scheduler: LRScheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=self.hparams["warmup_epochs"],
            warmup_start_lr=self.hparams["warmup_start_lr"],
            max_epochs=self.trainer.max_epochs,
        )

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
