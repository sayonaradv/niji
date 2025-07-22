"""
LightningModule for multilabel toxic comment classification using HuggingFace Transformers.

Implements a PyTorch LightningModule for toxic comment classification with a HuggingFace model.
"""

from typing import Any

import lightning.pytorch as pl
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import Tensor
from torchmetrics.functional.classification import multilabel_accuracy
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
)


class StormyTransformer(pl.LightningModule):
    """A custom LightningModule for multilabel toxic comment classification."""

    def __init__(
        self,
        model_name_or_path: str = "distilbert-base-cased",
        num_labels: int = 6,
        learning_rate: float = 2e-5,
    ) -> None:
        """Initialize the StormyTransformer module.

        Args:
            model_name_or_path (str): Name or path to the pretrained HuggingFace model.
            num_labels (int): Number of output labels for multilabel classification.
            learning_rate (float): Learning rate for the optimizer.
        """
        super().__init__()
        self.save_hyperparameters()

        self.model_name_or_path = model_name_or_path
        self.num_labels = num_labels
        self.learning_rate = learning_rate

        self.config = AutoConfig.from_pretrained(
            model_name_or_path,
            num_labels=num_labels,
            problem_type="multi_label_classification",
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, config=self.config
        )

    def forward(self, **inputs: Any) -> Any:
        """Forward pass through the HuggingFace model.

        Args:
            **inputs: Tokenized input data as keyword arguments (e.g., input_ids, attention_mask, etc.).

        Returns:
            transformers.modeling_outputs.SequenceClassifierOutput: Model outputs including loss (if labels provided) and logits.

        Notes:
            https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#forward
        """
        return self.model(**inputs)

    def training_step(self, batch: dict, batch_idx: int) -> STEP_OUTPUT:
        """Performs a single training step.

        Args:
            batch (dict): Batch containing tokenized inputs and labels.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Training loss.

        Notes:
            https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#training_step
        """
        outputs = self(**batch)
        loss = outputs[0]
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> STEP_OUTPUT:
        """Performs a single validation step using shared evaluation logic.

        Args:
            batch (dict): Batch containing tokenized inputs and labels.
            batch_idx (int): Index of the batch.

        Returns:
            dict: Dictionary containing validation loss and accuracy metrics.

        Notes:
            https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#validation_step
        """
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"val_acc": acc, "val_loss": loss}
        self.log_dict(metrics)
        return metrics

    def test_step(self, batch: dict, batch_idx: int) -> STEP_OUTPUT:
        """Performs a single test step using shared evaluation logic.

        Args:
            batch (dict): Batch containing tokenized inputs and labels.
            batch_idx (int): Index of the batch.

        Returns:
            dict: Dictionary containing test loss and accuracy metrics.

        Notes:
            https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#test_step
        """
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"test_acc": acc, "test_loss": loss}
        self.log_dict(metrics)
        return metrics

    def _shared_eval_step(self, batch: dict, batch_idx: int) -> tuple[Tensor, Tensor]:
        """Shared logic for validation and test steps.

        Args:
            batch (dict): Batch containing tokenized inputs and labels.
            batch_idx (int): Index of the batch.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - loss (torch.Tensor): Evaluation loss.
                - acc (torch.Tensor): Multilabel accuracy score.
        """
        labels = batch["labels"]
        outputs = self(**batch)
        loss, logits = outputs[:2]
        acc = multilabel_accuracy(logits, labels, num_labels=self.num_labels)
        return loss, acc

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Set up the optimizer for training.

        Returns:
            torch.optim.Optimizer: Adam optimizer instance with the specified learning rate.

        Notes:
            https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-optimizers
        """
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
