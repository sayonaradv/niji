"""
LightningModule for multilabel toxic comment classification using HuggingFace Transformers.

Implements a PyTorch LightningModule for toxic comment classification with a HuggingFace model.
"""

from typing import Any

import lightning.pytorch as pl
import torch
import torchmetrics
from lightning.pytorch.utilities.types import STEP_OUTPUT
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
)


class StormyTransformer(pl.LightningModule):
    """A custom LightningModule for multilabel toxic comment classification.

    Args:
        model_name_or_path (str): Name or path of the pretrained model.
        num_labels (int): Number of output classes.
        learning_rate (float): Learning rate for the optimizer.
        cache_dir (str): Directory to cache the model and tokenizer.
    """

    def __init__(
        self,
        model_name_or_path: str = "distilbert-base-cased",
        num_labels: int = 6,
        learning_rate: float = 2e-5,
        cache_dir: str = "data",
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.config = AutoConfig.from_pretrained(
            model_name_or_path,
            num_labels=num_labels,
            problem_type="multi_label_classification",
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, config=self.config
        )
        self.accuracy = torchmetrics.Accuracy(task="multilabel", num_labels=num_labels)

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
        """Performs a single validation step.

        Args:
            batch (dict): Batch containing tokenized inputs and labels.
            batch_idx (int): Index of the batch.

        Returns:
            None

        Notes:
            https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#validation_step
        """
        outputs = self(**batch)
        val_loss, logits = outputs[:2]
        self.log("val_loss", val_loss, prog_bar=True)

        if self.hparams.num_labels > 1:
            preds = torch.sigmoid(logits)
        elif self.hparams.num_labels == 1:
            preds = logits.squeeze()

        labels = batch["labels"]
        self.accuracy(preds, labels)
        self.log("val_acc", self.accuracy, on_epoch=True, prog_bar=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Set up the optimizer for training.

        Returns:
            torch.optim.Optimizer: Adam optimizer instance with the specified learning rate.

        Notes:
            https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-optimizers
        """
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
