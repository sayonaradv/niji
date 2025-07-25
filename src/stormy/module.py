import lightning.pytorch as pl
import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import Tensor
from torchmetrics.functional.classification import multilabel_accuracy
from transformers import AutoModelForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput


class SequenceClassificationModule(pl.LightningModule):
    """A PyTorch Lightning module for multi-label text classification using transformer models.

    This module wraps a HuggingFace transformer model for multi-label sequence classification
    tasks and provides training, validation, and testing steps with appropriate metrics logging.
    """

    def __init__(
        self,
        model_name: str,
        num_labels: int,
        learning_rate: float = 3e-5,
    ) -> None:
        """A custom LightningModule for multi-label sequence classification.

        Args:
            model_name: the name of the model and accompanying tokenizer
            num_labels: the number of target labels in the training dataset.
                        This value determines the size of the output layer of the model.
            learning_rate: learning rate passed to optimizer
        """
        super().__init__()
        self.save_hyperparameters()

        self.model_name_or_path = model_name
        self.num_labels = num_labels
        self.learning_rate = learning_rate

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            problem_type="multi_label_classification",
        )
        self.model.train()

    def forward(self, **inputs) -> SequenceClassifierOutput:
        """Forward pass through the transformer model.

        Args:
            **inputs: Keyword arguments containing model inputs (input_ids, attention_mask, etc.).

        Returns:
            SequenceClassifierOutput containing loss, logits, and other model outputs.
        """
        return self.model(**inputs)

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        """
        Notes:
            https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#training
        """
        outputs = self(**batch)
        self.log("train_loss", outputs.loss)
        return outputs.loss

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        """
        Notes:
            https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#validation
        """
        self._shared_eval_step(batch, stage="val")

    def test_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        """
        Notes:
            https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#testing
        """
        self._shared_eval_step(batch, stage="test")

    def _shared_eval_step(self, batch: dict[str, Tensor], stage: str) -> None:
        """Compute and log loss and accuracy metrics for evaluation steps."""
        outputs = self(**batch)
        loss, logits = outputs.loss, outputs.logits
        acc = multilabel_accuracy(logits, batch["labels"], num_labels=self.num_labels)
        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_acc", acc, prog_bar=True)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """
        Notes:
            https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.configure_optimizers
        """
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
