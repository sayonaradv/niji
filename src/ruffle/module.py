import lightning.pytorch as pl
import torch
from lightning.pytorch.utilities.types import (
    LRScheduler,
    OptimizerLRSchedulerConfig,
)
from pydantic import (
    ConfigDict,
    PositiveFloat,
    PositiveInt,
    validate_call,
)
from torch import Tensor
from torch.nn import functional as F
from torch.optim import Optimizer
from torchmetrics.functional.classification import multilabel_accuracy

from ruffle.config import ModuleConfig
from ruffle.schedulers import LinearWarmupCosineAnnealingLR
from ruffle.types import BATCH, MODEL_OUTPUT
from ruffle.utils import get_model_and_tokenizer


class RuffleModel(pl.LightningModule):
    """PyTorch Lightning module for fine-tuning transformer models on multi-label text classification.

    This module wraps a Hugging Face transformer model (e.g., BERT) for toxic comment
    classification or other multi-label text classification tasks. It provides training,
    validation, and testing steps with metrics, and integrates a custom linear warmup
    cosine annealing learning rate scheduler.
    """

    @validate_call(config=ConfigDict(validate_default=True))
    def __init__(
        self,
        model_name: str,
        num_labels: PositiveInt = ModuleConfig.num_labels,
        label_names: list[str] | None = ModuleConfig.label_names,
        max_token_len: PositiveInt = ModuleConfig.max_token_len,
        lr: PositiveFloat = ModuleConfig.lr,
        warmup_start_lr: PositiveFloat = ModuleConfig.warmup_start_lr,
        warmup_epochs: PositiveInt = ModuleConfig.warmup_epochs,
        cache_dir: str | None = ModuleConfig.cache_dir,
    ) -> None:
        """Initialize the RuffleModel.

        Args:
            model_name: Hugging Face model identifier (e.g., "bert-base-uncased").
            num_labels: Number of output labels for classification.
            label_names: Optional list of label names. Must match ``num_labels`` length if provided.
            max_token_len: Maximum token length for text inputs.
            lr: Initial learning rate for Adam optimizer.
            warmup_start_lr: Starting learning rate for warmup phase.
            warmup_epochs: Number of epochs for learning rate warmup.
            cache_dir: Directory to cache pretrained models and tokenizers.

        Raises:
            ValueError: If ``label_names`` is provided and its length does not equal ``num_labels``.
        """
        super().__init__()

        if label_names is not None and len(label_names) != num_labels:
            raise ValueError(
                f"Size of 'label_names' ({len(label_names)}) does not match 'num_labels' ({num_labels})"
            )

        self.save_hyperparameters()

        self.model, self.tokenizer = get_model_and_tokenizer(
            self.hparams["model_name"],
            cache_dir=self.hparams["cache_dir"],
            num_labels=self.hparams["num_labels"],
        )
        self.model.train()

    def configure_model(self) -> None:
        """Configure the underlying transformer model.

        Calls ``self.model.compile()`` to optimize training speed.
        """
        self.model.compile()

    def forward(
        self, text: str | list[str], labels: torch.Tensor | None = None
    ) -> MODEL_OUTPUT:
        """Forward pass through the model.

        Tokenizes input text, feeds tokens through the transformer model, and optionally
        computes the binary cross-entropy loss for multi-label classification.

        Args:
            text: Input text or list of text strings.
            labels: Optional tensor of shape ``(batch_size, num_labels)`` containing
                binary label indicators.

        Returns:
            If ``labels`` is provided: A tuple of ``(logits, loss)``.
            Otherwise: The raw logits tensor of shape ``(batch_size, num_labels)``.
        """
        inputs = self.tokenizer(
            text,
            max_length=self.hparams["max_token_len"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        logits = outputs.logits

        if labels is not None:
            labels = labels.to(self.model.device)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            return logits, loss
        else:
            return logits

    def training_step(self, batch: BATCH, batch_idx: int) -> MODEL_OUTPUT:
        """Run a single training step.

        Args:
            batch: A batch of training data containing ``"text"`` and ``"labels"``.
            batch_idx: Index of the current batch.

        Returns:
            Training loss tensor.
        """
        _, loss = self(**batch)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: BATCH, batch_idx: int) -> None:
        """Run a single validation step.

        Args:
            batch: A batch of validation data containing ``"text"`` and ``"labels"``.
            batch_idx: Index of the current batch.

        Returns:
            None
        """
        self._shared_eval_step(batch, stage="val")

    def test_step(self, batch: BATCH, batch_idx: int) -> None:
        """Run a single test step.

        Args:
            batch: A batch of test data containing ``"text"`` and ``"labels"``.
            batch_idx: Index of the current batch.

        Returns:
            None
        """
        self._shared_eval_step(batch, stage="test")

    def _shared_eval_step(self, batch: BATCH, stage: str) -> None:
        if "labels" not in batch:
            raise KeyError(
                "Cannot perform validation step: batch is missing `labels` key."
            )
        if not isinstance(batch["labels"], Tensor):
            raise TypeError(
                f"Cannot perform validation step: expected `labels` to be a tensor, got {type(batch['labels'])}"
            )

        logits, loss = self(**batch)
        acc = multilabel_accuracy(
            logits, batch["labels"], num_labels=self.hparams["num_labels"]
        )
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{stage}_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        """Configure the optimizer and learning rate scheduler.

        Returns:
            A dictionary containing:
                - ``optimizer``: Adam optimizer.
                - ``lr_scheduler``: Linear warmup cosine annealing scheduler.
        """
        optimizer: Optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.hparams["lr"]
        )
        scheduler: LRScheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=self.hparams["warmup_epochs"],
            warmup_start_lr=self.hparams["warmup_start_lr"],
            max_epochs=self.trainer.max_epochs or 20,
        )

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
