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

from ruffle.schedulers import LinearWarmupCosineAnnealingLR
from ruffle.types import BATCH, MODEL_OUTPUT
from ruffle.utils import get_model_and_tokenizer


class RuffleModel(pl.LightningModule):
    """PyTorch Lightning module for fine-tuning transformer models on multi-label classification.

    This module wraps a Hugging Face transformer model for toxic comment classification
    or other multi-label text classification tasks. It provides training, validation,
    and testing steps with integrated metrics and a custom linear warmup cosine annealing
    learning rate scheduler.

    Example:
        >>> model = RuffleModel(
        ...     model_name="bert-base-uncased",
        ...     label_names=["toxic", "severe_toxic", "obscene"]
        ... )
        >>> trainer = pl.Trainer(max_epochs=10)
        >>> trainer.fit(model, train_dataloader, val_dataloader)
    """

    @validate_call(config=ConfigDict(validate_default=True))
    def __init__(
        self,
        model_name: str,
        num_labels: PositiveInt = 6,
        label_names: list[str] | None = None,
        max_token_len: PositiveInt = 256,
        lr: PositiveFloat = 3e-5,
        warmup_start_lr: PositiveFloat = 1e-5,
        warmup_epochs: PositiveInt = 5,
        cache_dir: str | None = "./data",
    ) -> None:
        """Initialize the RuffleModel.

        Args:
            model_name (str): Hugging Face model identifier (e.g., "bert-base-uncased").
            num_labels (PositiveInt): Number of output labels for multi-label classification.
            label_names (list[str] | None): Optional list of label names for logging and
                visualization. When provided, `num_labels` will be overriden with `len(label_names)`.
            max_token_len (PositiveInt): Maximum sequence length for tokenized inputs.
                Longer sequences will be truncated.
            lr (PositiveFloat): Peak learning rate for the Adam optimizer.
            warmup_start_lr (PositiveFloat): Starting learning rate for the warmup phase.
                Should be smaller than `lr`.
            warmup_epochs (PositiveInt): Number of epochs for learning rate warmup before
                reaching peak `lr`.
            cache_dir (str | None): Directory to cache pretrained models and tokenizers.
                If None, uses the default transformers cache directory at
                `~/.cache/transformers`.
        """
        super().__init__()

        num_labels = len(label_names) if label_names is not None else num_labels

        self.save_hyperparameters()

        self.model, self.tokenizer = get_model_and_tokenizer(
            self.hparams["model_name"],
            cache_dir=self.hparams["cache_dir"],
            num_labels=self.hparams["num_labels"],
        )
        self.model.train()

    def configure_model(self) -> None:
        """Configure the underlying transformer model.

        Calls `torch.compile()` on the model to optimize training speed through
        graph compilation and kernel fusion.
        """
        self.model.compile()

    def forward(
        self, text: str | list[str], labels: torch.Tensor | None = None
    ) -> MODEL_OUTPUT:
        """Forward pass through the model.

        Tokenizes input text, processes it through the transformer model, and optionally
        computes binary cross-entropy loss for multi-label classification.

        Args:
            text (str | list[str]): Input text string or batch of text strings to classify.
            labels (torch.Tensor | None): Optional ground truth labels tensor of shape
                `(batch_size, num_labels)` with binary indicators (0 or 1).

        Returns:
            MODEL_OUTPUT: If `labels` is provided, returns a tuple of `(logits, loss)`.
                If `labels` is None, returns only the logits tensor.

                - logits: Float tensor of shape `(batch_size, num_labels)` containing
                  raw model outputs before sigmoid activation.
                - loss: Scalar tensor containing binary cross-entropy loss (only when
                  labels are provided).
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
        """Execute a single training step.

        Args:
            batch (BATCH): Training batch containing "text" and "labels" keys.
            batch_idx (int): Index of the current batch within the epoch.

        Returns:
            MODEL_OUTPUT: Training loss tensor for backpropagation.
        """
        _, loss = self(**batch)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: BATCH, batch_idx: int) -> None:
        """Execute a single validation step.

        Computes validation loss and multi-label accuracy metrics.

        Args:
            batch (BATCH): Validation batch containing "text" and "labels" keys.
            batch_idx (int): Index of the current batch within the validation epoch.
        """
        self._shared_eval_step(batch, stage="val")

    def test_step(self, batch: BATCH, batch_idx: int) -> None:
        """Execute a single test step.

        Computes test loss and multi-label accuracy metrics.

        Args:
            batch (BATCH): Test batch containing "text" and "labels" keys.
            batch_idx (int): Index of the current batch within the test epoch.
        """
        self._shared_eval_step(batch, stage="test")

    def _shared_eval_step(self, batch: BATCH, stage: str) -> None:
        """Shared evaluation logic for validation and test steps.

        Args:
            batch (BATCH): Evaluation batch containing "text" and "labels" keys.
            stage (str): Stage identifier ("val" or "test") for logging metrics.

        Raises:
            KeyError: If batch is missing the "labels" key.
            TypeError: If batch["labels"] is not a torch.Tensor.
        """
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

        Sets up an Adam optimizer with linear warmup followed by cosine annealing
        learning rate decay.

        Returns:
            OptimizerLRSchedulerConfig: Dictionary containing the optimizer and
                lr_scheduler configuration for PyTorch Lightning.
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
