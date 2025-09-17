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
    @validate_call(config=ConfigDict(validate_default=True))
    def __init__(
        self,
        model_name: str,
        num_labels: PositiveInt = 6,
        label_names: list[str] | None = None,
        max_token_len: PositiveInt = 256,
        lr: PositiveFloat = 3e-5,
        warmup_start_lr: PositiveFloat = 1e-5,
        warmup_epochs: PositiveInt = 20,
        cache_dir: str | None = "./data",
    ) -> None:
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
        self.model.compile()  # improves training speed

    def forward(self, *args, **kwargs) -> MODEL_OUTPUT:  # type: ignore[override]
        if len(args) >= 1:
            text = args[0]
            labels = args[1] if len(args) >= 2 else kwargs.get("labels")
        else:
            text = kwargs.get("text")
            labels = kwargs.get("labels")

        if text is None:
            raise ValueError("text argument is required")
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
        else:
            loss = None
        return logits, loss

    def training_step(self, batch: BATCH, batch_idx: int) -> Tensor:  # type: ignore[override]
        _, loss = self(**batch)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: BATCH, batch_idx: int) -> None:  # type: ignore[override]
        self._shared_eval_step(batch, stage="val")
        return None

    def test_step(self, batch: BATCH, batch_idx: int) -> None:  # type: ignore[override]
        self._shared_eval_step(batch, stage="test")
        return None

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
