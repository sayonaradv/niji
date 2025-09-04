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
        num_labels = self.hparams["num_labels"]
        label_names = self.hparams["label_names"]

        if label_names is not None and len(label_names) != num_labels:
            raise ValueError(
                f"Length of label_names ({len(label_names)}) must match num_labels ({num_labels})."
            )

    def configure_model(self) -> None:
        self.model.compile()  # improves training speed

    def forward(self, text: TextInput, labels: Tensor | None = None) -> TensorDict:  # type: ignore[override]
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
        loss: Tensor = self(**batch)["loss"]
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: Batch, batch_idx: int) -> STEP_OUTPUT:  # type: ignore[override]
        self._shared_eval_step(batch, stage="val")
        return None

    def test_step(self, batch: Batch, batch_idx: int) -> STEP_OUTPUT:  # type: ignore[override]
        self._shared_eval_step(batch, stage="test")
        return None

    def _shared_eval_step(self, batch: Batch, stage: str) -> STEP_OUTPUT:
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
