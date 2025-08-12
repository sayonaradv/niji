import lightning.pytorch as pl
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import Tensor
from torch.nn import functional as F
from torchmetrics.functional.classification import multilabel_accuracy
from transformers.modeling_outputs import SequenceClassifierOutput

from stormy.utils import get_model_and_tokenizer


class SequenceClassificationModule(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        *,
        num_labels: int,
        max_token_len: int = 128,
        cache_dir: str = "data",
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model, self.tokenizer = get_model_and_tokenizer(
            self.hparams["model_name"],
            num_labels=self.hparams["num_labels"],
            cache_dir=self.hparams["cache_dir"],
        )
        self.model.train()

    def forward(self, x: str | list[str]) -> SequenceClassifierOutput:
        inputs = self.tokenizer(
            x,
            max_length=self.hparams["max_token_len"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return self.model(**inputs)[0]

    def training_step(self, batch: dict, batch_idx: int) -> Tensor:
        x, y = batch["text"], batch["labels"]
        y_hat = self(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        self._shared_eval_step(batch, stage="val")

    def test_step(self, batch: dict, batch_idx: int) -> None:
        self._shared_eval_step(batch, stage="test")

    def _shared_eval_step(self, batch: dict, stage: str) -> None:
        x, y = batch["text"], batch["labels"]
        y_hat = self(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        acc = multilabel_accuracy(
            y_hat, y, num_labels=self.hparams["num_labels"], threshold=0.5
        )
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{stage}_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = self.optimizer(self.model.parameters())
        scheduler = self.scheduler(optimizer)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


if __name__ == "__main__":
    lit_module = SequenceClassificationModule(
        "gaunernst/bert-tiny-uncased",
        num_labels=6,
    )
    x = ["Fuck you nigga!", "cunt", "bitch"]
    y_hat = lit_module(x)
    print(y_hat)
