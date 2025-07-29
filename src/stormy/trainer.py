import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from lightning.pytorch.cli import ArgsType, LightningCLI

from stormy.datamodule import AutoTokenizerDataModule
from stormy.module import SequenceClassificationModule

# see https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
torch.set_float32_matmul_precision("medium")


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments(
            "data.label_columns",
            "model.num_labels",
            compute_fn=lambda label_columns: len(label_columns),
        )
        parser.link_arguments("model.model_name", "data.model_name")


def cli_main(args: ArgsType = None):
    MyLightningCLI(
        model_class=SequenceClassificationModule,
        datamodule_class=AutoTokenizerDataModule,
        trainer_class=pl.Trainer,
        seed_everything_default=1234,
        args=args,
        trainer_defaults={
            "max_epochs": 10,
            "deterministic": True,
            "callbacks": [
                EarlyStopping(monitor="val_loss", mode="min"),
                ModelCheckpoint(
                    monitor="val_loss",
                    mode="min",
                    filename="{epoch:02d}-{val_loss:.4f}",
                ),
                RichModelSummary(max_depth=-1),
                RichProgressBar(),
            ],
            "logger": True,
            "precision": "16-mixed",
        },
    )


if __name__ == "__main__":
    cli_main()
