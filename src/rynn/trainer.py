import torch
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from lightning.pytorch.cli import ArgsType, LightningArgumentParser, LightningCLI

from rynn.dataloaders import JigsawDataModule
from rynn.models import Classifier

# See https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
torch.set_float32_matmul_precision("medium")


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.link_arguments("data.labels", "model.label_names")
        parser.link_arguments(
            "data.labels", "model.num_labels", compute_fn=lambda x: len(x)
        )


def cli_main(args: ArgsType = None) -> None:
    MyLightningCLI(
        model_class=Classifier,
        datamodule_class=JigsawDataModule,
        trainer_defaults={
            "max_epochs": 20,
            "deterministic": True,
            "callbacks": [
                EarlyStopping(
                    monitor="val_loss",
                    mode="min",
                    patience=3,
                    verbose=True,
                ),
                ModelCheckpoint(
                    monitor="val_loss",
                    mode="min",
                    filename="{epoch:02d}-{val_loss:.4f}",
                    save_top_k=1,
                    verbose=True,
                ),
                LearningRateMonitor(logging_interval="epoch"),
                RichProgressBar(),
            ],
            "logger": True,
        },
        seed_everything_default=1234,
        args=args,
    )


if __name__ == "__main__":
    cli_main()
