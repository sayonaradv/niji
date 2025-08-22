import torch
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from lightning.pytorch.cli import ArgsType, LightningArgumentParser, LightningCLI

from blanket.dataloaders import JigsawDataModule
from blanket.models import ToxicityClassifier

# See https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
torch.set_float32_matmul_precision("medium")


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.link_arguments("trainer.max_epochs", "model.num_training_steps")


def main(args: ArgsType = None) -> None:
    MyLightningCLI(
        model_class=ToxicityClassifier,
        datamodule_class=JigsawDataModule,
        trainer_defaults={
            "max_epochs": 20,
            "deterministic": True,
            "callbacks": [
                EarlyStopping(monitor="val_loss", mode="min", patience=3, verbose=True),
                ModelCheckpoint(
                    monitor="val_loss",
                    mode="min",
                    filename="{epoch:02d}-{val_loss:.4f}",
                    save_top_k=1,
                    verbose=True,
                ),
                LearningRateMonitor(logging_interval="epoch"),
                RichModelSummary(max_depth=-1),
                RichProgressBar(),
            ],
            "logger": True,
        },
        seed_everything_default=1234,
        args=args,
    )


if __name__ == "__main__":
    main()
