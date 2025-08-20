import torch
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from lightning.pytorch.cli import ArgsType, LightningArgumentParser, LightningCLI

from blanket.datamodule import HFDataModule
from blanket.module import SequenceClassificationModule
from blanket.schedulers import LinearWarmupCosineAnnealingLR

# See https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
torch.set_float32_matmul_precision("medium")


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_optimizer_args(torch.optim.Adam)
        parser.set_defaults({"optimizer.lr": 3e-5})

        parser.add_lr_scheduler_args(LinearWarmupCosineAnnealingLR)
        parser.set_defaults(
            {"lr_scheduler.warmup_epochs": 5, "lr_scheduler.warmup_start_lr": 1.0 / 10}
        )

        parser.link_arguments("trainer.max_epochs", "lr_scheduler.max_epochs")
        parser.link_arguments(
            "data.label_columns",
            "model.num_labels",
            compute_fn=lambda label_columns: len(label_columns),
        )


def main(args: ArgsType = None) -> None:
    MyLightningCLI(
        model_class=SequenceClassificationModule,
        datamodule_class=HFDataModule,
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
