import torch
from jsonargparse import lazy_instance
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar
from lightning.pytorch.cli import ArgsType, LightningCLI
from lightning.pytorch.loggers import MLFlowLogger

from stormy.datamodule import AutoTokenizerDataModule
from stormy.module import SequenceClassificationModule

# see https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
torch.set_float32_matmul_precision("medium")


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_lightning_class_args(EarlyStopping, "early_stopping")
        parser.set_defaults(
            {
                "early_stopping.monitor": "val_loss",
                "early_stopping.patience": 3,
                "early_stopping.mode": "min",
                "early_stopping.verbose": True,
            }
        )

        parser.add_lightning_class_args(ModelCheckpoint, "checkpoint")
        parser.set_defaults(
            {
                "checkpoint.monitor": "val_loss",
                "checkpoint.filename": "jigsaw-{epoch:02d}-{val_loss:.4f}",
                "checkpoint.verbose": True,
                "checkpoint.save_top_k": 3,
            }
        )

        parser.link_arguments(
            "data.label_columns",
            "model.num_labels",
            compute_fn=lambda label_columns: len(label_columns),
        )
        parser.link_arguments("model.model_name", "data.model_name")


def cli_main(args: ArgsType = None):
    MyLightningCLI(
        SequenceClassificationModule,
        AutoTokenizerDataModule,
        args=args,
        seed_everything_default=1234,
        trainer_defaults={
            "max_epochs": 5,
            "deterministic": True,
            "logger": lazy_instance(
                MLFlowLogger,
                save_dir="./mlflow",
                experiment_name="lightning_logs",
                log_model="all",
            ),
            "callbacks": [lazy_instance(RichProgressBar)],
        },
    )


if __name__ == "__main__":
    cli_main()
