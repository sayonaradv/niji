from time import perf_counter

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from lightning.pytorch.loggers import TensorBoardLogger
from pydantic import (
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    validate_call,
)

from ruffle.config import Config, DataConfig, ModuleConfig, TrainerConfig
from ruffle.dataloader import JigsawDataModule
from ruffle.module import RuffleModel
from ruffle.utils import log_perf

# See https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
torch.set_float32_matmul_precision("medium")


@validate_call(config=ConfigDict(validate_default=True))
def train(
    model_name: str,
    data_dir: str = DataConfig.data_dir,
    labels: list[str] | None = None,
    batch_size: PositiveInt = DataConfig.batch_size,
    val_size: float = Field(default=0.2, ge=0, le=1),  # pyrefly: ignore
    max_token_len: PositiveInt = ModuleConfig.max_token_len,
    lr: PositiveFloat = ModuleConfig.lr,
    warmup_start_lr: PositiveFloat = ModuleConfig.warmup_start_lr,
    warmup_epochs: PositiveInt = ModuleConfig.warmup_epochs,
    max_epochs: PositiveInt = TrainerConfig.max_epochs,
    patience: PositiveInt = TrainerConfig.patience,
    run_name: str | None = None,
    perf: bool = False,
    fast_dev_run: bool = False,
    cache_dir: str = Config.cache_dir,
    log_dir: str = Config.log_dir,
    seed: NonNegativeInt = Config.seed,
) -> None:
    pl.seed_everything(seed, workers=True)

    datamodule = JigsawDataModule(data_dir, batch_size, val_size, labels)

    # If no labels provided, datamodule will automatically use JIGSAW_LABELS
    labels = datamodule.labels
    num_labels = len(labels)

    model = RuffleModel(
        model_name,
        num_labels,
        labels,
        max_token_len,
        lr,
        warmup_start_lr,
        warmup_epochs,
        cache_dir,
    )

    logger = TensorBoardLogger(save_dir=log_dir, name="training_runs", version=run_name)

    callbacks: list[Callback] = [
        ModelCheckpoint(filename="{epoch:02d}-{val_loss:.4f}"),
        LearningRateMonitor(logging_interval="epoch"),
        RichProgressBar(),
    ]

    # do not use EarlyStopping if getting perf benchmark
    if not perf:
        callbacks.append(
            EarlyStopping(monitor="val_loss", patience=patience),
        )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        logger=logger,
        callbacks=callbacks,
        fast_dev_run=fast_dev_run,
        deterministic=True,
    )

    start = perf_counter()
    trainer.fit(model, datamodule=datamodule)
    stop = perf_counter()

    if perf:
        log_perf(start, stop, trainer)
