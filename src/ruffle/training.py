from time import perf_counter

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import (
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

CACHE_DIR: str = Config.cache_dir


@validate_call(config=ConfigDict(validate_default=True))
def train(
    model_name: str,
    data_dir: str = DataConfig.data_dir,
    labels: list[str] | None = None,
    batch_size: PositiveInt = DataConfig.batch_size,
    val_size: float = Field(default=DataConfig.val_size, gt=0, lt=1),
    max_token_len: PositiveInt = ModuleConfig.max_token_len,
    lr: PositiveFloat = ModuleConfig.lr,
    warmup_start_lr: PositiveFloat = ModuleConfig.warmup_start_lr,
    warmup_epochs: PositiveInt = ModuleConfig.warmup_epochs,
    max_epochs: PositiveInt = TrainerConfig.max_epochs,
    patience: PositiveInt = TrainerConfig.patience,
    run_name: str | None = None,
    perf: bool = False,
    fast_dev_run: bool = False,
    seed: NonNegativeInt = Config.seed,
) -> None:
    pl.seed_everything(seed, workers=True)

    datamodule = JigsawDataModule(
        data_dir=data_dir,
        batch_size=batch_size,
        val_size=val_size,
        labels=labels,
    )

    model = RuffleModel(
        model_name=model_name,
        num_labels=len(datamodule.labels),
        label_names=datamodule.labels,
        max_token_len=max_token_len,
        lr=lr,
        warmup_start_lr=warmup_start_lr,
        warmup_epochs=warmup_epochs,
        cache_dir=CACHE_DIR,
    )

    logger = TensorBoardLogger(save_dir="runs", name="training_runs", version=run_name)

    callbacks = [
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
