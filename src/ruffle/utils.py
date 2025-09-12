import json
import os
from typing import Any

import torch
from lightning.pytorch import Trainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return torch.cuda.get_device_name()
    else:
        return "cpu"


def log_perf(
    start: float,
    stop: float,
    trainer: Trainer,
) -> None:
    # sync to last checkpoint
    perf = {
        "perf": {
            "device": get_device(),
            "num_node": trainer.num_nodes,
            "num_devices": trainer.num_devices,
            "strategy": trainer.strategy.__class__.__name__,
            "precision": trainer.precision,
            "global_step": trainer.global_step,
            "max_epochs": trainer.max_epochs,
            "batch_size": trainer.datamodule.batch_size,  # type: ignore[override]
            "runtime_min": (stop - start) / 60,
        }
    }

    with open(os.path.join(trainer.log_dir, "perf.json"), "w") as perf_file:
        json.dump(perf, perf_file, indent=4)


def get_model_and_tokenizer(
    model_name: str,
    num_labels: int,
    cache_dir: str | None,
) -> tuple[Any, Any]:
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        cache_dir=cache_dir,
        problem_type="multi_label_classification",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, cache_dir=cache_dir, use_fast=True
    )
    return model, tokenizer
