from pathlib import Path
from typing import Any

from torch import Tensor
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerFast,
)


def get_model_and_tokenizer(
    model_name: str,
    num_labels: int,
    cache_dir: str | None,
) -> tuple[PreTrainedModel, PreTrainedTokenizerFast]:
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        cache_dir=cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, cache_dir=cache_dir, use_fast=True
    )
    return model, tokenizer


def move_to(obj: Any, device: str) -> Any:
    if isinstance(obj, Tensor):
        return obj.to(device=device)

    if isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = move_to(v, device)
        return res

    if isinstance(obj, list):
        res = []
        for v in obj:
            res.append(move_to(v, device))
        return res

    msg = f"Invalid type for device movement: {type(obj)}. Supported types: Tensor, dict, list"
    raise TypeError(msg)


def create_dirs(dirs: list[str | Path]) -> None:
    for d in dirs:
        path = Path(d) if isinstance(d, str) else d

        if not path.is_dir():
            path.mkdir(parents=True, exist_ok=True)
