import os
import zipfile
from pathlib import Path
from typing import Any

from torch import Tensor
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)


def move_to(obj: Any, device: str) -> Any:
    """Moves tensors or collections of tensors to a specified device.

    Args:
        obj: The object to move (can be a Tensor, dict, or list containing tensors).
        device: The device to move the tensors to (e.g., 'cpu', 'cuda').
    """
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

    msg = "Invalid type"
    raise TypeError(msg)


def create_dirs(dirs: list[str | Path]) -> None:
    """Creates directories if they do not already exist.

    Args:
        dirs: List of directory paths to create.
    """
    for d in dirs:
        path = Path(d) if isinstance(d, str) else d
        if not path.is_dir():
            path.mkdir()


def get_model_and_tokenizer(
    model_name: str,
    num_labels: int,
    cache_dir: str | None,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Loads a model and tokenizer for sequence classification.

    Args:
        model_name: Name or path of the pretrained model.
        num_labels: Number of labels for classification.
        cache_dir: Directory to cache the model and tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        cache_dir=cache_dir,
        problem_type="multi_label_classification",
        return_dict=True,
    )
    return model, tokenizer


def unzip_directory(
    directory: str | Path,
    *,
    extract_to: str | Path | None = None,
    delete_existing: bool = False,
) -> None:
    """Extracts all .zip files in a directory.

    Args:
        directory: Directory containing .zip files.
        extract_to: Directory to extract files to. Defaults to the input directory.
        delete_existing: Whether to delete zip files after extraction.
    """
    extract_to = extract_to or directory
    zip_files = [f for f in Path(directory).iterdir() if f.suffix == ".zip"]
    if len(zip_files) == 0:
        return
    for file_path in zip_files:
        with zipfile.ZipFile(file_path, "r") as z:
            z.extractall(extract_to)
        if delete_existing:
            file_path.unlink()


def resolve_num_workers(num_workers):
    """Resolves the number of workers for DataLoader.

    Args:
        num_workers: 'auto', None, or an integer. If 'auto' or None, uses all available CPU cores.

    Returns:
        int: The number of workers to use.
    """
    if num_workers in (None, "auto"):
        return os.cpu_count()
    return int(num_workers)
