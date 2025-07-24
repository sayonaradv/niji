import os
from pathlib import Path
from typing import Any

import numpy as np
from torch import Tensor


def combine_labels(batch: dict[str, list[Any]], label_columns: list[str]) -> np.ndarray:
    """
    Combine multiple binary label columns into a single Numpy array.

    Args:
        batch: Dictionary containing batched examples where each key maps to a list of values
        label_columns: List of column names to combine into the labels tensor

    Returns:
        Combined Numpy array of shape (batch_size, num_labels)

    Raises:
        KeyError: If any of the specified label columns are not found in the dataset
    """
    missing_columns = [col for col in label_columns if col not in batch]
    if missing_columns:
        raise KeyError(f"Label columns not found in dataset: {missing_columns}")

    return np.stack([batch[col] for col in label_columns if col in batch], axis=1)


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


def get_num_workers() -> int:
    num_workers = os.cpu_count()
    if num_workers is not None:
        return num_workers
    else:
        return 0
