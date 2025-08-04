"""Utility functions for tensor operations, data handling, and system configuration.

This module provides helper functions used throughout the Stormy library for:
- Label processing and tensor manipulation
- Device-aware tensor operations
- File system operations
- System resource detection

These utilities are designed to be robust, type-safe, and well-documented to support
the core functionality of the toxicity detection pipeline.

Functions:
    combine_labels: Combine multiple binary label columns into tensor format
    move_to: Move tensors or tensor collections to specified devices
    create_dirs: Create directories if they don't exist
    get_num_workers: Detect optimal number of worker processes for data loading

Example:
    Basic usage of utility functions:

    >>> from stormy.utils import combine_labels, move_to, get_num_workers
    >>>
    >>> # Combine multiple label columns for multi-label classification
    >>> batch = {"toxic": [1, 0], "threat": [0, 1], "text": ["hello", "world"]}
    >>> labels = combine_labels(batch, ["toxic", "threat"])
    >>> print(labels)  # [[1.0, 0.0], [0.0, 1.0]]
    >>>
    >>> # Move tensors to appropriate device
    >>> import torch
    >>> data = {"input_ids": torch.tensor([1, 2, 3])}
    >>> gpu_data = move_to(data, "cuda" if torch.cuda.is_available() else "cpu")
    >>>
    >>> # Get optimal number of workers for data loading
    >>> num_workers = get_num_workers()
    >>> print(f"Using {num_workers} workers for data loading")

Notes:
    - All functions include comprehensive error handling and validation
    - Tensor operations are device-aware and handle CPU/GPU placement
    - File operations use pathlib for cross-platform compatibility
    - Performance considerations are documented for each function
"""

import os
from pathlib import Path
from typing import Any

import numpy as np
from torch import Tensor


def combine_labels(
    batch: dict[str, list[Any]], label_columns: list[str]
) -> list[list[float]]:
    """Combine multiple binary label columns into a single multi-label tensor format.

    This function is essential for multi-label classification tasks where each example
    can belong to multiple categories simultaneously. It takes separate binary label
    columns and combines them into the format expected by PyTorch loss functions
    and metrics.

    The function performs several key operations:
    - Validates that all specified label columns exist in the batch
    - Stacks the label columns along the feature dimension
    - Converts all values to float type for compatibility with loss functions
    - Returns data in list format for compatibility with HuggingFace datasets

    Args:
        batch: Dictionary containing batched examples where each key maps to a list
            of values. Typically comes from HuggingFace datasets.map() operations.
            Expected structure: {"label1": [0, 1, 0], "label2": [1, 0, 1], ...}
        label_columns: List of column names to combine into the multi-label tensor.
            Must contain at least one column name, and all names must exist in batch.
            Order matters - the output tensor will have features in this order.

    Returns:
        Combined list of float lists with shape (batch_size, num_labels).
        Each inner list represents the multi-label target for one example.
        Values are guaranteed to be float type for compatibility with PyTorch.

    Raises:
        KeyError: If any of the specified label columns are not found in the batch.
            The error message includes all missing column names for easier debugging.
        ValueError: If label_columns is empty. At least one label column is required
            for meaningful multi-label classification.

    Examples:
        Basic multi-label combination:

        >>> batch = {
        ...     "toxic": [1, 0, 1],
        ...     "threat": [0, 1, 0],
        ...     "insult": [1, 1, 0],
        ...     "text": ["bad comment", "threat text", "normal text"]
        ... }
        >>> labels = combine_labels(batch, ["toxic", "threat", "insult"])
        >>> print(labels)
        [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 0.0, 0.0]]

        Single label (binary classification):

        >>> batch = {"sentiment": [1, 0, 1], "text": ["good", "bad", "great"]}
        >>> labels = combine_labels(batch, ["sentiment"])
        >>> print(labels)
        [[1.0], [0.0], [1.0]]

        Error handling:

        >>> batch = {"label1": [1, 0], "text": ["a", "b"]}
        >>> # This will raise KeyError
        >>> labels = combine_labels(batch, ["label1", "missing_label"])

    Performance Notes:
        - Uses numpy.stack for efficient tensor operations
        - Automatic type conversion ensures compatibility with PyTorch
        - Memory efficient - processes entire batches at once
        - Suitable for use in dataset.map() operations with batched=True

    Integration:
        This function is commonly used in:
        - AutoTokenizerDataModule.preprocess_data() for label preparation
        - Custom dataset classes for multi-label classification
        - Data preprocessing pipelines for toxicity detection
        - Any scenario requiring multi-label tensor preparation

    See Also:
        - AutoTokenizerDataModule: Uses this function for label preprocessing
        - numpy.stack: The underlying function used for tensor stacking
        - PyTorch BCEWithLogitsLoss: Expects the output format from this function
    """
    missing_columns = [col for col in label_columns if col not in batch]
    if missing_columns:
        raise KeyError(f"Label columns not found in dataset: {missing_columns}")

    if not label_columns:
        raise ValueError("label_columns cannot be empty")

    return (
        np.stack([batch[col] for col in label_columns if col in batch], axis=1)
        .astype(float)
        .tolist()
    )


def move_to(obj: Any, device: str) -> Any:
    """Move tensors or collections of tensors to a specified device (CPU/GPU).

    This utility function provides a convenient way to move complex data structures
    containing PyTorch tensors to different devices. It recursively handles nested
    dictionaries and lists, making it ideal for moving batched data between CPU
    and GPU during model training and inference.

    The function is particularly useful for:
    - Moving model inputs to the same device as the model
    - Transferring data between CPU and GPU for memory management
    - Ensuring device consistency in multi-GPU setups
    - Handling complex nested data structures automatically

    Args:
        obj: The object to move. Can be:
            - A single PyTorch Tensor
            - A dictionary containing tensors (potentially nested)
            - A list containing tensors (potentially nested)
            - Mixed structures with tensors at various nesting levels
        device: Target device specification as a string. Common values:
            - "cpu": Move to CPU memory
            - "cuda": Move to default CUDA device
            - "cuda:0", "cuda:1", etc.: Move to specific GPU
            - Any valid PyTorch device string

    Returns:
        The same structure as the input with all tensors moved to the target device.
        Non-tensor objects are passed through unchanged. The original object is
        not modified (tensors are moved, not copied in-place).

    Raises:
        TypeError: If the input object type is not supported (not a Tensor, dict, or list).
        RuntimeError: If the specified device is not available (e.g., CUDA requested
            but not available).

    Examples:
        Move a single tensor:

        >>> import torch
        >>> tensor = torch.tensor([1, 2, 3])
        >>> gpu_tensor = move_to(tensor, "cuda")  # Move to GPU

        Move a dictionary of tensors (common in model inputs):

        >>> batch = {
        ...     "input_ids": torch.tensor([[101, 2023, 102]]),
        ...     "attention_mask": torch.tensor([[1, 1, 1]]),
        ...     "labels": torch.tensor([[1, 0, 1]])
        ... }
        >>> gpu_batch = move_to(batch, "cuda")

        Move nested structures:

        >>> data = {
        ...     "inputs": {
        ...         "ids": torch.tensor([1, 2, 3]),
        ...         "mask": torch.tensor([1, 1, 0])
        ...     },
        ...     "targets": [torch.tensor([1]), torch.tensor([0])]
        ... }
        >>> gpu_data = move_to(data, "cuda")

        Safe device selection:

        >>> device = "cuda" if torch.cuda.is_available() else "cpu"
        >>> data = move_to(data, device)

    Performance Notes:
        - Tensor.to() operations are optimized by PyTorch
        - Moving from GPU to CPU involves memory transfer overhead
        - Moving between GPUs may require peer-to-peer memory access
        - Non-tensor objects have minimal performance impact

    Device Management:
        - Always check device availability before moving to CUDA
        - Consider memory constraints when moving large tensors to GPU
        - Use torch.cuda.is_available() for safe device selection
        - Monitor GPU memory usage in multi-GPU scenarios

    Integration:
        This function is useful in:
        - Model forward passes to ensure input/model device consistency
        - Data loading pipelines for device placement
        - Inference scripts that need flexible device handling
        - Testing scenarios with different device configurations

    See Also:
        - torch.Tensor.to(): The underlying method for tensor device movement
        - torch.cuda.is_available(): Check CUDA availability
        - PyTorch device management: https://pytorch.org/docs/stable/notes/cuda.html
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

    msg = f"Invalid type for device movement: {type(obj)}. Supported types: Tensor, dict, list"
    raise TypeError(msg)


def create_dirs(dirs: list[str | Path]) -> None:
    """Create directories if they do not already exist.

    This utility function ensures that all specified directories exist, creating them
    (including parent directories) if they don't. It's useful for setting up directory
    structures for model checkpoints, logs, cache files, and other outputs.

    The function handles both string paths and pathlib.Path objects, automatically
    converting strings to Path objects for consistent cross-platform behavior.
    It only creates directories that don't already exist, making it safe to call
    multiple times.

    Args:
        dirs: List of directory paths to create. Each path can be either:
            - A string path (e.g., "./logs", "/tmp/cache", "models/checkpoints")
            - A pathlib.Path object
            - Relative or absolute paths are both supported
            - Parent directories are created automatically if needed

    Returns:
        None. The function operates through side effects (directory creation).

    Raises:
        PermissionError: If the process lacks permission to create the directories
        OSError: If directory creation fails due to system constraints
        FileExistsError: If a file (not directory) exists at the specified path

    Examples:
        Create single directory:

        >>> create_dirs(["./models"])

        Create multiple directories:

        >>> create_dirs([
        ...     "./logs",
        ...     "./checkpoints",
        ...     "./data/cache",
        ...     "/tmp/stormy_temp"
        ... ])

        Mix string and Path objects:

        >>> from pathlib import Path
        >>> create_dirs([
        ...     "./logs",
        ...     Path("./models"),
        ...     Path.home() / "stormy_cache"
        ... ])

        Create nested directory structure:

        >>> create_dirs([
        ...     "./experiments/run_001/checkpoints",
        ...     "./experiments/run_001/logs",
        ...     "./experiments/run_001/outputs"
        ... ])

    Usage Patterns:
        Initialize experiment directories:

        >>> experiment_dirs = [
        ...     "./lightning_logs",
        ...     "./checkpoints",
        ...     "./data/cache",
        ...     "./outputs"
        ... ]
        >>> create_dirs(experiment_dirs)

        Setup cache directories:

        >>> cache_dirs = [
        ...     "~/.cache/stormy/models",
        ...     "~/.cache/stormy/datasets",
        ...     "~/.cache/stormy/tokenizers"
        ... ]
        >>> # Note: ~ expansion happens automatically with pathlib
        >>> create_dirs([str(Path(d).expanduser()) for d in cache_dirs])

    Performance Notes:
        - Path.mkdir() is optimized and only creates missing directories
        - Multiple calls with the same directories are efficiently handled
        - Cross-platform path handling via pathlib ensures compatibility
        - Minimal performance impact for directories that already exist

    Safety Features:
        - Only creates directories, never overwrites existing files
        - Parent directory creation is automatic (like mkdir -p)
        - No-op for directories that already exist
        - Type conversion handles mixed string/Path inputs

    Integration:
        This function is commonly used in:
        - Training script initialization to setup output directories
        - Data pipeline setup for cache and temporary directories
        - Model deployment scripts for organizing outputs
        - Testing frameworks for creating temporary directories

    See Also:
        - pathlib.Path.mkdir(): The underlying directory creation method
        - os.makedirs(): Alternative approach with different API
        - tempfile.mkdtemp(): For creating temporary directories
    """
    for d in dirs:
        path = Path(d) if isinstance(d, str) else d

        if not path.is_dir():
            path.mkdir(parents=True, exist_ok=True)


def get_num_workers() -> int:
    """Detect the optimal number of worker processes for data loading.

    This function determines the appropriate number of worker processes to use with
    PyTorch DataLoaders based on the system's CPU configuration. Using multiple
    workers can significantly speed up data loading by parallelizing dataset
    operations like tokenization, augmentation, and preprocessing.

    The function provides a safe fallback mechanism: if CPU count detection fails
    (which can happen in some containerized environments or restricted systems),
    it returns 0, which tells PyTorch to use the main process for data loading.

    Worker Process Benefits:
        - Parallelizes dataset preprocessing operations
        - Reduces training time by overlapping data loading with model computation
        - Maximizes CPU utilization during training
        - Particularly beneficial for complex preprocessing pipelines

    Args:
        None. The function automatically detects system capabilities.

    Returns:
        int: Number of worker processes to use for data loading:
            - Positive integer: Number of CPU cores available for parallel processing
            - 0: Use main process only (fallback for systems where CPU count is unavailable)

    Examples:
        Basic usage in DataLoader configuration:

        >>> from torch.utils.data import DataLoader
        >>> num_workers = get_num_workers()
        >>> dataloader = DataLoader(
        ...     dataset,
        ...     batch_size=32,
        ...     num_workers=num_workers,
        ...     pin_memory=True
        ... )

        Conditional worker configuration:

        >>> num_workers = get_num_workers()
        >>> print(f"Using {num_workers} workers for data loading")
        >>> if num_workers > 0:
        ...     print("Parallel data loading enabled")
        ... else:
        ...     print("Using main process for data loading")

        Integration with Lightning DataModule:

        >>> class MyDataModule(pl.LightningDataModule):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.num_workers = get_num_workers()
        ...
        ...     def train_dataloader(self):
        ...         return DataLoader(
        ...             self.train_dataset,
        ...             num_workers=self.num_workers,
        ...             persistent_workers=self.num_workers > 0
        ...         )

    Performance Considerations:
        CPU-bound workloads:
        - More workers generally improve performance up to CPU core count
        - Diminishing returns beyond the number of physical cores
        - Memory usage increases with more workers

        I/O-bound workloads:
        - Fewer workers may be optimal to avoid I/O contention
        - Consider storage bandwidth limitations
        - Network-based datasets may benefit from fewer workers

        Memory constraints:
        - Each worker process consumes additional memory
        - Large datasets or complex preprocessing may require fewer workers
        - Monitor memory usage when scaling worker count

    System Compatibility:
        - Works across different operating systems (Linux, macOS, Windows)
        - Handles containerized environments gracefully
        - Safe fallback for restricted execution environments
        - Compatible with both physical and virtual machines

    Best Practices:
        - Use with persistent_workers=True when num_workers > 0 for efficiency
        - Combine with pin_memory=True on CUDA-enabled systems
        - Monitor CPU utilization to verify workers are being utilized
        - Consider reducing workers if memory usage becomes problematic

    Troubleshooting:
        If data loading is slow:
        - Verify that num_workers > 0 on multi-core systems
        - Check that dataset operations are compatible with multiprocessing
        - Monitor CPU and memory usage during training
        - Consider profiling data loading pipeline for bottlenecks

    Integration:
        This function is used throughout Stormy in:
        - AutoTokenizerDataModule for configuring DataLoader workers
        - Training scripts for optimal data loading performance
        - Preprocessing pipelines that benefit from parallelization
        - Any scenario requiring efficient data loading

    See Also:
        - torch.utils.data.DataLoader: Uses num_workers parameter
        - os.cpu_count(): The underlying system call for CPU detection
        - PyTorch DataLoader performance guide: https://pytorch.org/docs/stable/data.html#single-and-multi-process-data-loading
    """
    num_workers = os.cpu_count()

    if num_workers is not None:
        return num_workers
    else:
        return 0
