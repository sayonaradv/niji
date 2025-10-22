from pathlib import Path

import torch

from niji.exceptions import ModelNotFoundError
from niji.module import Classifier

# See https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
torch.set_float32_matmul_precision("medium")

DOWNLOAD_BASE_URL = "https://github.com/sudojayder/niji/releases/download/"
AVAILABLE_MODELS = {
    "bert-tiny": f"{DOWNLOAD_BASE_URL}v0.0.1alpha4/finetuned-bert-tiny.ckpt",
    "distilbert": f"{DOWNLOAD_BASE_URL}v0.0.1alpha4/finetuned-distilbert.ckpt",
}

TEST_CKPT_PATH: str = (
    "runs/training_runs/bert-tiny/checkpoints/epoch=01-val_loss=0.0726.ckpt"
)


def load_checkpoint(
    model_name: str | None = None,
    ckpt_path: str | None = None,
    device: str | None = None,
) -> Classifier:
    """Load a Classifier from checkpoint, either locally or from remote location.

    Args:
        model_name (str, optional): Name of available model to download remotely.
        ckpt_path (str, optional): Path to local checkpoint file.
        device (str, optional): Device to load model on (e.g., 'cuda', 'cpu').

    Returns:
        Classifier: The loaded model instance.

    Raises:
        ValueError: If neither model_name nor ckpt_path is provided, or if
            model_name is not in AVAILABLE_MODELS.
        ModelNotFoundError: If the local checkpoint file doesn't exist.
    """
    if not (model_name or ckpt_path):
        raise ValueError("Must provide either 'model_name' or 'ckpt_path'.")

    if ckpt_path:
        # Use local checkpoint - validate it exists
        checkpoint_path = Path(ckpt_path)
        if not checkpoint_path.is_file():
            raise ModelNotFoundError(f"Checkpoint file does not exist: {ckpt_path}")
        final_path = str(checkpoint_path)
    else:
        # Use remote model - validate model_name
        if model_name is None or model_name not in AVAILABLE_MODELS:
            available_models: str = ", ".join(AVAILABLE_MODELS.keys())
            raise ValueError(
                f"Unknown model '{model_name}'. Available models: {available_models}"
            )
        final_path: str = AVAILABLE_MODELS[model_name]

    return Classifier.load_from_checkpoint(final_path, map_location=device)
