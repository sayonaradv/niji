from collections.abc import Mapping
from pathlib import Path
from time import perf_counter

import lightning.pytorch as pl
import torch
from colorama import Fore, Style, init
from lightning.pytorch.loggers import TensorBoardLogger

from ruffle.dataloader import JigsawDataModule
from ruffle.module import RuffleModel
from ruffle.training import DATA_DIR
from ruffle.utils import log_perf

# See https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
torch.set_float32_matmul_precision("medium")

DOWNLOAD_BASE_URL = "https://github.com/zuzo-sh/ruffle/releases/download/"
AVAILABLE_MODELS = {
    "bert-tiny": f"{DOWNLOAD_BASE_URL}v0.0.1alpha4/finetuned-bert-tiny.ckpt",
    "distilbert": f"{DOWNLOAD_BASE_URL}v0.0.1alpha4/finetuned-distilbert.ckpt",
}


def load_checkpoint(
    model_name: str | None = None,
    ckpt_path: str | None = None,
    device: str | None = None,
) -> RuffleModel:
    """Load a RuffleModel from checkpoint, either locally or from remote location.

    Args:
        model_name (str, optional): Name of available model to download remotely.
        ckpt_path (str, optional): Path to local checkpoint file.
        device (str, optional): Device to load model on (e.g., 'cuda', 'cpu').

    Returns:
        RuffleModel: The loaded model instance.

    Raises:
        ValueError: If neither model_name nor ckpt_path is provided, or if
            model_name is not in AVAILABLE_MODELS.
        FileNotFoundError: If the local checkpoint file doesn't exist.
    """
    if not (model_name or ckpt_path):
        raise ValueError("Must provide either 'model_name' or 'ckpt_path'.")

    if ckpt_path:
        # Use local checkpoint - validate it exists
        checkpoint_path = Path(ckpt_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint file does not exist: {ckpt_path}")
        final_path = str(checkpoint_path)
    else:
        # Use remote model - validate model_name
        if model_name is None or model_name not in AVAILABLE_MODELS:
            available_models = ", ".join(AVAILABLE_MODELS.keys())
            raise ValueError(
                f"Unknown model '{model_name}'. Available models: {available_models}"
            )
        final_path = AVAILABLE_MODELS[model_name]

    return RuffleModel.load_from_checkpoint(final_path, map_location=device)


def test(
    model_name: str | None = None,
    ckpt_path: str | None = None,
    data_dir: str = DATA_DIR,
    batch_size: int = 64,
    perf: bool = True,
    run_name: str | None = None,
) -> Mapping[str, float]:
    """Evaluate a fine-tuned model on the test dataset.

    Args:
        model_name (str, optional): Name of available model to download remotely.
        ckpt_path (str, optional): Path to local checkpoint file.
        data_dir (str): Directory containing the test data. Defaults to DataConfig.data_dir.
        batch_size (int): Batch size for evaluation. Defaults to DataConfig.batch_size.
        run_name (str, optional): Name of the experiment for logging.
        perf (bool): Whether to log performance metrics. Defaults to True.

    Returns:
        Mapping[str, float]: Dictionary with metrics logged during the test phase.

    Raises:
        ValueError: If neither model_name nor ckpt_path is provided.
        FileNotFoundError: If checkpoint file doesn't exist.
    """
    model = load_checkpoint(model_name, ckpt_path)
    model.eval()

    datamodule = JigsawDataModule(
        data_dir, batch_size=batch_size, labels=model.hparams["label_names"]
    )

    logger = TensorBoardLogger(save_dir="runs", name="test_runs", version=run_name)
    trainer = pl.Trainer(logger=logger)

    if perf:
        start_time = perf_counter()
        metrics = trainer.test(model, datamodule=datamodule)[0]
        end_time = perf_counter()
        log_perf(start_time, end_time, trainer)
    else:
        metrics = trainer.test(model, datamodule=datamodule)[0]

    return metrics


def predict(
    text: str | list[str],
    model_name: str | None = None,
    ckpt_path: str | None = None,
    threshold: float = 0.5,
    device: str | None = "cpu",
    verbose: bool = True,
    return_logits: bool = False,
) -> torch.Tensor:
    """Make predictions on raw text using a fine-tuned model.

    Args:
        text (str or list[str]): Input text(s) to classify.
        model_name (str, optional): Name of available model to download remotely.
        ckpt_path (str, optional): Path to local checkpoint file.
        threshold (float):
        device (str, optional): Device to run inference on. Defaults to model's device.
        verbose (bool): Whether to print human-readable results. Defaults to True.
        return_logits (bool): Whether to return raw logits instead of probabilities.
            Defaults to False.

    Returns:
        torch.Tensor: Model predictions as probabilities (or logits if return_logits=True).
            Shape: (batch_size, num_labels) or (num_labels,) for single input.

    Raises:
        ValueError: If neither model_name nor ckpt_path is provided, or if text is empty.
        TypeError: If text is not a string or list of strings.
    """
    if isinstance(text, str):
        is_single_input = True
        text_batch = [text]
    else:
        is_single_input = False
        text_batch = text

    model = load_checkpoint(model_name, ckpt_path, device)
    model.eval()

    with torch.no_grad():
        logits, _ = model(text)

        if return_logits:
            predictions = logits.detach()
        else:
            predictions = torch.sigmoid(logits).detach()

    if verbose:
        label_names = model.hparams["label_names"]
        _print_results(text_batch, predictions, label_names, threshold)

    if is_single_input:
        return predictions.squeeze(0)
    else:
        return predictions


def _print_results(
    text_batch: list[str],
    predictions: torch.Tensor,
    label_names: list[str] | None = None,
    threshold: float = 0.5,
) -> None:
    """Print pretty prediction results with colorful output.

    Args:
        text_batch (list[str]): Input texts that were classified.
        predictions (torch.Tensor): Model predictions as probabilities.
            Shape: (batch_size, num_labels).
        label_names (list[str], optional): Names of the classification labels.
            If None, shows indexed predictions.
        threshold (float): Threshold for positive classification. Defaults to 0.5.
    """
    init(autoreset=True)  # Auto-reset colors after each print

    # Color definitions
    POSITIVE_COLOR = Fore.RED + Style.BRIGHT
    NEGATIVE_COLOR = Fore.GREEN + Style.DIM
    TEXT_COLOR = Fore.CYAN + Style.BRIGHT
    SEPARATOR_COLOR = Fore.BLUE + Style.BRIGHT
    SUMMARY_COLOR = Fore.YELLOW + Style.BRIGHT
    RESET = Style.RESET_ALL

    separator = f"{SEPARATOR_COLOR}{'=' * 80}{RESET}"

    for i, (text, pred_vector) in enumerate(zip(text_batch, predictions, strict=True)):
        # Add spacing between samples (except for the first one)
        if i > 0:
            print()

        print(separator)
        print(f"{TEXT_COLOR}Text:{RESET} {text}")
        print(f"{SEPARATOR_COLOR}{'-' * 40}{RESET}")

        if label_names:
            # Print predictions with named labels
            positive_labels = []
            for pred_score, label in zip(pred_vector, label_names, strict=True):
                if pred_score > threshold:
                    color = POSITIVE_COLOR
                    icon = "✓"
                    status = "POSITIVE"
                    positive_labels.append(label)
                else:
                    color = NEGATIVE_COLOR
                    icon = "✗"
                    status = "negative"

                print(f"{label:.<20} {pred_score:.3f} {color}{icon} {status}{RESET}")

            # Summary line for quick reference
            print()  # Add spacing before summary
            if positive_labels:
                labels_str = ", ".join(positive_labels)
                print(
                    f"{SUMMARY_COLOR}Predicted labels:{RESET} {POSITIVE_COLOR}{labels_str}{RESET}"
                )
            else:
                print(
                    f"{SUMMARY_COLOR}Result:{RESET} {NEGATIVE_COLOR}No labels above threshold ({threshold}){RESET}"
                )
        else:
            # Print predictions with indices when no label names available
            positive_indices = []
            for idx, pred_score in enumerate(pred_vector):
                if pred_score > threshold:
                    color = POSITIVE_COLOR
                    icon = "✓"
                    status = "POSITIVE"
                    positive_indices.append(str(idx))
                else:
                    color = NEGATIVE_COLOR
                    icon = "✗"
                    status = "negative"

                print(
                    f"Label {idx}............... {pred_score:.3f} {color}{icon} {status}{RESET}"
                )

            # Summary line for quick reference
            print()  # Add spacing before summary
            if positive_indices:
                indices_str = ", ".join(positive_indices)
                print(
                    f"{SUMMARY_COLOR}📋 Positive indices:{RESET} {POSITIVE_COLOR}{indices_str}{RESET}"
                )
            else:
                print(
                    f"{SUMMARY_COLOR}📋 Result:{RESET} {NEGATIVE_COLOR}No predictions above threshold ({threshold}){RESET}"
                )


if __name__ == "__main__":
    ckpt_path = "runs/training_runs/bert-tiny/checkpoints/epoch=01-val_loss=0.0726.ckpt"

    sample_text = ["i hate you", "you're a dumbass!", "have fun guys"]
    preds = predict(sample_text, ckpt_path=ckpt_path)
