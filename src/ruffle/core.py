"""High-level prediction interface for toxicity classification.

This module provides the main user-facing API for loading pre-trained toxicity
classification models and making predictions on text data. It includes both
a Python class interface and a command-line interface.
"""

import torch
import torch.nn.functional as F
from colorama import Fore, Style, init
from lightning.pytorch import LightningModule
from torch import Tensor

from ruffle.model import RuffleModel

DOWNLOAD_BASE_URL = "https://github.com/zuzo-sh/ruffle/releases/download/"
AVAILABLE_MODELS = {
    "bert-tiny": f"{DOWNLOAD_BASE_URL}v0.0.1alpha2/finetuned-bert-tiny.ckpt",
}

CKPT_PATH = "lightning_logs/version_2/checkpoints/epoch=07-val_loss=0.0454.ckpt"


class Ruffle:
    """A pre-trained toxicity classification model for content moderation.

    This class provides an interface for loading and using fine-tuned transformer models
    to detect toxic content in text. It supports both local checkpoints and remote
    pre-trained models with configurable classification thresholds.

    Example:
        >>> classifier = Ruffle(model_name="bert-tiny", threshold=0.7)
        >>> results = classifier.predict("I love your work!")
        >>> print(results)
    """

    def __init__(
        self,
        model_name: str | None = None,
        ckpt_path: str | None = None,
        threshold: float = 0.5,
        device: str = "cpu",
    ) -> None:
        """Initialize the toxicity classification model.

        Args:
            model_name (str): Name of the pre-trained model to use. Available models
                can be found in the AVAILABLE_MODELS dictionary. Defaults to "bert-tiny".
            ckpt_path (str | None): Path to a local model checkpoint
                file. If None, the model will be downloaded from the remote repository
                using the specified model_name. Defaults to None.
            threshold (float): Classification threshold for determining positive predictions.
                Values above this threshold are classified as toxic. Must be between
                0.0 and 1.0. Defaults to 0.5.
            device (str): PyTorch device specification for model inference. Common values
                include "cpu", "cuda", "cuda:0", etc. Defaults to "cpu".

        Raises:
            ValueError: If model_name is not found in AVAILABLE_MODELS or if threshold
                is not within the valid range [0.0, 1.0].

        Note:
            The model will be automatically downloaded on first use if no local
            ckpt_path is provided.
        """
        self.model_name = model_name
        self.ckpt_path = ckpt_path
        self.threshold = threshold
        self.device = device
        self._validate_inputs()

        self.model = self._load_model()

    def _validate_inputs(self) -> None:
        """Validate constructor inputs.

        Args:
            model_name: Name of the model to validate.
            threshold: Classification threshold to validate.

        Raises:
            ValueError: If model_name is not available or threshold is invalid.
        """
        if self.model_name is None and self.ckpt_path is None:
            raise ValueError("Must provided either `model_name` or `ckpt_path`.")

        if self.model_name is not None and self.model_name not in AVAILABLE_MODELS:
            available = ", ".join(AVAILABLE_MODELS.keys())
            raise ValueError(
                f"Unknown model '{self.model_name}'. Available: {available}"
            )

        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(
                f"Threshold must be between 0.0 and 1.0, got {self.threshold}"
            )

    def _load_model(self) -> LightningModule:
        if self.ckpt_path is not None:
            return RuffleModel.load_from_checkpoint(
                self.ckpt_path, map_location=self.device
            )
        else:
            return RuffleModel.load_from_checkpoint(
                AVAILABLE_MODELS[self.model_name], map_location=self.device
            )

    def classify(self, text: str | list[str], pretty_print: bool = True) -> list:
        self.model.eval()
        with torch.no_grad():
            logits = self.model(text)[0].detach()
            probabilities = F.sigmoid(logits)
        results = self._make_results_list(text, probabilities)
        if pretty_print:
            self._print_results(results)
        return results

    def _make_results_list(self, text: str | list[str], probs: Tensor) -> list[dict]:
        texts = [text] if isinstance(text, str) else text
        label_names = self.model.hparams.get("label_names")

        results = []
        for text, prob_vector in zip(texts, probs, strict=True):
            result = {}
            result["text"] = text
            if label_names is not None:
                for label, prob in zip(label_names, prob_vector, strict=True):
                    result[label] = prob.item()
            else:
                result["probabilities"] = prob_vector
            results.append(result)
        return results

    def _print_results(self, results: list[dict]) -> None:
        init(autoreset=True)
        separator = f"{Style.BRIGHT}{'=' * 60}{Style.RESET_ALL}"

        for result in results:
            print(separator)
            for key, val in result.items():
                if isinstance(val, float):
                    color = Fore.RED if val >= self.threshold else Fore.GREEN
                    print(f"{key:<15}: {color}{val:.2f}{Style.RESET_ALL}")
                else:
                    print(f"{key:<15}: {val}")
            print(separator)
            print()


if __name__ == "__main__":
    ruffle = Ruffle(ckpt_path=CKPT_PATH)
    r = ruffle.classify(["i miss maddie", "fuck you cunt"], pretty_print=True)
