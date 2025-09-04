"""High-level prediction interface for toxicity classification.

This module provides the main user-facing API for loading pre-trained toxicity
classification models and making predictions on text data. It includes both
a Python class interface and a command-line interface.
"""

import torch
from jsonargparse import auto_cli
from lightning.pytorch import LightningModule
from torch import Tensor

from ruffle.models import Classifier
from ruffle.types import PredResult, TextInput

DOWNLOAD_BASE_URL = "https://github.com/zuzo-sh/ruffle/releases/download/"
AVAILABLE_MODELS = {
    "bert-tiny": f"{DOWNLOAD_BASE_URL}v0.0.1alpha2/finetuned-bert-tiny.ckpt",
}
"""Dictionary mapping model names to their download URLs."""


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
        model_name: str = "bert-tiny",
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
        self._validate_inputs(model_name, threshold)

        self.model_name = model_name
        self.threshold = threshold
        self.device = device
        self.model = self._load_model(ckpt_path)

    def _validate_inputs(self, model_name: str, threshold: float) -> None:
        """Validate constructor inputs.

        Args:
            model_name: Name of the model to validate.
            threshold: Classification threshold to validate.

        Raises:
            ValueError: If model_name is not available or threshold is invalid.
        """
        if model_name not in AVAILABLE_MODELS:
            available = ", ".join(AVAILABLE_MODELS.keys())
            raise ValueError(f"Unknown model '{model_name}'. Available: {available}")

        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be between 0.0 and 1.0, got {threshold}")

    def _load_model(self, ckpt_path: str | None) -> LightningModule:
        """Load model from checkpoint path or download URL.

        Args:
            ckpt_path: Local path to checkpoint file, or None to use
                pre-configured download URL.

        Returns:
            Loaded LightningModule model ready for inference.
        """
        if ckpt_path is None:
            ckpt_path = AVAILABLE_MODELS[self.model_name]

        return Classifier.load_from_checkpoint(ckpt_path, map_location=self.device)

    @torch.no_grad()
    def predict(
        self,
        texts: TextInput,
        verbose: bool = True,
    ) -> PredResult:
        """Predict toxicity for input text(s).

        Args:
            texts: Single text string or list of strings to classify.
            verbose: If True, print formatted results to stdout.

        Returns:
            Dictionary mapping input texts to their prediction results.
            If the model has label_names, returns probabilities for each label.
            Otherwise returns raw probability tensors.
        """
        self.model.eval()

        text_list = [texts] if isinstance(texts, str) else texts
        outputs: Tensor = self.model(text_list)["outputs"].detach().cpu()
        results = self._format_predictions(text_list, outputs)

        if verbose:
            self._print_results(results)

        return results

    def _format_predictions(
        self,
        texts: TextInput,
        outputs: Tensor,
    ) -> PredResult:
        """Format raw model outputs into structured results.

        Args:
            texts: Input texts (list of strings).
            outputs: Model probability outputs with shape (batch_size, num_labels).

        Returns:
            Dictionary mapping each input text to either a dictionary of
            label probabilities (if model has label_names) or raw tensor.
        """
        results = {}
        label_names = self.model.hparams.get("label_names")

        for text, prob_vector in zip(texts, outputs, strict=True):
            if label_names:
                results[text] = dict(zip(label_names, prob_vector, strict=True))
            else:
                results[text] = prob_vector

        return results

    def _print_results(self, results: PredResult) -> None:
        """Print formatted prediction results to stdout.

        Args:
            results: Dictionary of prediction results to display.
        """
        separator = "=" * 60

        for text, result in results.items():
            print(separator)
            print(f'Input: "{text}"')
            if isinstance(result, Tensor):
                print("Raw outputs:", result)
            elif isinstance(result, dict):
                print("Predictions:")
                for label, prob in result.items():
                    prob_float = float(prob)
                    marker = "✓" if prob_float >= self.threshold else " "
                    print(f"  [{marker}] {label:<15} {prob_float:.2%}")
            print(separator)
            print()


def classify(
    texts: TextInput,
    model_name: str = "bert-tiny",
    ckpt_path: str | None = None,
    threshold: float = 0.5,
    device: str = "cpu",
) -> None:
    """Command-line interface for toxicity classification.

    A convenience function that creates a Ruffle classifier and makes predictions
    on the provided texts. Results are automatically printed to stdout.

    Args:
        texts: Single text string or list of strings to classify.
        model_name: Name of the pre-trained model to use. Available models
            can be found in the AVAILABLE_MODELS dictionary.
        ckpt_path: Path to a local model checkpoint file. If None,
            the model will be downloaded from the remote repository.
        threshold: Classification threshold for determining positive predictions.
            Values above this threshold are classified as toxic.
        device: PyTorch device specification for model inference.

    Example:
        Command line usage:

        ```bash
        python predictor.py --texts "Hello world" --threshold 0.7
        python predictor.py --texts '["Text 1", "Text 2"]' --model_name bert-tiny
        ```
    """
    classifier = Ruffle(
        model_name=model_name,
        ckpt_path=ckpt_path,
        threshold=threshold,
        device=device,
    )
    _ = classifier.predict(texts)


def cli_main() -> None:
    """Entry point for the command-line interface.

    Uses jsonargparse to automatically generate a CLI from the classify function.
    This allows users to call the predictor from the command line with automatic
    argument parsing and help generation.
    """
    auto_cli(classify)


if __name__ == "__main__":
    cli_main()
