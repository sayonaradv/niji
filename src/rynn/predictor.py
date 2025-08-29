import torch
from jsonargparse import auto_cli
from lightning.pytorch import LightningModule
from torch import Tensor

from rynn.models import Classifier
from rynn.types import PredResult, TextInput

DOWNLOAD_BASE_URL = "https://github.com/dbozbay/rynn/releases/download/"
AVAILABLE_MODELS = {
    "bert-tiny": f"{DOWNLOAD_BASE_URL}v0.0.1alpha1/bert_tiny.ckpt",
}


class Rynn:
    """A pre-trained toxicity classification model for content moderation.

    This class provides an interface for loading and using fine-tuned transformer models
    to detect toxic content in text. It supports both local checkpoints and remote
    pre-trained models with configurable classification thresholds.

    Example:
        >>> classifier = Rynn(model_name="bert-tiny", threshold=0.7)
        >>> results = classifier.predict("I love your work!")
        >>> print(results)
    """

    def __init__(
        self,
        model_name: str = "bert-tiny",
        checkpoint_path: str | None = None,
        threshold: float = 0.5,
        device: str = "cpu",
    ) -> None:
        """Initialize the toxicity classification model.

        Args:
            model_name (str): Name of the pre-trained model to use. Available models
                can be found in the AVAILABLE_MODELS dictionary. Defaults to "bert-tiny".
            checkpoint_path (str | None): Path to a local model checkpoint
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
            checkpoint_path is provided.
        """
        self._validate_inputs(model_name, threshold)

        self.model_name = model_name
        self.threshold = threshold
        self.device = device
        self.model = self._load_model(checkpoint_path)

    def _validate_inputs(self, model_name: str, threshold: float) -> None:
        """Validate constructor inputs."""
        if model_name not in AVAILABLE_MODELS:
            available = ", ".join(AVAILABLE_MODELS.keys())
            raise ValueError(f"Unknown model '{model_name}'. Available: {available}")

        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be between 0.0 and 1.0, got {threshold}")

    def _load_model(self, checkpoint_path: str | None) -> LightningModule:
        """Load model from checkpoint path or download URL."""
        if checkpoint_path is None:
            checkpoint_path = AVAILABLE_MODELS[self.model_name]

        return Classifier.load_from_checkpoint(
            checkpoint_path, map_location=self.device
        )

    @torch.no_grad()
    def predict(
        self,
        texts: TextInput,
        verbose: bool = True,
    ) -> PredResult:
        """Predict toxicity for input text(s).

        Args:
            texts: Single text string or list of strings to classify.
            verbose: If True, print results to stdout.

        Returns:
            Dictionary mapping input texts to their prediction results.
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
        """Format raw model outputs into stuctured results."""
        results = {}
        label_names = self.model.hparams.get("label_names")

        for text, prob_vector in zip(texts, outputs, strict=True):
            if label_names:
                results[text] = dict(zip(label_names, prob_vector, strict=True))
            else:
                results[text] = prob_vector

        return results

    def _print_results(self, results: PredResult) -> None:
        """Print formatted prediction results."""
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
    checkpoint_path: str | None = None,
    threshold: float = 0.5,
    device: str = "cpu",
) -> None:
    """A toxicity classifier that can load pretrained models and make predictions.

    Args:
        texts (TextInput): Single text string or list of strings to classify.
        model_name (str): Name of the pre-trained model to use. Available models
            can be found in the AVAILABLE_MODELS dictionary. Defaults to "bert-tiny".
        checkpoint_path (str | None): Path to a local model checkpoint
            file. If None, the model will be downloaded from the remote repository
            using the specified model_name. Defaults to None.
        threshold (float): Classification threshold for determining positive predictions.
            Values above this threshold are classified as toxic. Must be between
            0.0 and 1.0. Defaults to 0.5.
        device (str): PyTorch device specification for model inference. Common values
            include "cpu", "cuda", "cuda:0", etc. Defaults to "cpu".

    Example:
        Command line usage:

        ```bash
        python predictor.py --text "Hello world" --threshold 0.7
        python predictor.py --text '["Text 1", "Text 2"]' --model_name bert-tiny
        ```
    """
    classifier = Rynn(
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        threshold=threshold,
        device=device,
    )
    _ = classifier.predict(texts)


def cli_main():
    """Entry point for CLI."""
    auto_cli(classify)


if __name__ == "__main__":
    cli_main()
