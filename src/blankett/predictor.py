import torch
from jsonargparse import auto_cli
from lightning.pytorch import LightningModule
from torch import Tensor
from transformers import logging

from blankett.models import ToxicityClassifier

logging.set_verbosity_error()

DOWNLOAD_URL: str = "https://github.com/sulzyy/blankett/releases/download/"

MODEL_URLS: dict[str, str] = {
    "bert-tiny": DOWNLOAD_URL + "v0.0.1alpha1/bert_tiny.ckpt",
}


class Blankett:
    def __init__(
        self,
        model_name: str = "bert-tiny",
        checkpoint_path: str | None = None,
        threshold: float = 0.5,
        device: str = "cpu",
    ) -> None:
        """
        Args:
            model_name: Name of the remote pre-trained model to download and use.
            checkpoint_path: Path to local model checkpoint (if None, downloads from URL with `model_name`).
            threshold: Classification threshold for positive predictions.
            device: Device to run inference on (cpu/cuda).
        """
        self.threshold = threshold
        self.model = self._load_model(model_name, checkpoint_path, device)

    @torch.no_grad()
    def predict(
        self,
        text: str | list[str],
        pretty_print: bool = True,
    ) -> dict[str, Tensor | dict[str, Tensor]]:
        """Predict toxicity for given text(s).

        Args:
            text: Single text string or list of text strings to classify.

        Returns:
            Dictionary mapping input text to prediction results.
        """
        self.model.eval()
        text = [text] if isinstance(text, str) else text
        outputs: Tensor = self.model(text)["outputs"].detach().cpu()
        results: dict[str, Tensor | dict[str, Tensor]] = {}
        label_names: list[str] | None = self.model.hparams.get("label_names", None)
        for _text, prob_vector in zip(text, outputs, strict=True):
            if label_names is not None:
                prob_dict = dict(zip(label_names, prob_vector, strict=True))
                results[_text] = prob_dict
            else:
                results[_text] = prob_vector
        self._print_results(results)
        return results

    def _load_model(
        self, model_name: str, checkpoint_path: str | None, device: str
    ) -> LightningModule:
        if checkpoint_path is None:
            checkpoint_path = MODEL_URLS[model_name]
        return ToxicityClassifier.load_from_checkpoint(
            checkpoint_path, map_location=device
        )

    def _print_results(self, results: dict[str, Tensor | dict[str, Tensor]]) -> None:
        sep = "-" * 50
        for text, result in results.items():
            print(sep)
            print(f'Input: "{text}"')
            if isinstance(result, Tensor):
                print("Raw outputs:", result)
            elif isinstance(result, dict):
                print("Predictions:")
                for label, prob in result.items():
                    prob_float = float(prob)
                    marker = "✓" if prob_float >= self.threshold else " "
                    print(f"  [{marker}] {label:<15} {prob_float:.2%}")
            print(sep)
            print()


def classify(
    text: str | list[str],
    model_name: str = "bert-tiny",
    checkpoint_path: str | None = None,
    threshold: float = 0.5,
    device: str = "cpu",
) -> None:
    """CLI interface for Blankett toxicity classification.

    Args:
        text: Text to classify for toxicity.
        model_name: Name of the remote pre-trained model to download and use.
        checkpoint_path: Path to local model checkpoint (if None, downloads from URL with `model_name`).
        threshold: Classification threshold for positive predictions.
        device: Device to run inference on (cpu/cuda).
    """
    classifier = Blankett(
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        threshold=threshold,
        device=device,
    )
    _ = classifier.predict(text, pretty_print=True)


def cli_main():
    auto_cli(classify)


if __name__ == "__main__":
    cli_main()
