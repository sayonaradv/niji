import torch

from .exceptions import DataNotFoundError, ModelNotFoundError, NijiError
from .inference import load_checkpoint
from .setup import env_vars, logging

env_vars()
logging()


__version__ = "0.0.1a4"

__all__: list[str] = [
    "DataNotFoundError",
    "ModelNotFoundError",
    "Niji",
    "NijiError",
]


class Niji:
    """Main class for Niji toxicity detection."""

    def __init__(self, model_name: str | None = None, ckpt_path: str | None = None):
        self.model = load_checkpoint(model_name, ckpt_path)

    def predict(self, text: str | list[str]) -> dict:
        """Predict toxicity scores for text(s)."""
        texts = [text] if isinstance(text, str) else text
        logits = self.model(texts)
        probs = torch.sigmoid(logits).detach().cpu()
        labels = self.model.hparams["label_names"]
        results = {}
        for i, txt in enumerate(texts):
            scores = {label: probs[i, j].item() for j, label in enumerate(labels)}
            results[txt] = scores
        return results


def main() -> None:
    print("Hello from niji! ⛈️")
