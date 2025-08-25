import torch
from lightning.pytorch import LightningModule
from torch import Tensor

from blanket.models import ToxicityClassifier

DOWNLOAD_URL: str = "https://github.com/th0rne/blanket/releases/download/"

MODEL_URLS: dict[str, str] = {
    "bert-tiny": DOWNLOAD_URL + "v0.0.1a2/toxic_bert_tiny.ckpt",
}


class Blanket:
    def __init__(
        self,
        model_name: str = "bert-tiny",
        checkpoint_path: str | None = None,
        threshold: float = 0.5,
        device: str = "cpu",
    ) -> None:
        self.threshold = threshold
        self.model = self.load_model(model_name, checkpoint_path, device)

    @torch.no_grad()
    def predict(self, text: str | list[str]) -> dict[str, Tensor | dict[str, Tensor]]:
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

        return results

    def load_model(
        self, model_name: str, checkpoint_path: str | None, device: str
    ) -> LightningModule:
        if checkpoint_path is None:
            checkpoint_path = MODEL_URLS[model_name]
        return ToxicityClassifier.load_from_checkpoint(
            checkpoint_path, map_location=device
        )
