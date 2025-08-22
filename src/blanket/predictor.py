import os

import torch

from blanket.models import ToxicityClassifier

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Blanket:
    def __init__(self, checkpoint: str, threshold: float = 0.5, device="cpu") -> None:
        self.model = ToxicityClassifier.load_from_checkpoint(
            checkpoint, map_location=device
        )

    @torch.no_grad()
    def predict(self, text: str | list[str]) -> None:
        self.model.eval()
        text = [text] if isinstance(text, str) else text
        predictions = self.model(text)["outputs"].detach().cpu()

        for _text, _pred in zip(text, predictions, strict=True):
            print(f"\nText: {_text}")
            print("Label probabilities:")
            print(_pred)


if __name__ == "__main__":
    ckpt_path = "lightning_logs/version_0/checkpoints/epoch=06-val_loss=0.0487.ckpt"
    Blanket(ckpt_path).predict(
        ["you're such a bitch!", "can we be friends?", "the weather is great."]
    )
