import torch
from lightning.pytorch import LightningModule
from torch import Tensor

from blankett.models import ToxicityClassifier

DOWNLOAD_URL: str = "https://github.com/yourthorne/blankett/releases/download/"

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
        self.threshold = threshold
        self.model = self.load_model(model_name, checkpoint_path, device)

    @torch.no_grad()
    def predict(
        self, text: str | list[str], verbose: bool = False
    ) -> dict[str, Tensor | dict[str, Tensor]]:
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
        if verbose:
            self._print_results(results)
        return results

    def load_model(
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
                    marker = "✔" if prob_float >= self.threshold else " "
                    print(f"  [{marker}] {label:<15} {prob_float:.2%}")
            print(sep)
            print()


if __name__ == "__main__":
    _ = Blankett().predict(
        [
            "I love your work!",
            "You're a total loser.",
            "If you don't refund me, I will report your crap company.",
        ],
        verbose=True,
    )
