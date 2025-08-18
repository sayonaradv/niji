import torch
from jsonargparse import auto_cli

from stormy.config import JIGSAW_LABELS
from stormy.module import SequenceClassificationModule


def predict(
    texts: str | list[str],
    checkpoint_path: str = "lightning_logs/version_1/checkpoints/epoch=09-val_loss=0.0479.ckpt",
    threshold: float = 0.5,
) -> list[list[str]] | list[list[float]]:
    """
    Predict toxicity labels for given texts.

    Args:
        texts: Text strings to classify.
        checkpoint_path: Path to the model checkpoint.
        threshold: Threshold for positive label classification.
    """
    model = SequenceClassificationModule.load_from_checkpoint(
        checkpoint_path, map_location="cpu"
    )
    model.eval()

    texts = [texts] if isinstance(texts, str) else texts

    with torch.no_grad():
        logits = model(texts)
        probabilities = torch.sigmoid(logits).detach().cpu()

    results = []

    for text, prob_vector in zip(texts, probabilities, strict=True):
        print(f"\nText: {text}")

        print("Label probabilities:")
        for name, prob in zip(JIGSAW_LABELS, prob_vector, strict=True):
            print(f"  {name:<20} {prob:.3f}")

        positive_labels = [
            name
            for name, p in zip(JIGSAW_LABELS, prob_vector, strict=True)
            if p >= threshold
        ]
        print(f"Positive labels: {positive_labels or 'None'}")
        results.append(positive_labels)

    return results


def cli_main() -> None:
    auto_cli(predict)


if __name__ == "__main__":
    cli_main()
