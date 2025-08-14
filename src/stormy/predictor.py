import torch
from jsonargparse import CLI

# from stormy.config import JIGSAW_LABELS
from stormy.module import SequenceClassificationModule


def predict(
    texts: str | list[str],
    checkpoint_path: str = "lightning_logs/version_1/checkpoints/epoch=09-val_loss=0.0479.ckpt",
    threshold: float = 0.5,
    label_names: list[str] | None = None,
) -> list[list[str]] | list[list[float]]:
    """
    Predict labels for one or more texts using a trained multi-label classifier.

    Args:
        texts: A string or list of text strings to classify.
        checkpoint_path: Path to the trained model checkpoint.
        threshold: Probability threshold for positive labels.
        label_names: Optional list of label names corresponding to model outputs.

    Returns:
        If label_names is given:
            List of predicted label lists, one per input text.
        Else:
            List of probability lists, one per input text.
    """
    if isinstance(texts, str):
        texts = [texts]

    model = SequenceClassificationModule.load_from_checkpoint(
        checkpoint_path, map_location="cpu"
    )
    model.eval()

    with torch.no_grad():
        logits = model(texts)
        probabilities = torch.sigmoid(logits).detach().cpu()

    results = []

    for text, prob_vector in zip(texts, probabilities, strict=True):
        print(f"\nText: {text}")

        if label_names:
            print("Label probabilities:")
            for name, prob in zip(label_names, prob_vector, strict=True):
                print(f"  {name:<20} {prob:.3f}")

            positive_labels = [
                name
                for name, p in zip(label_names, prob_vector, strict=True)
                if p >= threshold
            ]
            print(f"Predicted labels: {positive_labels or 'None'}")
            results.append(positive_labels)
        else:
            rounded_probs = [round(p.item(), 3) for p in prob_vector]
            print(f"Predicted probabilities: {rounded_probs}")
            results.append(rounded_probs)

    return results


def main() -> None:
    CLI(predict, as_positional=False)


if __name__ == "__main__":
    main()
