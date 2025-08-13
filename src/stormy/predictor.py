import torch
from jsonargparse import auto_cli

from stormy.module import SequenceClassificationModule


def predict(
    texts: str | list[str],
    checkpoint_path: str,
    threshold: float = 0.5,
    labels: list[str] | None = None,
) -> None:
    """
    Predict labels for text using a trained multi-label classifier.

    Args:
        texts: One or more text strings to classify.
        checkpoint_path: Path to the trained model checkpoint.
        threshold: Probability threshold for positive labels.
        labels: List of label names.
    """
    model = SequenceClassificationModule.load_from_checkpoint(
        checkpoint_path, map_location="cpu"
    )
    model.to("cpu")
    model.eval()

    with torch.no_grad():
        logits = model(texts)
        probs = torch.sigmoid(logits).detach().cpu()
        predictions = (probs >= threshold).int().tolist()

    for text, preds in zip(texts, predictions, strict=False):
        print(f"Text: {text}")
        print(f"Predicted labels: {preds}")
        print("-" * 40)

    # results = []
    # for pred in predictions:
    #     pos_indices = torch.nonzero(pred, as_tuple=True)[0].tolist()
    #     if labels is not None:
    #         results.append([labels[i] for i in pos_indices])
    #     else:
    #         results.append(pos_indices)


if __name__ == "__main__":
    auto_cli(predict)
