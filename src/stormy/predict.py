from argparse import ArgumentParser

import torch

from stormy.module import SequenceClassificationModule


@torch.inference_mode()
def main(text: str, checkpoint_path: str) -> None:
    model = SequenceClassificationModule.load_from_checkpoint(
        checkpoint_path, map_location="cpu"
    )
    model.eval()
    logits = model(text)
    probabilities = torch.sigmoid(logits).cpu()

    print(f"Text: {text}")
    print(f"Output probabilities: {probabilities}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        help="Text to classify",
        default="This is an amazing movie! So much depth and emotion. I loved it!",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to the checkpoint file",
        default="lightning_logs/version_4/checkpoints/epoch=19-val_loss=0.3007.ckpt",
    )
    args = parser.parse_args()
    main(args.input, args.checkpoint)
