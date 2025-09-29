"""
Blanki CLI - A command-line interface for toxicity classification model training and inference.

This CLI provides three main commands:
- train: Fine-tune transformer models on the Jigsaw toxicity dataset
- test: Evaluate trained models on test data
- predict: Make predictions on raw text using trained models
"""

import sys

from jsonargparse import ArgumentParser

from blanki import inference, training


def print_welcome_msg() -> None:
    """Print ASCII art welcome message."""
    msg = """
 .o8       oooo                        oooo         o8o
"888       `888                        `888         `"'
 888oooo.   888   .oooo.   ooo. .oo.    888  oooo  oooo
 d88' `88b  888  `P  )88b  `888P"Y88b   888 .8P'   `888
 888   888  888   .oP"888   888   888   888888.     888
 888   888  888  d8(  888   888   888   888 `88b.   888
 `Y8bod8P' o888o `Y888""8o o888o o888o o888o o888o o888o
    """
    print(msg)


def create_parser() -> ArgumentParser:
    """Create and configure the main argument parser with subcommands."""

    # Main parser
    parser = ArgumentParser(
        prog="blanki",
        description="Multi-label toxicity classification with transformer models",
    )

    train_subparser = create_subparser("train")
    test_subparser = create_subparser("test")
    predict_subparser = create_subparser("predict")

    # Create subcommands
    subcommands = parser.add_subcommands()

    # Add subcommands using function signatures and docstrings
    subcommands.add_subcommand(
        "train",
        train_subparser,
        help="Train a transformer model for toxicity classification on the Jigsaw dataset.",
    )
    subcommands.add_subcommand(
        "test",
        test_subparser,
        help="Evaluate a trained model on the Jigsaw test dataset.",
    )
    subcommands.add_subcommand(
        "predict",
        predict_subparser,
        help="Predict the toxicity of text using a trained model.",
    )

    return parser


def create_subparser(subcommand: str) -> ArgumentParser:
    map = {
        "train": training.train,
        "test": inference.test,
        "predict": inference.predict,
    }

    parser = ArgumentParser()
    parser.add_function_arguments(map[subcommand])
    return parser


def run_train(args) -> None:
    """Execute the train command."""
    try:
        # Call training.train with the arguments directly
        # jsonargparse already matched the function signature
        training.train(**vars(args))
    except Exception as e:
        print(f"Error during training: {e}", file=sys.stderr)
        sys.exit(1)


def run_test(args) -> None:
    """Execute the test command."""
    try:
        # Call inference.test with the arguments directly
        metrics = inference.test(**vars(args))

        # Print results to stdout
        print("\n=== Test Results ===")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

    except Exception as e:
        print(f"Error during testing: {e}", file=sys.stderr)
        sys.exit(1)


def run_predict(args) -> None:
    """Execute the predict command."""
    try:
        # Call inference.predict with the arguments directly
        predictions = inference.predict(**vars(args))

        # If not verbose, we might want to output the raw predictions
        if not getattr(args, "verbose", True):
            if isinstance(args.text, str):
                print("Predictions:", predictions.tolist())
            else:
                print("Batch predictions:")
                for i, pred in enumerate(predictions):
                    print(f"  Text {i + 1}: {pred.tolist()}")

    except Exception as e:
        print(f"Error during prediction: {e}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    """Main CLI entry point."""
    print_welcome_msg()
    parser = create_parser()
    args = parser.parse_args()

    # Route to appropriate command handler
    if hasattr(args, "train"):
        run_train(args.train)
    elif hasattr(args, "test"):
        run_test(args.test)
    elif hasattr(args, "predict"):
        run_predict(args.predict)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
