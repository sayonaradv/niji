"""
Blanki CLI - A command-line interface for toxicity classification model training and inference.

This CLI provides three main commands:
- train: Fine-tune transformer models on the Jigsaw toxicity dataset
- test: Evaluate trained models on test data
- predict: Make predictions on raw text using trained models
"""

import sys
from argparse import Namespace
from collections.abc import Callable
from typing import Literal

from jsonargparse import ArgumentParser
from pydantic import ValidationError
from rich.console import Console

from blanki import __version__, inference, training
from blanki.exceptions import DataNotFoundError, ModelNotFoundError

# Global console instance for consistent styling
console = Console()


def create_parser() -> ArgumentParser:
    """Create and configure the main argument parser with subcommands."""
    parser = ArgumentParser(
        prog="blanki",
        description="⚡ Lightning-fast toxicity classification with transformer models.",
        version=__version__,
    )

    train_subparser: ArgumentParser = create_subparser("train")
    test_subparser: ArgumentParser = create_subparser("test")
    predict_subparser: ArgumentParser = create_subparser("predict")

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


def create_subparser(subcommand: Literal["train", "test", "predict"]) -> ArgumentParser:
    """Create a subparser for the given subcommand.

    Args:
        subcommand: The subcommand name, must be one of "train", "test", or "predict"

    Returns:
        ArgumentParser configured for the specified subcommand

    Raises:
        ValueError: If subcommand is not one of the valid options
    """
    # Define the mapping with explicit validation
    command_map: dict[Literal["train", "test", "predict"], Callable] = {
        "train": training.train,
        "test": inference.test,
        "predict": inference.predict,
    }

    # This check is redundant with Literal type but provides clear error message
    if subcommand not in command_map:
        valid_commands: str = ", ".join(f'"{cmd}"' for cmd in command_map)
        raise ValueError(
            f"Invalid subcommand '{subcommand}'. Must be one of: {valid_commands}"
        )

    parser = ArgumentParser()
    parser.add_function_arguments(command_map[subcommand])
    return parser


def run_train(**kwargs) -> None:
    """Execute the train command."""
    try:
        with console.status("[bold blue]Training model...", spinner="aesthetic"):
            training.train(**kwargs)

        console.print("[bold green]Training completed successfully![/bold green]")
    except DataNotFoundError as e:
        console.print(f"[bold red]Data not found:[/bold red] {e}")
        sys.exit(2)
    except Exception as e:
        console.print(f"[bold red]Training failed:[/bold red] {e}")
        console.print_exception()
        sys.exit(1)


def run_test(**kwargs) -> None:
    """Execute the test command."""
    try:
        with console.status("[bold blue]Evaluating model...", spinner="aesthetic"):
            inference.test(**kwargs)
        console.print("[bold green]Evaluation completed successfully![/bold green]")

    except DataNotFoundError as e:
        console.print(f"[bold red]Data not found:[/bold red] {e}")
        sys.exit(2)

    except ModelNotFoundError as e:
        console.print(f"[bold red]Model not found:[/bold red] {e}")
        sys.exit(2)
    except ValueError as e:
        console.print(f"[bold red]Invalid configuration:[/bold red] {e}")
        sys.exit(3)
    except Exception as e:
        console.print(f"[bold red]Testing failed:[/bold red] {e}")
        console.print_exception()
        sys.exit(1)


def run_predict(**kwargs) -> None:
    """Execute the predict command."""
    try:
        text: str | list[str] | None = kwargs.get("text")
        if not text:
            raise ValidationError("No text provided for prediction.")

        if isinstance(text, str) and not text.strip():
            raise ValidationError("Empty text provided for prediction.")

        if isinstance(text, list) and len(text) == 0:
            raise ValidationError("Empty list provided for prediction.")

        with console.status("[bold blue]Classifying text...", spinner="aesthetic"):
            inference.predict(**kwargs)

        console.print("[bold green]Prediction completed successfully![/bold green]")

    except ValidationError as e:
        console.print(f"[bold red]Invalid input:[/bold red] {e}")
        console.print("[yellow]Please provide valid text for prediction.[/yellow]")
        sys.exit(3)
    except ModelNotFoundError as e:
        console.print(f"[bold red]Model not found:[/bold red] {e}")
        console.print(
            "[yellow]Make sure the model checkpoint exists or use a valid model name.[/yellow]"
        )
        sys.exit(2)
    except ValueError as e:
        console.print(f"[bold red]Invalid configuration:[/bold red] {e}")
        console.print(
            "[yellow]Check your prediction parameters and try again.[/yellow]"
        )
        sys.exit(3)
    except Exception as e:
        console.print(f"[bold red]Prediction failed:[/bold red] {e}")
        console.print_exception()
        sys.exit(1)


def main() -> None:
    """Main CLI entry point."""
    parser: ArgumentParser = create_parser()
    args: Namespace = parser.parse_args()

    try:
        if args.subcommand == "train":
            run_train(**args.train)
        elif args.subcommand == "test":
            run_test(**args.test)
        elif args.subcommand == "predict":
            run_predict(**args.predict)
        else:
            parser.print_help()
            sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]Unexpected error:[/bold red] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
