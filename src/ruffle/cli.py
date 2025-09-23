import sys
from typing import Literal

from jsonargparse import ArgumentParser

from ruffle import inference, training


def print_welcome_msg() -> None:
    """Print ASCII art welcome message."""
    msg = """
    █████████                 ░████     ░████ ░██
    ░██     ░██               ░██       ░██    ░██
    ░██     ░██ ░██    ░██ ░████████ ░████████ ░██  ░███████
    ░█████████  ░██    ░██    ░██       ░██    ░██ ░██    ░██
    ░██   ░██   ░██    ░██    ░██       ░██    ░██ ░█████████
    ░██    ░██  ░██   ░███    ░██       ░██    ░██ ░██
    ░██     ░██  ░█████░██    ░██       ░██    ░██  ░███████
    """
    print(msg)


def create_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="ruffle",
        description="Ruffle: Multi-label toxicity classification with transformer models.",
    )

    subcommands = parser.add_subcommands()
    subcommands.add_subcommand("train", _create_sub_parser("train"))
    subcommands.add_subcommand("test", _create_sub_parser("test"))
    subcommands.add_subcommand("predict", _create_sub_parser("predict"))

    return parser


def _create_sub_parser(command: Literal["train", "test", "predict"]) -> ArgumentParser:
    parser_func_maps = {
        "train": training.train,
        "test": inference.test,
        "predict": inference.predict,
    }

    parser = ArgumentParser()
    parser.add_function_arguments(parser_func_maps[command])
    return parser


def main():
    print_welcome_msg()
    parser = create_parser()
    args = parser.parse_args()

    if args.subcommand == "train":
        training.train(**vars(args.train))
    elif args.subcommand == "test":
        inference.test(**vars(args.test))
    elif args.subcommand == "predict":
        inference.predict(**vars(args.predict))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
