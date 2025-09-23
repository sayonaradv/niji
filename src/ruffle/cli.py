from jsonargparse import auto_cli

from ruffle import inference, training


def cli_main() -> None:
    commands = [training.train, inference.test, inference.predict]
    auto_cli(commands)


if __name__ == "__main__":
    cli_main()
