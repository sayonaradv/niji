from jsonargparse import auto_cli

from ruffle.training import train


def cli_main() -> None:
    auto_cli(train)


if __name__ == "__main__":
    cli_main()
