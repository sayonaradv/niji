from .exceptions import BlankiError, DataNotFoundError, ModelNotFoundError
from .setup import env_vars, logging

env_vars()
logging()


__version__ = "0.0.1a4"

__all__: list[str] = [
    "BlankiError",
    "DataNotFoundError",
    "ModelNotFoundError",
]


def main() -> None:
    print("Hello from blanki! ⛈️")
