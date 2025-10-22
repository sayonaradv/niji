import os

import transformers


def env_vars() -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def logging() -> None:
    transformers.logging.set_verbosity_error()
