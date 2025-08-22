import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

JIGSAW_LABELS: list[str] = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]
