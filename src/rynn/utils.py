from transformers import AutoModelForSequenceClassification, AutoTokenizer

from rynn.types import ModelTokenizerPair


def get_model_and_tokenizer(
    model_name: str,
    num_labels: int,
    cache_dir: str | None,
) -> ModelTokenizerPair:
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        cache_dir=cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, cache_dir=cache_dir, use_fast=True
    )
    return model, tokenizer
