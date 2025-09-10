from typing import Any

from transformers import AutoModelForSequenceClassification, AutoTokenizer


def get_model_and_tokenizer(
    model_name: str,
    num_labels: int,
    cache_dir: str | None,
) -> tuple[Any, Any]:
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        cache_dir=cache_dir,
        problem_type="multi_label_classification",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, cache_dir=cache_dir, use_fast=True
    )
    return model, tokenizer
