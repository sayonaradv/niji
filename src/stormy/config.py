from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, BeforeValidator, Field


def validate_cache_dir(value: str | Path) -> str:
    """Convert Path objects to strings, keep strings as-is.

    Args:
        value: A string or Path object representing a cache directory.

    Returns:
        A string representation of the cache directory path.
    """
    if isinstance(value, Path):
        return str(value)
    return value


# Create the custom type
CacheDir = Annotated[str, BeforeValidator(validate_cache_dir)]


class DataModuleConfig(BaseModel):
    """Configuration for AutoTokenizerDataModule.

    Validates parameters for dataset loading, tokenization, and data processing
    in sequence classification tasks.
    """

    dataset_name: str = Field(
        description="Name of the Hugging Face dataset to load",
        examples=["imdb", "ag_news", "mat55555/jigsaw_toxic_comment"],
    )
    model_name: str = Field(
        description="Name of the pretrained Hugging Face model to use",
        examples=[
            "bert-base-uncased",
            "distilbert-base-uncased",
            "prajjwal1/bert-tiny",
        ],
    )
    train_split: str = Field(
        description="Name of the dataset split to use for training",
        examples=["train", "train[:80%]"],
    )
    test_split: str = Field(
        description="Name of the dataset split to use for testing",
        examples=["test", "validation", "train[80%:]"],
    )
    text_column: str = Field(
        description="Name of the column containing the input text data",
        examples=["text", "sentence", "comment_text"],
    )
    label_columns: list[str] = Field(
        min_length=1,
        description="List of column names containing classification labels. Must contain at least one column.",
        examples=[
            ["label"],
            ["toxic", "severe_toxic", "obscene"],
            ["positive", "negative"],
        ],
    )
    loader_columns: list[str] = Field(
        min_length=1,
        description="List of dataset columns to include in DataLoaders.",
    )
    max_token_len: int = Field(
        gt=0,
        description="Maximum number of tokens per input sequence. Longer sequences will be truncated.",
        examples=[128, 256, 512],
    )
    val_size: float = Field(
        gt=0,
        lt=1,
        description="Proportion of training data to use for validation (between 0 and 1)",
        examples=[0.1, 0.2, 0.3],
    )
    batch_size: int = Field(
        gt=0,
        description="Batch size for training and evaluation DataLoaders",
        examples=[16, 32, 64, 128],
    )
    cache_dir: CacheDir = Field(
        description="Directory path where datasets and tokenizer files will be cached",
        examples=["./data", "/tmp/cache", "~/.cache/huggingface"],
    )

    model_config = {
        "validate_assignment": True,  # Validate on attribute assignment
    }


class ModuleConfig(BaseModel):
    """Configuration for SequenceClassificationModule.

    Validates parameters for model initialization, training, and optimization
    in sequence classification tasks.
    """

    model_name: str = Field(
        description="Name of the pretrained Hugging Face model to use for sequence classification",
        examples=[
            "bert-base-uncased",
            "distilbert-base-uncased",
            "roberta-base",
            "prajjwal1/bert-tiny",
        ],
    )
    num_labels: int = Field(
        gt=0,
        description="Number of target labels in the classification task. Must be a positive integer.",
        examples=[2, 6, 10],
    )
    learning_rate: float = Field(
        gt=0,
        description="Learning rate for the Adam optimizer. Should be a small positive value.",
        examples=[1e-5, 3e-5, 5e-5, 1e-4],
    )

    model_config = {
        "validate_assignment": True,
    }
