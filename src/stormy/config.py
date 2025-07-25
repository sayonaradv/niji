from pathlib import Path

from pydantic import BaseModel, Field, field_validator


class DataModuleConfig(BaseModel):
    dataset_name: str = Field(
        description="Name of the Hugging Face dataset to load (e.g., 'imdb', 'ag_news')"
    )
    model_name: str = Field(
        description="Name of the pretrained Hugging Face model to use (e.g., 'bert-base-uncased')"
    )
    train_split: str = Field(description="Name of the split to use for training")
    test_split: str = Field(description="Name of the split to use for testing")
    text_column: str = Field(
        description="Name of the column in the dataset that contains input text"
    )
    label_columns: list[str] = Field(
        min_length=1,
        description="List of column names containing the classification labels (must contain at least one)",
    )
    loader_columns: list[str] | None = Field(
        default_factory=lambda: ["input_ids", "attention_mask", "labels"],
        min_length=1,
        description="List of dataset columns to be in the dataloaders",
        validate_default=True,
    )
    max_token_len: int = Field(
        default=128,
        gt=0,
        description="Maximum number of tokens per input sequence (must be positive)",
        validate_default=True,
    )
    val_size: float = Field(
        default=0.2,
        gt=0,
        lt=1,
        description="Proportion of training data to use for validation (must be between 0 and 1)",
        validate_default=True,
    )
    batch_size: int = Field(
        default=32,
        gt=0,
        description="Batch size to use for training and evaluation (must be positive)",
        validate_default=True,
    )
    cache_dir: str | Path = Field(
        default=Path("./data"),
        description="Directory path to cache the dataset and tokenizer files",
        validate_default=True,
    )

    @field_validator("cache_dir")
    @classmethod
    def convert_path_to_str(cls, v):
        """Convert Path objects to strings for consistency."""
        return str(v) if isinstance(v, Path) else v

    model_config = {
        "arbitrary_types_allowed": True,  # Allow Path objects
        "validate_assignment": True,  # Validate on attribute assignment
    }
