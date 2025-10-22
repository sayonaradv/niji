# """
# Configuration management for Niji CLI.

# This module provides configuration file support with validation and default values.
# """

# import json
# from pathlib import Path
# from typing import Any

# import yaml
# from pydantic import BaseModel, Field
# from pydantic import ValidationError as PydanticValidationError


# class TrainingConfig(BaseModel):
#     """Configuration for training command."""

#     model_name: str = Field(..., description="Hugging Face model identifier")
#     data_dir: str = Field(
#         default="./data/jigsaw-toxic-comment-classification-challenge"
#     )
#     labels: list[str] | None = Field(default=None)
#     batch_size: int = Field(default=64, ge=1)
#     val_size: float = Field(default=0.2, ge=0.0, le=1.0)
#     max_token_len: int = Field(default=256, ge=1)
#     lr: float = Field(default=3e-5, gt=0)
#     warmup_start_lr: float = Field(default=1e-5, gt=0)
#     warmup_epochs: int = Field(default=5, ge=1)
#     max_epochs: int = Field(default=20, ge=1)
#     patience: int = Field(default=3, ge=1)
#     run_name: str | None = Field(default=None)
#     perf: bool = Field(default=False)
#     fast_dev_run: bool = Field(default=False)
#     cache_dir: str | None = Field(default="./data")
#     log_dir: str = Field(default="./runs")
#     seed: int = Field(default=18, ge=0)


# class TestConfig(BaseModel):
#     """Configuration for test command."""

#     model_name: str | None = Field(default=None)
#     ckpt_path: str | None = Field(default=None)
#     data_dir: str = Field(
#         default="./data/jigsaw-toxic-comment-classification-challenge"
#     )
#     batch_size: int = Field(default=64, ge=1)
#     perf: bool = Field(default=True)
#     run_name: str | None = Field(default=None)


# class PredictConfig(BaseModel):
#     """Configuration for predict command."""

#     text: str | list[str] = Field(..., description="Text(s) to classify")
#     model_name: str | None = Field(default=None)
#     ckpt_path: str | None = Field(default=None)
#     threshold: float = Field(default=0.5, ge=0.0, le=1.0)
#     device: str | None = Field(default="cpu")
#     verbose: bool = Field(default=True)
#     return_logits: bool = Field(default=False)


# class NijiConfig(BaseModel):
#     """Main configuration class for Niji CLI."""

#     training: TrainingConfig | None = Field(default=None)
#     test: TestConfig | None = Field(default=None)
#     predict: PredictConfig | None = Field(default=None)
#     verbose: bool = Field(default=False)
#     quiet: bool = Field(default=False)


# def load_config(config_path: str) -> NijiConfig:
#     """Load configuration from a YAML or JSON file.

#     Args:
#         config_path: Path to the configuration file

#     Returns:
#         NijiConfig: Loaded configuration object

#     Raises:
#         FileNotFoundError: If the config file doesn't exist
#         ValueError: If the config file format is invalid
#     """
#     config_file = Path(config_path)

#     if not config_file.exists():
#         raise FileNotFoundError(f"Configuration file not found: {config_path}")

#     try:
#         with open(config_file) as f:
#             if config_file.suffix.lower() in [".yaml", ".yml"]:
#                 data = yaml.safe_load(f)
#             elif config_file.suffix.lower() == ".json":
#                 data = json.load(f)
#             else:
#                 raise ValueError(
#                     f"Unsupported config file format: {config_file.suffix}"
#                 )

#         return NijiConfig(**data)

#     except PydanticValidationError as e:
#         raise ValueError(f"Invalid configuration: {e}") from e
#     except (yaml.YAMLError, json.JSONDecodeError) as e:
#         raise ValueError(f"Failed to parse configuration file: {e}") from e


# def save_config(config: NijiConfig, config_path: str, format: str = "yaml") -> None:
#     """Save configuration to a file.

#     Args:
#         config: Configuration object to save
#         config_path: Path where to save the configuration
#         format: Output format ("yaml" or "json")
#     """
#     config_file = Path(config_path)
#     config_file.parent.mkdir(parents=True, exist_ok=True)

#     data = config.model_dump(exclude_none=True)

#     with open(config_file, "w") as f:
#         if format.lower() == "yaml":
#             yaml.dump(data, f, default_flow_style=False, indent=2)
#         elif format.lower() == "json":
#             json.dump(data, f, indent=2)
#         else:
#             raise ValueError(f"Unsupported output format: {format}")


# def create_default_config(command: str) -> dict[str, Any]:
#     """Create a default configuration for a specific command.

#     Args:
#         command: Command name ("train", "test", or "predict")

#     Returns:
#         Dict containing default configuration values
#     """
#     defaults = {
#         "train": {
#             "training": {
#                 "model_name": "bert-base-uncased",
#                 "data_dir": "./data/jigsaw-toxic-comment-classification-challenge",
#                 "batch_size": 64,
#                 "val_size": 0.2,
#                 "max_token_len": 256,
#                 "lr": 3e-5,
#                 "warmup_start_lr": 1e-5,
#                 "warmup_epochs": 5,
#                 "max_epochs": 20,
#                 "patience": 3,
#                 "perf": False,
#                 "fast_dev_run": False,
#                 "cache_dir": "./data",
#                 "log_dir": "./runs",
#                 "seed": 18,
#             }
#         },
#         "test": {
#             "test": {
#                 "data_dir": "./data/jigsaw-toxic-comment-classification-challenge",
#                 "batch_size": 64,
#                 "perf": True,
#             }
#         },
#         "predict": {
#             "predict": {
#                 "text": "Sample text to classify",
#                 "threshold": 0.5,
#                 "device": "cpu",
#                 "verbose": True,
#                 "return_logits": False,
#             }
#         },
#     }

#     return defaults.get(command, {})
