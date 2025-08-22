# Blanket ⛈️

**Professional-grade toxicity detection powered by transformer models**

Blanket is a modern Python library for detecting toxic and harmful content in text using state-of-the-art transformer models. Built on PyTorch Lightning and Hugging Face Transformers, it provides robust, scalable solutions for content moderation and safety applications.

---

## Features

- 🧠 **Transformer-Based Detection**: Leverages BERT, DistilBERT, and other transformer models for superior accuracy
- ⚡ **Lightning Fast Training**: Built on PyTorch Lightning for efficient, scalable model training
- 🎯 **Multi-Label Classification**: Detects multiple toxicity categories simultaneously (toxic, severe_toxic, obscene, threat, insult, identity_hate)
- 🔧 **Flexible Configuration**: YAML-based configuration system for easy experimentation
- 📊 **Rich Monitoring**: Built-in logging, progress tracking, and model checkpointing
- 🚀 **Production Ready**: Optimized inference with model compilation and efficient tokenization

---

## Usage

### Making Quick Predictions

For users who simply want to use pre-trained models for toxicity detection:

#### Installation

```bash
# Install using UV (recommended)
uv add blanket

# Or using pip
pip install blanket
```

#### Basic Prediction

```python
from blanket import Blanket

# Load a pre-trained model
detector = Blanket(checkpoint="path/to/pretrained/model.ckpt")

# Detect toxicity in single text
detector.predict("This is a sample text to analyze")

# Process multiple texts at once
texts = [
    "This is a normal comment",
    "Another piece of text to check",
    "Multiple texts can be processed together"
]
detector.predict(texts)
```

### Training / Fine-tuning

For users who want to train their own models or fine-tune existing ones:

#### Setup for Training

```bash
# Clone the repository
git clone <your-repo-url>
cd blanket

# Install with development dependencies using UV (recommended)
uv sync

# Or using pip
pip install -e ".[dev]"
```

#### Training Commands

```bash
# Quick training with BERT-tiny for testing/development
uv run train fit --config configs/jigsaw_test.yaml

# Full training with DistilBERT for production
uv run train fit --config configs/jigsaw_full.yaml

# Custom training with your own config
uv run train fit --config path/to/your/config.yaml
```

#### Making Predictions with Custom Models

```python
from blanket import Blanket

# Load your trained model
detector = Blanket(checkpoint="lightning_logs/version_0/checkpoints/best.ckpt")

# Use for inference
detector.predict("Text to analyze with your custom model")
```

---

## Architecture

Blanket is built with a modular architecture:

- **Models** (`blanket.models`): ToxicityClassifier built on transformer architectures
- **Data** (`blanket.dataloaders`): JigsawDataModule for efficient data loading and preprocessing  
- **Training** (`blanket.trainer`): Lightning CLI with professional callbacks and monitoring
- **Inference** (`blanket.predictor`): Optimized prediction pipeline for production use

### Supported Models

- DistilBERT (recommended for production)
- BERT variants (including BERT-tiny for development)
- Any Hugging Face transformer model compatible with sequence classification

---

## Configuration

Blanket uses YAML configuration files for reproducible experiments:

```yaml
model:
  model_name: distilbert/distilbert-base-uncased
  max_token_len: 256
  cache_dir: data

data:
  dataset_name: mat55555/jigsaw_toxic_comment
  batch_size: 64
  val_size: 0.2

trainer:
  max_epochs: 20
  deterministic: true
```

See `configs/` directory for complete examples.

---

## Development

### Prerequisites

- Python 3.13+
- CUDA-compatible GPU (recommended for training)

### Setup Development Environment

```bash
# Clone and install with development dependencies
git clone <your-repo-url>
cd blanket
pip install -e ".[dev]"

# Run tests
pytest

# Code formatting and linting
ruff check .
ruff format .
```

### Project Structure

```
src/blanket/
├── __init__.py          # Main exports
├── config.py           # Configuration constants
├── dataloaders.py      # Data loading and preprocessing
├── models.py           # ToxicityClassifier implementation
├── predictor.py        # Inference pipeline
├── trainer.py          # Training CLI and callbacks
├── schedulers.py       # Learning rate scheduling
└── utils.py            # Model and tokenizer utilities
```

---

## Training Details

### Dataset

Blanket trains on the Jigsaw Toxic Comment Classification dataset, which includes six toxicity categories:

- `toxic`: General toxicity
- `severe_toxic`: Severely toxic content
- `obscene`: Obscene language
- `threat`: Threatening language
- `insult`: Insulting content
- `identity_hate`: Identity-based hate speech

### Model Performance

- **Loss Function**: Binary cross-entropy for multi-label classification
- **Metrics**: Multi-label accuracy across all toxicity categories
- **Optimization**: Adam optimizer with cosine annealing and linear warmup
- **Regularization**: Model compilation for improved training speed

---

## API Reference

### Core Classes

#### `ToxicityClassifier`
PyTorch Lightning module for toxicity detection.

```python
classifier = ToxicityClassifier(
    model_name="distilbert/distilbert-base-uncased",
    max_token_len=256,
    lr=3e-5
)
```

#### `JigsawDataModule`
Lightning DataModule for the Jigsaw dataset.

```python
datamodule = JigsawDataModule(
    batch_size=64,
    val_size=0.2
)
```

#### `Blanket`
High-level inference interface.

```python
detector = Blanket(
    checkpoint="model.ckpt",
    threshold=0.5,
    device="cpu"
)
```

---

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and add tests
4. Run the test suite (`pytest`)
5. Format your code (`ruff format .`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Code Style

This project uses Ruff for formatting and linting. The configuration is in `pyproject.toml`.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers/) for the transformer implementations
- [PyTorch Lightning](https://lightning.ai/) for the training framework
- [Jigsaw/Conversation AI](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) for the toxicity dataset

---

**Blanket** - Professional toxicity detection for safer digital spaces.