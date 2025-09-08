# Ruffle

An extremely fast and accurate Python library for detecting toxic and harmful content in text using state-of-the-art transformer models.

---

## ✨ Highlights

- ⚡ **Lightning Fast**: Built on PyTorch Lightning for efficient training and inference
- 🧠 **Transformer-Powered**: Leverages BERT, DistilBERT, and other state-of-the-art models
- 🎯 **Multi-Label Detection**: Simultaneously detects `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, and `identity_hate`
- 🚀 **Production Ready**: Optimized inference pipeline with model compilation
- 🔧 **Flexible Configuration**: YAML-based configuration system for easy experimentation
- 📊 **Rich Monitoring**: Built-in progress tracking, logging, and model checkpointing

Ruffle provides both high-level prediction APIs for quick content moderation and comprehensive training tools for custom model development.

---

## 🚀 Installation

### Quick Installation

Install Ruffle using uv (recommended):

```bash
uv add ruffle
```

Or using pip:

```bash
pip install ruffle
```

### Development Installation

For training custom models or contributing to development:

```bash
# Clone the repository
git clone https://github.com/zuzo-sh/ruffle.git
cd ruffle

# Install with development dependencies using uv
uv sync

# Or using pip
pip install -e ".[dev]"
```

---

## 📖 Usage

### API

Get started with toxicity detection in just a few lines:

```python
from ruffle import Ruffle

# Load a pre-trained model
ruffle = Ruffle(model_name="bert-tiny")

# Detect toxicity in text
result = ruffle.predict("This is a sample comment to analyze")
print(result)

# Process multiple texts efficiently  
texts = [
    "This is a normal comment",
    "Another piece of text to check", 
    "Batch processing is supported"
]
results = ruffle.predict(texts)
```

### CLI

Or run Ruffle directly from the command line:

```bash
# Classify single text
uv run python -m ruffle.predictor --texts "Hello world" --threshold 0.7

# Classify multiple texts  
uv run python -m ruffle.predictor --texts '["Text 1", "Text 2"]' --model_name bert-tiny

# Use custom checkpoint
uv run python -m ruffle.predictor --texts "Sample text" --checkpoint_path model.ckpt
```

See the [documentation](https://ruffle.readthedocs.io) for comprehensive guides and API reference.

---

## 📊 Detection Categories

Ruffle detects six categories of toxicity based on the Jigsaw Toxic Comment Classification dataset:

| Category | Description |
|----------|-------------|
| `toxic` | General toxicity and harmful content |
| `severe_toxic` | Severely toxic content requiring immediate action |
| `obscene` | Obscene language and explicit content |
| `threat` | Threatening language and intimidation |
| `insult` | Insulting and demeaning content |
| `identity_hate` | Identity-based hate speech and discrimination |

Example output:
```python
{
    "This is offensive content": {
        "toxic": 0.89,
        "severe_toxic": 0.23,
        "obscene": 0.67,
        "threat": 0.12,
        "insult": 0.78,
        "identity_hate": 0.34
    }
}
```

---

### Train your own models

After you've downloaded the [Jigsaw](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge)  dataset, you can finetune any model available on https://huggingface.co/models:

```bash
uv run -m ruffle.trainer <your_model_name> --data_dir=<jigsaw_dir>
```

---

## 🔬 Development

### Prerequisites

- Python 3.13+
- CUDA-compatible GPU (recommended for training)

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/zuzo-sh/ruffle.git
cd ruffle

# Install with development dependencies
uv sync
```

---

## 📚 Documentation

- **[Getting Started Guide](https://ruffle.readthedocs.io/getting-started/)** - Basic usage and installation
- **[Training Guide](https://ruffle.readthedocs.io/training/)** - Custom model training and fine-tuning  
- **[API Reference](https://ruffle.readthedocs.io/api/)** - Complete API documentation
- **[Configuration Guide](https://ruffle.readthedocs.io/configuration/)** - YAML configuration options
- **[Production Deployment](https://ruffle.readthedocs.io/production/)** - Performance optimization and scaling

---

### Code Style

This project uses:
- **Ruff** for formatting and linting (configuration in `pyproject.toml`)
- **ty** for type checking
- **pytest** for testing
- **Conventional Commits** for commit messages

See our [Contributing Guide](CONTRIBUTING.md) for detailed guidelines.

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **[Hugging Face Transformers](https://huggingface.co/transformers/)** - For the transformer model implementations
- **[PyTorch Lightning](https://lightning.ai/)** - For the training framework and utilities
- **[Jigsaw/Conversation AI](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)** - For the toxicity classification dataset

---

## 📈 Benchmarks

*Benchmarks run on Macbook Pro M1 Pro

---

**Ruffle** - Professional toxicity detection for safer digital spaces.
