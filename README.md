<!-- LOGO -->
<h1>
<p align="center">
  <img src="images/icon/web/icon-512.png" alt="stormy logo" width="128">
  <br>Stormy
</h1>
  <p align="center">
    Open-source Python library for detecting hateful or offensive language using state-of-the-art machine learning models.
    <br />
    <a href="#about">About</a>
    ·
    <a href="#features">Features</a>
    ·
    <a href="#installation">Installation</a>
    ·
    <a href="#development">Development</a>
    ·
    <a href="#roadmap-and-status">Roadmap</a>
    ·
    <a href="#planned-projects">Planned Projects</a>
  </p>
</p>

---

## About

**Stormy** is an open-source Python library that detects hateful or offensive language. Stormy models are trained to predict toxic comments on the Jigsaw Toxic Comment Classification challenge. Unlike traditional keyword-based approaches, Stormy leverages advanced natural language processing to understand context and nuance, providing more accurate and fair toxicity detection.

**Stormy** is designed to help maintain healthy and positive online environments by identifying and handling inappropriate messages in real-time, with customizable rules and actionable analytics.

---

## Features

- 🤖 **AI-Powered Detection**: Fine-tuned BERT models for accurate toxicity detection.
- ⚡ **Real-time Analysis**: Instant text analysis for toxicity and offensive language.
- 🛡️ **Customizable Thresholds**: Configure detection thresholds and actions.
- 📊 **Analytics Tools**: Track moderation metrics and analyze text datasets.
- 🔌 **API-Ready**: Easily integrate toxicity detection into your own Python applications.

---

## Roadmap and Status

The high-level plan for Stormy, in order:

|  #  | Step                        | Status |
| :-: | --------------------------- | :----: |
|  1  | Data Handling & Preprocessing |   ✅   |
|  2  | Model Building & Training     |   ⚠️   |
|  3  | API Deployment                |   ❌   |
|  4  | Library Usage Examples        |   ❌   |
|  5  | Testing & Deployment          |   ❌   |

### Details

#### Data Handling & Preprocessing ✅
- Dataset acquisition and storage
- Text cleaning and preprocessing
- Train-test split implementation

#### Model Building & Training ⚠️
- BERT model fine-tuning in progress
- Performance evaluation metrics defined
- Model checkpointing system implemented

#### API Deployment ❌
- FastAPI endpoint design
- Model serving infrastructure
- API documentation

#### Library Usage Examples ❌
- Example scripts and notebooks
- Documentation for integration

#### Testing & Deployment ❌
- Packaging and distribution
- Cloud deployment setup
- Monitoring and logging

---

## Planned Projects

### Stormy-Bot (Future Work)

We plan to build **Stormy-Bot**, a Discord bot that leverages Stormy models to detect offensive messages sent in Discord servers. This will be a separate project and repository. Stay tuned for updates!

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/stormy.git
   cd stormy
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables (if needed):**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Run example usage:**
   ```bash
   python -m src.stormy.module  # Or see example scripts in the documentation
   ```

---

## Development

### Prerequisites

- Python 3.8+
- GPU (recommended for model training)

### Setting Up Development Environment

1. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install development dependencies:**
   ```bash
   pip install -r requirements-dev.txt
   ```

3. **Run tests:**
   ```bash
   pytest
   ```

---

## Contributing

We welcome contributions! To get started:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [Hugging Face](https://huggingface.co/) for the BERT model
- [FastAPI](https://fastapi.tiangolo.com/) for the API framework

---

**stormy** is in active development. Stay tuned for updates and new features!

---
