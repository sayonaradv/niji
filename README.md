<!-- LOGO -->
<h1>
<p align="center">
  <img src="/images/stormy-high-resolution-logo.png" alt="stormy logo" width="128">
</h1>
  <p align="center">
    AI-powered Discord bot for automated content moderation and toxicity detection.
    <br />
    <a href="#about">About</a>
    ·
    <a href="#features">Features</a>
    ·
    <a href="#installation">Installation</a>
    ·
    <a href="#development">Development</a>
  </p>
</p>

## About

stormy (short for "no more") is a powerful Discord bot that leverages machine learning to automatically detect and moderate toxic content in your server. Built with modern AI technologies and best practices, stormy helps maintain a healthy and positive community environment by identifying and handling inappropriate messages in real-time.

Unlike other moderation bots that rely on simple keyword matching, stormy uses advanced natural language processing to understand context and nuance, providing more accurate and fair moderation decisions.

## Features

- 🤖 **AI-Powered Detection**: Utilizes fine-tuned BERT models for accurate toxicity detection
- ⚡ **Real-time Moderation**: Instant message analysis and automated actions
- 🛡️ **Customizable Rules**: Configure moderation thresholds and actions
- 📊 **Analytics Dashboard**: Track moderation metrics and server health
- 🔌 **API Access**: Use the toxicity detection API in your own applications


## Development Status

| Phase | Feature | Status |
|:-----:|---------|:------:|
| 1 | Data Handling & Preprocessing | ✅ |
| 2 | Model Building & Training | ⚠️ |
| 3 | API Deployment | ❌ |
| 4 | Discord Bot Integration | ❌ |
| 5 | Testing & Deployment | ❌ |

### Current Progress

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

#### Discord Bot Integration ❌
- Bot command structure
- Moderation workflow
- User interaction handling

#### Testing & Deployment ❌
- Docker containerization
- Cloud deployment setup
- Monitoring and logging

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/stormy.git
cd stormy
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. Run the bot:
```bash
python main.py
```

## Development

### Prerequisites
- Python 3.8+
- Discord Bot Token
- GPU (recommended for model training)

### Setting Up Development Environment

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

3. Run tests:
```bash
pytest
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Discord.py](https://discordpy.readthedocs.io/) for the Discord bot framework
- [Hugging Face](https://huggingface.co/) for the BERT model
- [FastAPI](https://fastapi.tiangolo.com/) for the API framework
