# Deepfake Audio Detection

[![CI](https://github.com/vishalkonduru/deepfake-audio-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/vishalkonduru/deepfake-audio-detection/actions/workflows/ci.yml)

A machine-learning pipeline that classifies audio files as **REAL** or **FAKE (deepfake)** using MFCC, spectral and chroma features fed into an SVM classifier.

---

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Training](#training)
- [API](#api)
- [Docker](#docker)
- [Configuration](#configuration)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

---

## Features

| Feature | Detail |
|---|---|
| Audio features | 40 MFCCs (mean + std), spectral centroid/bandwidth/rolloff, 12-bin chroma |
| Classifier | SVM with RBF kernel + StandardScaler pipeline |
| Hyperparameter tuning | GridSearchCV (`tune_model.py`) |
| REST API | Flask with `/health`, `/predict`, `/info` endpoints |
| CLI | `predict.py` for batch file classification |
| Evaluation | `evaluate.py` with JSON report + confusion matrix PNG |
| Containerisation | Multi-stage Dockerfile + docker-compose |
| CI | GitHub Actions: lint (flake8, black, isort) + tests (py3.10, py3.11) |

---

## Project Structure

```
.
├── app.py                 # Flask REST API
├── config.py              # Centralised configuration (env-overridable)
├── extract_features.py    # Feature extraction (MFCC, spectral, chroma)
├── train_model.py         # Training script
├── tune_model.py          # Hyperparameter search
├── evaluate.py            # Post-training evaluation
├── predict.py             # CLI predictor
├── tests/
│   ├── test_features.py
│   └── test_api.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── pyproject.toml
```

---

## Quick Start

```bash
pip install -r requirements.txt

# Place audio files in data/real/ and data/fake/
python extract_features.py
python train_model.py
python predict.py path/to/audio.wav --verbose
```

---

## Training

```bash
python extract_features.py
python train_model.py
python tune_model.py      # optional GridSearch
python evaluate.py --output-dir evaluation/
```

---

## API

```bash
python app.py  # starts on port 8000

curl -X POST http://localhost:8000/predict -F "file=@audio.wav"
curl http://localhost:8000/health
curl http://localhost:8000/info
```

Response:
```json
{"label": "FAKE", "confidence": 0.92, "probabilities": {"real": 0.08, "fake": 0.92}}
```

---

## Docker

```bash
docker compose up --build
```

---

## Configuration

All settings in `config.py` are overridable via environment variables:

| Variable | Default | Description |
|---|---|---|
| `SAMPLE_RATE` | `16000` | Audio sample rate |
| `N_MFCC` | `40` | MFCC coefficients |
| `MODEL_OUT` | `model.joblib` | Saved model path |
| `API_PORT` | `8000` | Flask port |
| `MAX_FILE_SIZE_MB` | `25` | Max upload size |

---

## Testing

```bash
pip install pytest pytest-cov soundfile
pytest tests/ --cov=. --cov-report=term-missing
```

---

## License

MIT
