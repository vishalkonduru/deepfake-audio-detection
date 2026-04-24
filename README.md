# Deepfake Audio Detection for Phishing Calls

This project detects AI‑generated (deepfake) speech in audio clips to help flag phishing calls and voice‑cloning fraud attempts. It uses MFCC audio features and a classical machine learning model (SVM), exposed through a simple REST API and optionally containerized with Docker.[web:74][web:76][web:87]

---

## Problem Overview

Attackers increasingly use AI‑generated voices to impersonate trusted people (managers, family members, bank staff) over phone calls, tricking victims into sharing sensitive information or transferring money.[web:40][web:43]  
The goal of this project is to analyze voice audio and estimate whether it is **REAL** (human) or **FAKE** (synthetic / deepfake), so that suspicious calls can be flagged in real time or in near‑real‑time.[web:39][web:44]

---

## Approach

### Dataset

For this prototype, we use the Kaggle dataset:

- **Real vs Fake Human Voice – Deepfake Audio Dataset**[web:63]  
  - Contains recordings from different speakers (male/female, UK/USA).
  - Each speaker folder has:
    - `original.*` → real human voice.
    - `synthetic_*.mp3` → AI‑generated versions (deepfake audio).

These are mapped into:

- `data/real/`  → originals (label 0).
- `data/fake/` → synthetics (label 1).

### Features

We extract **MFCC (Mel‑Frequency Cepstral Coefficients)** for each audio file using `librosa`:[web:74][web:76]

- Resample to 16 kHz mono.
- Compute \( n\_mfcc = 40 \) MFCCs over time.
- Take **mean** and **standard deviation** per coefficient.
- Concatenate `[mean, std]` → 80‑dimensional feature vector per clip.

This captures the timbral / spectral characteristics that differ between human and synthetic speech.[web:74]

### Model

We train a **Support Vector Machine (SVM)** classifier with an RBF kernel using `scikit‑learn`:[web:76]

- Input: 80‑dimensional MFCC feature vector.
- Output: Binary label:
  - 0 → REAL.
  - 1 → FAKE.
- We also enable `probability=True` to obtain soft probabilities for REAL vs FAKE.

The training pipeline:

1. Load MFCC feature matrix (`features.npy`) and labels (`labels.npy`).
2. Split into train/test sets (e.g., 80/20 stratified split).
3. Train SVM on the training set.
4. Evaluate on the test set (accuracy, precision, recall, F1).
5. Save the trained model as `model.joblib`.

This MFCC+SVM setup is a standard baseline in audio deepfake detection research and works well on curated datasets.[web:74][web:76]

---

## Project Structure

```text
deepfake-audio-detection/
├─ app.py                    # Flask API exposing /health and /predict
├─ extract_features.py       # Script to extract MFCC features and save features.npy, labels.npy
├─ train_model.py            # Script to train SVM model and save model.joblib
├─ predict.py                # CLI script to classify a single audio file
├─ Dockerfile                # Docker image for production-style serving (Gunicorn)
├─ requirements-docker.txt   # Minimal dependencies used inside Docker
├─ requirements.txt          # Dev dependencies (local venv)
├─ .gitignore
├─ README.md
├─ data/
│  ├─ real/                  # Real/original audio (not committed)
│  └─ fake/                  # Fake/synthetic audio (not committed)
└─ raw_data/                 # Raw Kaggle dataset extraction (not committed)
```

> Note: `data/`, `raw_data/`, and model artifacts are excluded from Git via `.gitignore`.

---

## Setup (Local, via WSL/Linux)

### 1. Clone the repository

```bash
git clone https://github.com/vishalkonduru/deepfake-audio-detection.git
cd deepfake-audio-detection
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Dataset Preparation

Download the Kaggle dataset (or use your own):

```bash
mkdir -p raw_data
cd raw_data

# Requires Kaggle CLI configured with API key
kaggle datasets download -d unidpro/real-vs-fake-human-voice-deepfake-audio
unzip real-vs-fake-human-voice-deepfake-audio.zip
cd ..
```

Organize data into `data/real` and `data/fake`:

```bash
mkdir -p data/real data/fake

# Copy originals (real)
cp raw_data/UK/female/*/original.* data/real/
cp raw_data/UK/male/*/original.*   data/real/
cp raw_data/USA/female/*/original.* data/real/
cp raw_data/USA/male/*/original.*   data/real/

# Copy synthetics (fake)
cp raw_data/UK/female/*/synthetic_*.* data/fake/
cp raw_data/UK/male/*/synthetic_*.*   data/fake/
cp raw_data/USA/female/*/synthetic_*.* data/fake/
cp raw_data/USA/male/*/synthetic_*.*   data/fake/
```

---

## Training Pipeline

### 1. Extract features

```bash
source .venv/bin/activate
python extract_features.py
```

This will:

- Read all audio files under `data/real` and `data/fake`.
- Compute MFCC-based features.
- Save:
  - `features.npy`
  - `labels.npy`

### 2. Train the SVM model

```bash
python train_model.py
```

This will:

- Load `features.npy` and `labels.npy`.
- Split into train/test sets.
- Train an SVM classifier.
- Print classification metrics.
- Save `model.joblib`.

---

## CLI Inference

To classify a single audio file from the command line:

```bash
python predict.py path/to/audio_file.wav
```

Example with dataset samples:

```bash
# Real sample
python predict.py data/real/$(ls data/real | head -n 1)

# Fake sample
python predict.py data/fake/$(ls data/fake | head -n 1)
```

Output example:

```text
Prediction: REAL
Probabilities [REAL, FAKE]: [0.41 0.59]
```

---

## REST API (Flask)

The project includes a Flask app that exposes two endpoints: `/health` and `/predict`.

### Run the API (development)

```bash
source .venv/bin/activate
python app.py
```

By default, the server runs on:

- `http://0.0.0.0:8000`

### Endpoints

#### 1. Health check

```bash
curl http://localhost:8000/health
```

Response:

```json
{"status": "ok"}
```

#### 2. Predict (upload audio file)

```bash
curl -X POST \
  -F "file=@path/to/audio.wav" \
  http://localhost:8000/predict
```

Example with dataset samples:

```bash
# Fake
curl -X POST \
  -F "file=@data/fake/$(ls data/fake | head -n 1)" \
  http://localhost:8000/predict

# Real
curl -X POST \
  -F "file=@data/real/$(ls data/real | head -n 1)" \
  http://localhost:8000/predict
```

Sample response:

```json
{
  "label": "FAKE",
  "probabilities": {
    "real": 0.42,
    "fake": 0.58
  }
}
```

---

## Docker Deployment

For more production-like serving, use the provided Dockerfile and minimal requirements.

### 1. Build the Docker image

```bash
cd deepfake-audio-detection

docker build -t deepfake-audio-api .
```

This uses:

- Base image: `python:3.11-slim`.
- Installs `ffmpeg`.
- Installs only the minimal dependencies from `requirements-docker.txt`.
- Runs the app with Gunicorn on port 8000. [web:83][web:86]

### 2. Run the container

```bash
docker run -d -p 8000:8000 --name deepfake-audio-api deepfake-audio-api
```

Check it:

```bash
docker ps
```

Test:

```bash
curl http://localhost:8000/health
curl -X POST \
  -F "file=@data/fake/$(ls data/fake | head -n 1)" \
  http://localhost:8000/predict
```

---

## How This Applies to Phishing Calls

In a real system integrated with telephony or VoIP:[web:39][web:44]

1. A call audio stream is chunked into short segments (e.g., 2–3 seconds).
2. Each segment is passed through this model:
   - Extract MFCCs.
   - Get REAL/FAKE probabilities.
3. The system aggregates predictions over time:
   - If many segments are flagged as “FAKE” with high probability, the call is labeled suspicious.
4. A UI or backend service can then:
   - Warn the user.
   - Ask for additional challenge‑response (e.g., repeat random phrases).
   - Escalate or log the call for further analysis.

This project demonstrates the **core detection engine** that could be embedded into such a pipeline.

---

## Limitations and Future Work

- Current model is trained on a relatively small curated dataset and may not generalize to all real‑world deepfake engines or noisy phone environments. [web:41][web:45]
- MFCC+SVM is lightweight but less powerful than larger CNN/transformer models; adding more robust features (CQCC, spectrogram CNNs) and training on diverse datasets (e.g., ASVspoof) would improve robustness.[web:41][web:74]
- Real‑time streaming integration and more advanced challenge‑response strategies can further harden against sophisticated attacks.[web:44]

---

## License

- Code: Your chosen license (e.g., MIT).  
- Dataset: Follow the original Kaggle “Real vs Fake Human Voice – Deepfake Audio Dataset” license (CC BY‑NC‑ND 4.0). [web:63]

---
