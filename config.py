"""Central configuration for deepfake-audio-detection."""

import os

# ── Audio ──────────────────────────────────────────────────────────────────────
SAMPLE_RATE: int = int(os.getenv("SAMPLE_RATE", 16000))
N_MFCC: int = int(os.getenv("N_MFCC", 40))
AUDIO_EXTS: tuple = (".wav", ".flac", ".mp3", ".m4a")

# ── Paths ──────────────────────────────────────────────────────────────────────
REAL_DIR: str = os.getenv("REAL_DIR", "data/real")
FAKE_DIR: str = os.getenv("FAKE_DIR", "data/fake")
FEATURES_OUT: str = os.getenv("FEATURES_OUT", "features.npy")
LABELS_OUT: str = os.getenv("LABELS_OUT", "labels.npy")
MODEL_OUT: str = os.getenv("MODEL_OUT", "model.joblib")
SCALER_OUT: str = os.getenv("SCALER_OUT", "scaler.joblib")

# ── Model ──────────────────────────────────────────────────────────────────────
TEST_SIZE: float = float(os.getenv("TEST_SIZE", 0.2))
RANDOM_STATE: int = int(os.getenv("RANDOM_STATE", 42))
SVM_KERNEL: str = os.getenv("SVM_KERNEL", "rbf")
SVM_C: float = float(os.getenv("SVM_C", 1.0))
SVM_GAMMA: str = os.getenv("SVM_GAMMA", "scale")

# ── API ────────────────────────────────────────────────────────────────────────
API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", 8000))
MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", 25))
