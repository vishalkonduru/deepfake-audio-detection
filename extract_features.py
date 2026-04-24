"""Feature extraction utilities for deepfake audio detection."""

import os
import numpy as np
import librosa
from tqdm import tqdm
from typing import Tuple

import config


def extract_mfcc(file_path: str, n_mfcc: int = config.N_MFCC) -> np.ndarray:
    """Extract mean + std of MFCCs from an audio file."""
    y, sr = librosa.load(file_path, sr=config.SAMPLE_RATE, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.concatenate([np.mean(mfcc, axis=1), np.std(mfcc, axis=1)])


def extract_spectral(file_path: str) -> np.ndarray:
    """Extract spectral centroid, bandwidth and rolloff statistics."""
    y, sr = librosa.load(file_path, sr=config.SAMPLE_RATE, mono=True)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    feats = []
    for f in (centroid, bandwidth, rolloff):
        feats += [np.mean(f), np.std(f)]
    return np.array(feats)


def extract_chroma(file_path: str) -> np.ndarray:
    """Extract chroma feature statistics."""
    y, sr = librosa.load(file_path, sr=config.SAMPLE_RATE, mono=True)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    return np.concatenate([np.mean(chroma, axis=1), np.std(chroma, axis=1)])


def extract_all_features(file_path: str) -> np.ndarray:
    """Concatenate MFCC, spectral and chroma features."""
    mfcc = extract_mfcc(file_path)
    spectral = extract_spectral(file_path)
    chroma = extract_chroma(file_path)
    return np.concatenate([mfcc, spectral, chroma])


def load_data(use_all_features: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Load real/fake audio and return feature matrix and labels."""
    extractor = extract_all_features if use_all_features else extract_mfcc
    X, y = [], []

    for label, directory in ((0, config.REAL_DIR), (1, config.FAKE_DIR)):
        tag = "REAL" if label == 0 else "FAKE"
        print(f"Processing {tag} audio from {directory} ...")
        for fname in tqdm(os.listdir(directory)):
            if not fname.lower().endswith(config.AUDIO_EXTS):
                continue
            path = os.path.join(directory, fname)
            try:
                X.append(extractor(path))
                y.append(label)
            except Exception as exc:
                print(f"  [WARN] Skipping {fname}: {exc}")

    return np.array(X), np.array(y)


if __name__ == "__main__":
    X, y = load_data(use_all_features=True)
    np.save(config.FEATURES_OUT, X)
    np.save(config.LABELS_OUT, y)
    print(f"Saved {config.FEATURES_OUT} and {config.LABELS_OUT} ({X.shape})")
