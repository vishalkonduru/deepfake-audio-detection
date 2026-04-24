import os
import numpy as np
import librosa
from tqdm import tqdm

REAL_DIR = "data/real"
FAKE_DIR = "data/fake"
FEATURES_OUT = "features.npy"
LABELS_OUT = "labels.npy"

def extract_mfcc(file_path, n_mfcc=40):
    y, sr = librosa.load(file_path, sr=16000, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    return np.concatenate([mfcc_mean, mfcc_std])

def load_data():
    X = []
    y = []
    print("Processing REAL audio...")
    for fname in tqdm(os.listdir(REAL_DIR)):
        if not fname.lower().endswith((".wav", ".flac", ".mp3", ".m4a")):
            continue
        path = os.path.join(REAL_DIR, fname)
        X.append(extract_mfcc(path))
        y.append(0)

    print("Processing FAKE audio...")
    for fname in tqdm(os.listdir(FAKE_DIR)):
        if not fname.lower().endswith((".wav", ".flac", ".mp3", ".m4a")):
            continue
        path = os.path.join(FAKE_DIR, fname)
        X.append(extract_mfcc(path))
        y.append(1)

    return np.array(X), np.array(y)

if __name__ == "__main__":
    X, y = load_data()
    np.save(FEATURES_OUT, X)
    np.save(LABELS_OUT, y)
    print("Saved", FEATURES_OUT, "and", LABELS_OUT)
