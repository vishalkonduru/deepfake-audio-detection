import os
import numpy as np
import joblib
import config

from extract_features import extract_all_features

REAL_DIR = "manual_test/real"
FAKE_DIR = "manual_test/fake"

def load_manual():
    X = []
    y = []
    paths = []
    for label, folder in [(0, REAL_DIR), (1, FAKE_DIR)]:
        if not os.path.isdir(folder):
            continue
        for fname in os.listdir(folder):
            if not fname.lower().endswith((".wav", ".flac", ".mp3", ".m4a", ".ogg")):
                continue
            path = os.path.join(folder, fname)
            try:
                feat = extract_all_features(path)
                X.append(feat)
                y.append(label)
                paths.append(path)
            except Exception as e:
                print("Error on", path, "->", e)
    return np.array(X), np.array(y), paths

if __name__ == "__main__":
    model = joblib.load(config.MODEL_OUT)
    X, y, paths = load_manual()
    if len(X) == 0:
        print("No manual test files found.")
        raise SystemExit(0)
    print("Manual feature shape:", X.shape)
    preds = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    for path, true, pred, p in zip(paths, y, preds, proba):
        print(f"{path} | true={'REAL' if true==0 else 'FAKE'} | pred={'REAL' if pred==0 else 'FAKE'} | deepfake_prob={p:.3f}")
