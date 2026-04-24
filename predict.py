import sys
import joblib
import librosa
import numpy as np

MODEL_OUT = "model.joblib"

def extract_mfcc(file_path, n_mfcc=40):
    y, sr = librosa.load(file_path, sr=16000, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    return np.concatenate([mfcc_mean, mfcc_std])

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py path/to/audio")
        sys.exit(1)

    audio_path = sys.argv[1]
    model = joblib.load(MODEL_OUT)

    feat = extract_mfcc(audio_path).reshape(1, -1)
    proba = model.predict_proba(feat)[0]
    pred = model.predict(feat)[0]

    label = "FAKE" if pred == 1 else "REAL"
    print(f"Prediction: {label}")
    print(f"Probabilities [REAL, FAKE]: {proba}")
