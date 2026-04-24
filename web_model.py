import numpy as np
import joblib
import config
from extract_features import extract_all_features

# Load once at startup
model = joblib.load(config.MODEL_OUT)

def predict_audio_file(file_path: str):
    """Return (label_str, prob_fake) for a single audio file path."""
    feat = extract_all_features(file_path)  # shape (110,)
    X = np.array([feat])
    pred = model.predict(X)[0]
    proba_fake = model.predict_proba(X)[0, 1]  # assumes class 1 = FAKE
    label_str = "FAKE" if pred == 1 else "REAL"
    return label_str, float(proba_fake)
