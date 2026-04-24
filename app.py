import os
import tempfile

import joblib
import librosa
import numpy as np
from flask import Flask, jsonify, render_template, request

import config

app = Flask(__name__)

_model = None


def _get_model():
    """Lazy-load the model so the module can be imported without model.joblib."""
    global _model
    if _model is None:
        _model = joblib.load(config.MODEL_OUT)
    return _model


def extract_mfcc(file_path: str, n_mfcc: int = config.N_MFCC) -> np.ndarray:
    y, sr = librosa.load(file_path, sr=config.SAMPLE_RATE, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    return np.concatenate([mfcc_mean, mfcc_std])


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "no file"}), 400

    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "empty filename"}), 400

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            f.save(tmp.name)
            tmp_path = tmp.name

        feat = extract_mfcc(tmp_path).reshape(1, -1)
        model = _get_model()
        proba = model.predict_proba(feat)[0]
        pred = model.predict(feat)[0]

        label = "REAL" if pred == 0 else "FAKE"

        return jsonify(
            {
                "label": label,
                "probabilities": {
                    "real": float(proba[0]),
                    "fake": float(proba[1]),
                },
            }
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    app.run(host=config.API_HOST, port=config.API_PORT)
