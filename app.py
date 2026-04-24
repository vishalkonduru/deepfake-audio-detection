"""Flask API for deepfake audio detection."""

import os
import tempfile
import logging
from pathlib import Path

import joblib
import librosa
import numpy as np
from flask import Flask, request, jsonify

import config
from extract_features import extract_all_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load model once at startup
try:
    model = joblib.load(config.MODEL_OUT)
    logger.info("Model loaded from %s", config.MODEL_OUT)
except FileNotFoundError:
    model = None
    logger.warning("Model file not found at %s – /predict will return 503", config.MODEL_OUT)


def _check_file_size(file_storage) -> bool:
    """Return True if file is within the configured size limit."""
    file_storage.seek(0, 2)
    size_mb = file_storage.tell() / (1024 * 1024)
    file_storage.seek(0)
    return size_mb <= config.MAX_FILE_SIZE_MB


@app.route("/health", methods=["GET"])
def health():
    """Liveness probe."""
    return jsonify({"status": "ok", "model_loaded": model is not None})


@app.route("/predict", methods=["POST"])
def predict():
    """Accept a WAV/FLAC/MP3 file and return real/fake prediction."""
    if model is None:
        return jsonify({"error": "model not loaded"}), 503

    if "file" not in request.files:
        return jsonify({"error": "no file field in request"}), 400

    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "empty filename"}), 400

    ext = Path(f.filename).suffix.lower()
    if ext not in config.AUDIO_EXTS:
        return jsonify({"error": f"unsupported format {ext}"}), 415

    if not _check_file_size(f):
        return jsonify({"error": f"file exceeds {config.MAX_FILE_SIZE_MB} MB limit"}), 413

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        f.save(tmp.name)
        tmp_path = tmp.name

    try:
        feat = extract_all_features(tmp_path).reshape(1, -1)
        proba = model.predict_proba(feat)[0]
        pred = model.predict(feat)[0]
        label = "FAKE" if pred == 1 else "REAL"
        return jsonify({
            "label": label,
            "confidence": float(max(proba)),
            "probabilities": {
                "real": float(proba[0]),
                "fake": float(proba[1]),
            },
        })
    except Exception as exc:
        logger.exception("Prediction failed")
        return jsonify({"error": str(exc)}), 500
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.route("/info", methods=["GET"])
def info():
    """Return runtime configuration info."""
    return jsonify({
        "sample_rate": config.SAMPLE_RATE,
        "n_mfcc": config.N_MFCC,
        "max_file_size_mb": config.MAX_FILE_SIZE_MB,
        "supported_formats": list(config.AUDIO_EXTS),
    })


if __name__ == "__main__":
    app.run(host=config.API_HOST, port=config.API_PORT, debug=False)
