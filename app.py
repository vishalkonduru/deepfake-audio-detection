"""Flask API – adds /predict/batch and sliding-window endpoint."""

import io
import os
import tempfile
import logging
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, request, jsonify

import config
from extract_features import extract_all_features
from sliding_predict import sliding_predict
from logger import get_logger

log = get_logger(__name__)
app = Flask(__name__)

try:
    model = joblib.load(config.MODEL_OUT)
    log.info("Model loaded from %s", config.MODEL_OUT)
except FileNotFoundError:
    model = None
    log.warning("Model file not found at %s", config.MODEL_OUT)


def _save_upload(f) -> str:
    ext = Path(f.filename).suffix.lower()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    f.save(tmp.name)
    return tmp.name


def _check_file(f):
    if f.filename == "":
        return jsonify({"error": "empty filename"}), 400
    ext = Path(f.filename).suffix.lower()
    if ext not in config.AUDIO_EXTS:
        return jsonify({"error": f"unsupported format {ext}"}), 415
    f.seek(0, 2)
    size_mb = f.tell() / (1024 * 1024)
    f.seek(0)
    if size_mb > config.MAX_FILE_SIZE_MB:
        return jsonify({"error": f"file exceeds {config.MAX_FILE_SIZE_MB} MB"}), 413
    return None


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


@app.route("/info", methods=["GET"])
def info():
    return jsonify({
        "sample_rate": config.SAMPLE_RATE,
        "n_mfcc": config.N_MFCC,
        "max_file_size_mb": config.MAX_FILE_SIZE_MB,
        "supported_formats": list(config.AUDIO_EXTS),
    })


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "model not loaded"}), 503
    if "file" not in request.files:
        return jsonify({"error": "no file field in request"}), 400
    f = request.files["file"]
    err = _check_file(f)
    if err:
        return err
    tmp_path = _save_upload(f)
    try:
        feat = extract_all_features(tmp_path).reshape(1, -1)
        proba = model.predict_proba(feat)[0]
        pred = model.predict(feat)[0]
        label = "FAKE" if pred == 1 else "REAL"
        return jsonify({
            "label": label,
            "confidence": float(max(proba)),
            "probabilities": {"real": float(proba[0]), "fake": float(proba[1])},
        })
    except Exception as exc:
        log.exception("Prediction failed")
        return jsonify({"error": str(exc)}), 500
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.route("/predict/sliding", methods=["POST"])
def predict_sliding():
    """Sliding-window prediction for long audio files."""
    if model is None:
        return jsonify({"error": "model not loaded"}), 503
    if "file" not in request.files:
        return jsonify({"error": "no file"}), 400
    f = request.files["file"]
    err = _check_file(f)
    if err:
        return err
    window = float(request.form.get("window_sec", 3.0))
    hop = float(request.form.get("hop_sec", 1.5))
    tmp_path = _save_upload(f)
    try:
        result = sliding_predict(tmp_path, model, window_sec=window, hop_sec=hop)
        return jsonify(result)
    except Exception as exc:
        log.exception("Sliding prediction failed")
        return jsonify({"error": str(exc)}), 500
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    app.run(host=config.API_HOST, port=config.API_PORT, debug=False)
