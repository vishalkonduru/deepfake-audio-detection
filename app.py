from flask import Flask, request, jsonify, render_template
import joblib
import librosa
import numpy as np
import os
import tempfile

MODEL_OUT = "model.joblib"

app = Flask(__name__)
model = joblib.load(MODEL_OUT)

def extract_mfcc(file_path, n_mfcc=40):
    y, sr = librosa.load(file_path, sr=16000, mono=True)
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
        proba = model.predict_proba(feat)[0]
        pred = model.predict(feat)[0]

        # Labels: 0 = REAL, 1 = FAKE (same as extract_features.py)
        label = "REAL" if pred == 0 else "FAKE"

        return jsonify({
            "label": label,
            "probabilities": {
                "real": float(proba[0]),
                "fake": float(proba[1])
            }
        })
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
