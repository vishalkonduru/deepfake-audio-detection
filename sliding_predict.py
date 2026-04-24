"""Streaming prediction for long audio files (sliding window approach)."""

import argparse
import numpy as np
import librosa
import joblib

import config
from extract_features import _features_from_waveform
from logger import get_logger

log = get_logger(__name__)

DEFAULT_WINDOW_SEC = 3.0
DEFAULT_HOP_SEC = 1.5


def sliding_predict(
    file_path: str,
    model,
    window_sec: float = DEFAULT_WINDOW_SEC,
    hop_sec: float = DEFAULT_HOP_SEC,
) -> dict:
    """Classify overlapping windows of *file_path* and aggregate results."""
    y, sr = librosa.load(file_path, sr=config.SAMPLE_RATE, mono=True)
    win = int(window_sec * sr)
    hop = int(hop_sec * sr)

    segments = []
    start = 0
    while start + win <= len(y):
        chunk = y[start : start + win]
        feat = _features_from_waveform(chunk, sr).reshape(1, -1)
        proba = model.predict_proba(feat)[0]
        segments.append({
            "start_sec": round(start / sr, 2),
            "end_sec": round((start + win) / sr, 2),
            "p_fake": float(proba[1]),
        })
        start += hop

    if not segments:
        log.warning("Audio too short for window size %.1fs", window_sec)
        return {"label": "UNKNOWN", "segments": []}

    mean_fake = np.mean([s["p_fake"] for s in segments])
    label = "FAKE" if mean_fake >= 0.5 else "REAL"
    return {
        "label": label,
        "mean_p_fake": float(mean_fake),
        "segments": segments,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sliding-window deepfake detection")
    parser.add_argument("file", help="Audio file to analyse")
    parser.add_argument("--window", type=float, default=DEFAULT_WINDOW_SEC)
    parser.add_argument("--hop", type=float, default=DEFAULT_HOP_SEC)
    args = parser.parse_args()

    model = joblib.load(config.MODEL_OUT)
    result = sliding_predict(args.file, model, args.window, args.hop)
    print(f"Overall: {result['label']} (mean_p_fake={result['mean_p_fake']:.3f})")
    for seg in result["segments"]:
        print(f"  [{seg['start_sec']:6.1f}s – {seg['end_sec']:6.1f}s]  p_fake={seg['p_fake']:.3f}")
