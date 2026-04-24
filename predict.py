"""CLI prediction tool."""

import argparse
import sys

import joblib
import numpy as np

import config
from extract_features import extract_all_features


def predict_file(file_path: str) -> dict:
    """Load model, extract features from *file_path* and return prediction."""
    try:
        model = joblib.load(config.MODEL_OUT)
    except FileNotFoundError:
        print(f"[ERROR] Model not found at '{config.MODEL_OUT}'. Run train_model.py first.",
              file=sys.stderr)
        sys.exit(1)

    feat = extract_all_features(file_path).reshape(1, -1)
    pred = model.predict(feat)[0]
    proba = model.predict_proba(feat)[0]
    label = "FAKE" if pred == 1 else "REAL"
    return {
        "file": file_path,
        "label": label,
        "confidence": float(max(proba)),
        "probabilities": {"real": float(proba[0]), "fake": float(proba[1])},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deepfake audio detection – CLI predictor")
    parser.add_argument("files", nargs="+", metavar="FILE", help="Audio file(s) to classify")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print probabilities too")
    args = parser.parse_args()

    for fp in args.files:
        result = predict_file(fp)
        if args.verbose:
            print(f"{result['file']}: {result['label']} "
                  f"(confidence={result['confidence']:.3f}, "
                  f"p_real={result['probabilities']['real']:.3f}, "
                  f"p_fake={result['probabilities']['fake']:.3f})")
        else:
            print(f"{result['file']}: {result['label']}")
