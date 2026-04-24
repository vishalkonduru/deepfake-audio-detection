"""Batch prediction utility: classify an entire directory of audio files."""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np

import config
from extract_features import extract_all_features
from logger import get_logger

log = get_logger(__name__)


def batch_predict(input_dir: str, output_format: str = "csv") -> list:
    """Classify all audio files in *input_dir* and return results list."""
    try:
        model = joblib.load(config.MODEL_OUT)
    except FileNotFoundError:
        log.error("Model not found at '%s'. Run train_model.py first.", config.MODEL_OUT)
        sys.exit(1)

    results = []
    for fname in sorted(os.listdir(input_dir)):
        if not fname.lower().endswith(config.AUDIO_EXTS):
            continue
        fpath = os.path.join(input_dir, fname)
        try:
            feat = extract_all_features(fpath).reshape(1, -1)
            pred = model.predict(feat)[0]
            proba = model.predict_proba(feat)[0]
            results.append({
                "file": fname,
                "label": "FAKE" if pred == 1 else "REAL",
                "confidence": float(max(proba)),
                "p_real": float(proba[0]),
                "p_fake": float(proba[1]),
            })
            log.info("%s -> %s (conf=%.3f)", fname, results[-1]["label"], results[-1]["confidence"])
        except Exception as exc:
            log.warning("Failed on %s: %s", fname, exc)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch-predict a directory of audio files")
    parser.add_argument("input_dir", help="Directory containing audio files")
    parser.add_argument("--format", choices=["csv", "json"], default="csv")
    parser.add_argument("--output", default="batch_results")
    args = parser.parse_args()

    results = batch_predict(args.input_dir)

    if args.format == "json":
        out_path = args.output + ".json"
        with open(out_path, "w") as fh:
            json.dump(results, fh, indent=2)
    else:
        out_path = args.output + ".csv"
        with open(out_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["file", "label", "confidence", "p_real", "p_fake"])
            writer.writeheader()
            writer.writerows(results)

    print(f"Results saved to {out_path} ({len(results)} files processed)")
