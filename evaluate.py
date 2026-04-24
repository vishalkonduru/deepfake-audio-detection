"""Evaluate a saved model and produce a classification report + confusion matrix plot."""

import argparse
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split

import config


def evaluate(model_path: str, features: str, labels: str, output_dir: str = ".") -> None:
    model = joblib.load(model_path)
    X = np.load(features)
    y = np.load(labels)

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, target_names=["REAL", "FAKE"], output_dict=True)
    auc = roc_auc_score(y_test, y_proba)
    report["roc_auc"] = auc

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "eval_report.json", "w") as fh:
        json.dump(report, fh, indent=2)

    print(classification_report(y_test, y_pred, target_names=["REAL", "FAKE"]))
    print(f"ROC-AUC: {auc:.4f}")

    # Confusion matrix plot (requires matplotlib)
    try:
        import matplotlib.pyplot as plt
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["REAL", "FAKE"])
        fig, ax = plt.subplots()
        disp.plot(ax=ax)
        fig.savefig(out / "confusion_matrix.png", dpi=150, bbox_inches="tight")
        print(f"Saved confusion matrix to {out / 'confusion_matrix.png'}")
    except ImportError:
        print("matplotlib not installed – skipping confusion matrix plot.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate deepfake detection model")
    parser.add_argument("--model", default=config.MODEL_OUT)
    parser.add_argument("--features", default=config.FEATURES_OUT)
    parser.add_argument("--labels", default=config.LABELS_OUT)
    parser.add_argument("--output-dir", default="evaluation")
    args = parser.parse_args()
    evaluate(args.model, args.features, args.labels, args.output_dir)
