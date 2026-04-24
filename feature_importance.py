"""Lightweight feature importance analysis using permutation importance."""

import argparse
import json
import numpy as np
import joblib
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

import config
from logger import get_logger

log = get_logger(__name__)

FEATURE_NAMES = (
    [f"mfcc_mean_{i}" for i in range(config.N_MFCC)]
    + [f"mfcc_std_{i}" for i in range(config.N_MFCC)]
    + ["spec_centroid_mean", "spec_centroid_std",
       "spec_bandwidth_mean", "spec_bandwidth_std",
       "spec_rolloff_mean", "spec_rolloff_std"]
    + [f"chroma_mean_{i}" for i in range(12)]
    + [f"chroma_std_{i}" for i in range(12)]
)


def run_permutation_importance(
    model_path: str,
    features_path: str,
    labels_path: str,
    top_n: int = 20,
    n_repeats: int = 10,
) -> list:
    model = joblib.load(model_path)
    X = np.load(features_path)
    y = np.load(labels_path)

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE, stratify=y
    )

    log.info("Running permutation importance (n_repeats=%d) ...", n_repeats)
    result = permutation_importance(
        model, X_test, y_test,
        n_repeats=n_repeats,
        random_state=config.RANDOM_STATE,
        scoring="roc_auc",
    )

    names = FEATURE_NAMES[:X_test.shape[1]]
    ranked = sorted(
        zip(names, result.importances_mean, result.importances_std),
        key=lambda x: x[1],
        reverse=True,
    )

    print(f"{'Feature':<35} {'Importance':>12} {'Std':>8}")
    print("-" * 58)
    for name, imp, std in ranked[:top_n]:
        print(f"{name:<35} {imp:>12.4f} {std:>8.4f}")

    return [{"feature": n, "importance": float(i), "std": float(s)} for n, i, s in ranked]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature importance analysis")
    parser.add_argument("--model", default=config.MODEL_OUT)
    parser.add_argument("--features", default=config.FEATURES_OUT)
    parser.add_argument("--labels", default=config.LABELS_OUT)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--n-repeats", type=int, default=10)
    parser.add_argument("--output", default="feature_importance.json")
    args = parser.parse_args()

    ranking = run_permutation_importance(
        args.model, args.features, args.labels, args.top_n, args.n_repeats
    )
    with open(args.output, "w") as fh:
        json.dump(ranking, fh, indent=2)
    log.info("Saved to %s", args.output)
