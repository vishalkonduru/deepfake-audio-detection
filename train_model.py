"""Model training script with StandardScaler and improved reporting."""

import json

import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import config


def build_pipeline() -> Pipeline:
    """Return a StandardScaler + SVC pipeline."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                SVC(
                    kernel=config.SVM_KERNEL,
                    C=config.SVM_C,
                    gamma=config.SVM_GAMMA,
                    probability=True,
                    random_state=config.RANDOM_STATE,
                ),
            ),
        ]
    )


if __name__ == "__main__":
    X = np.load(config.FEATURES_OUT)
    y = np.load(config.LABELS_OUT)

    # Optional: debug to see dataset size
    print("Total samples:", X.shape[0])
    unique, counts = np.unique(y, return_counts=True)
    print("Label counts:", dict(zip(unique, counts)))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )

    pipe = build_pipeline()

    # Cross-validation on training data
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
    print(f"CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Fit on full training set
    pipe.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred, target_names=["REAL", "FAKE"]))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"Test ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

    # Save model
    joblib.dump(pipe, config.MODEL_OUT)
    print(f"Saved pipeline to {config.MODEL_OUT}")

    # Persist metrics for CI
    metrics = {
        "cv_roc_auc_mean": float(cv_scores.mean()),
        "cv_roc_auc_std": float(cv_scores.std()),
        "test_roc_auc": float(roc_auc_score(y_test, y_proba)),
    }
    with open("metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)
    print("Saved metrics.json")
