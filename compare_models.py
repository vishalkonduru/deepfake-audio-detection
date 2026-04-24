"""Model comparison script: runs SVM and RF and prints a side-by-side report."""

import json
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score

import config
from logger import get_logger

log = get_logger(__name__)

MODELS = {
    "SVM-RBF": Pipeline([
        ("sc", StandardScaler()),
        ("clf", SVC(kernel="rbf", probability=True, random_state=config.RANDOM_STATE)),
    ]),
    "RandomForest": Pipeline([
        ("sc", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=300, random_state=config.RANDOM_STATE, n_jobs=-1)),
    ]),
    "GradientBoosting": Pipeline([
        ("sc", StandardScaler()),
        ("clf", GradientBoostingClassifier(n_estimators=200, random_state=config.RANDOM_STATE)),
    ]),
}


if __name__ == "__main__":
    X = np.load(config.FEATURES_OUT)
    y = np.load(config.LABELS_OUT)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
    comparison = {}

    print(f"{'Model':<20} {'CV AUC':>10} {'Test AUC':>10}")
    print("-" * 44)

    for name, pipe in MODELS.items():
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
        pipe.fit(X_train, y_train)
        test_auc = roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])
        comparison[name] = {"cv_auc": float(cv_scores.mean()), "test_auc": float(test_auc)}
        print(f"{name:<20} {cv_scores.mean():>10.4f} {test_auc:>10.4f}")

    with open("comparison.json", "w") as fh:
        json.dump(comparison, fh, indent=2)
    print("\nSaved comparison.json")
