"""Hyperparameter search using GridSearchCV."""

import numpy as np
import joblib
import json
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split

import config

PARAM_GRID = {
    "clf__C": [0.1, 1.0, 10.0],
    "clf__kernel": ["rbf", "linear"],
    "clf__gamma": ["scale", "auto"],
}

if __name__ == "__main__":
    X = np.load(config.FEATURES_OUT)
    y = np.load(config.LABELS_OUT)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )

    pipe = Pipeline([("scaler", StandardScaler()), ("clf", SVC(probability=True))])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)

    gs = GridSearchCV(pipe, PARAM_GRID, cv=cv, scoring="roc_auc", n_jobs=-1, verbose=2)
    gs.fit(X_train, y_train)

    print("Best params:", gs.best_params_)
    print(f"Best CV ROC-AUC: {gs.best_score_:.4f}")

    best = gs.best_estimator_
    y_pred = best.predict(X_test)
    y_proba = best.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, y_pred, target_names=["REAL", "FAKE"]))
    print(f"Test ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

    joblib.dump(best, config.MODEL_OUT)
    print(f"Best model saved to {config.MODEL_OUT}")

    result = {
        "best_params": gs.best_params_,
        "best_cv_roc_auc": gs.best_score_,
        "test_roc_auc": roc_auc_score(y_test, y_proba),
    }
    with open("hparam_results.json", "w") as fh:
        json.dump(result, fh, indent=2)
