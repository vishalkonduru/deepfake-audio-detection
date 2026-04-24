"""Random-Forest alternative classifier for comparison."""

import numpy as np
import joblib
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score

import config
from logger import get_logger

log = get_logger(__name__)
RF_MODEL_OUT = "model_rf.joblib"


if __name__ == "__main__":
    X = np.load(config.FEATURES_OUT)
    y = np.load(config.LABELS_OUT)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        )),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
    log.info("RF CV ROC-AUC: %.4f (+/- %.4f)", cv_scores.mean(), cv_scores.std())

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred, target_names=["REAL", "FAKE"]))
    auc = roc_auc_score(y_test, y_proba)
    log.info("RF Test ROC-AUC: %.4f", auc)

    joblib.dump(pipe, RF_MODEL_OUT)
    log.info("Saved RF model to %s", RF_MODEL_OUT)

    metrics = {"cv_roc_auc_mean": float(cv_scores.mean()), "test_roc_auc": float(auc)}
    with open("metrics_rf.json", "w") as fh:
        json.dump(metrics, fh, indent=2)
