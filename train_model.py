import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import joblib

FEATURES_OUT = "features.npy"
LABELS_OUT = "labels.npy"
MODEL_OUT = "model.joblib"

if __name__ == "__main__":
    X = np.load(FEATURES_OUT)
    y = np.load(LABELS_OUT)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = SVC(kernel="rbf", probability=True)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    joblib.dump(clf, MODEL_OUT)
    print("Saved model to", MODEL_OUT)
