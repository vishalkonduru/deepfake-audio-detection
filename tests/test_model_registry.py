"""Unit tests for model_registry utilities."""

import json
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.svm import SVC

import model_registry as mr


@pytest.fixture()
def tmp_registry(tmp_path, monkeypatch):
    monkeypatch.setattr(mr, "MODEL_REGISTRY", tmp_path / "registry")
    return tmp_path


@pytest.fixture()
def dummy_model_path(tmp_path):
    clf = SVC(probability=True)
    X = np.random.randn(10, 4)
    y = np.array([0] * 5 + [1] * 5)
    clf.fit(X, y)
    path = str(tmp_path / "model.joblib")
    joblib.dump(clf, path)
    return path


def test_register_creates_metadata(tmp_registry, dummy_model_path):
    version = mr.register_model(dummy_model_path, metrics={"auc": 0.95}, tag="test")
    meta_path = mr.MODEL_REGISTRY / version / "metadata.json"
    assert meta_path.exists()
    with open(meta_path) as fh:
        meta = json.load(fh)
    assert meta["metrics"]["auc"] == 0.95
    assert "sha256" in meta


def test_list_versions_returns_entries(tmp_registry, dummy_model_path):
    mr.register_model(dummy_model_path, metrics={}, tag="v1")
    mr.register_model(dummy_model_path, metrics={}, tag="v2")
    versions = mr.list_versions()
    assert len(versions) == 2


def test_load_version(tmp_registry, dummy_model_path):
    version = mr.register_model(dummy_model_path, metrics={}, tag="load_test")
    loaded = mr.load_version(version)
    assert hasattr(loaded, "predict")
