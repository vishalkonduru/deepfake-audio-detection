"""Integration tests for the Flask API."""

import io
import numpy as np
import pytest
import soundfile as sf
import tempfile

import config

# Patch model loading before importing app
import joblib
from unittest.mock import MagicMock, patch
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """Create a Flask test client with a mock model."""
    # Build a tiny fitted pipeline so predict_proba works
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=20, n_features=110, random_state=0)
    pipe = Pipeline([("sc", StandardScaler()), ("clf", SVC(probability=True))])
    pipe.fit(X, y)

    model_path = str(tmp_path / "model.joblib")
    joblib.dump(pipe, model_path)
    monkeypatch.setattr(config, "MODEL_OUT", model_path)

    # Re-import app with patched config
    import importlib
    import app as flask_app
    importlib.reload(flask_app)
    flask_app.app.config["TESTING"] = True
    with flask_app.app.test_client() as c:
        yield c


def _wav_bytes() -> bytes:
    sr = 16000
    t = np.linspace(0, 1, sr, endpoint=False)
    wave = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, wave, sr, format="WAV")
    buf.seek(0)
    return buf.read()


class TestHealthEndpoint:
    def test_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_body(self, client):
        data = resp = client.get("/health").get_json()
        assert data["status"] == "ok"


class TestInfoEndpoint:
    def test_200(self, client):
        assert client.get("/info").status_code == 200

    def test_fields(self, client):
        data = client.get("/info").get_json()
        assert "sample_rate" in data
        assert "supported_formats" in data


class TestPredictEndpoint:
    def test_no_file_returns_400(self, client):
        resp = client.post("/predict")
        assert resp.status_code == 400

    def test_valid_wav(self, client):
        data = {"file": (io.BytesIO(_wav_bytes()), "test.wav")}
        resp = client.post("/predict", data=data, content_type="multipart/form-data")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["label"] in ("REAL", "FAKE")
        assert 0.0 <= body["confidence"] <= 1.0

    def test_unsupported_format_returns_415(self, client):
        data = {"file": (io.BytesIO(b"garbage"), "test.txt")}
        resp = client.post("/predict", data=data, content_type="multipart/form-data")
        assert resp.status_code == 415
