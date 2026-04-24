"""Unit tests for sliding_predict."""

import numpy as np
import pytest
import soundfile as sf
import tempfile
from unittest.mock import MagicMock

import config
from sliding_predict import sliding_predict


def _make_wav(duration: float = 5.0) -> str:
    sr = config.SAMPLE_RATE
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    wave = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(tmp.name, wave, sr)
    return tmp.name


@pytest.fixture(scope="module")
def mock_model():
    m = MagicMock()
    m.predict_proba.return_value = np.array([[0.3, 0.7]])
    return m


def test_segments_returned(mock_model):
    wav = _make_wav(5.0)
    result = sliding_predict(wav, mock_model, window_sec=2.0, hop_sec=1.0)
    assert len(result["segments"]) > 0


def test_label_fake_when_high_proba(mock_model):
    wav = _make_wav(5.0)
    result = sliding_predict(wav, mock_model, window_sec=2.0, hop_sec=1.0)
    assert result["label"] == "FAKE"


def test_label_real_when_low_proba():
    m = MagicMock()
    m.predict_proba.return_value = np.array([[0.9, 0.1]])
    wav = _make_wav(5.0)
    result = sliding_predict(wav, m, window_sec=2.0, hop_sec=1.0)
    assert result["label"] == "REAL"


def test_too_short_audio(mock_model):
    wav = _make_wav(1.0)
    result = sliding_predict(wav, mock_model, window_sec=5.0, hop_sec=2.5)
    assert result["label"] == "UNKNOWN"
