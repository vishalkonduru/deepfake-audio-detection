"""Unit tests for feature extraction."""

import tempfile

import numpy as np
import pytest
import soundfile as sf

import config
from extract_features import (
    extract_all_features,
    extract_chroma,
    extract_mfcc,
    extract_spectral,
)


def _make_sine_wav(freq: float = 440.0, duration: float = 1.0) -> str:
    """Generate a short sine-wave WAV and return its temp path."""
    sr = config.SAMPLE_RATE
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    wave = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(tmp.name, wave, sr)
    return tmp.name


@pytest.fixture(scope="module")
def sine_wav():
    return _make_sine_wav()


class TestExtractMFCC:
    def test_shape(self, sine_wav):
        feat = extract_mfcc(sine_wav)
        assert feat.shape == (config.N_MFCC * 2,)

    def test_no_nan(self, sine_wav):
        feat = extract_mfcc(sine_wav)
        assert not np.any(np.isnan(feat))

    def test_custom_n_mfcc(self, sine_wav):
        feat = extract_mfcc(sine_wav, n_mfcc=20)
        assert feat.shape == (40,)


class TestExtractSpectral:
    def test_shape(self, sine_wav):
        feat = extract_spectral(sine_wav)
        assert feat.shape == (6,)

    def test_no_nan(self, sine_wav):
        assert not np.any(np.isnan(extract_spectral(sine_wav)))


class TestExtractChroma:
    def test_shape(self, sine_wav):
        feat = extract_chroma(sine_wav)
        assert feat.shape == (24,)


class TestExtractAllFeatures:
    def test_combined_shape(self, sine_wav):
        feat = extract_all_features(sine_wav)
        expected = config.N_MFCC * 2 + 6 + 24
        assert feat.shape == (expected,)

    def test_deterministic(self, sine_wav):
        f1 = extract_all_features(sine_wav)
        f2 = extract_all_features(sine_wav)
        np.testing.assert_array_equal(f1, f2)
