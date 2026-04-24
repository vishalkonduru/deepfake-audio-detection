"""Unit tests for extra_features helpers."""

import numpy as np
import pytest

import config
from extra_features import extract_zcr, extract_rms, extract_tonnetz, extract_mel_spectrogram_stats

SR = config.SAMPLE_RATE


@pytest.fixture(scope="module")
def sine_wave():
    t = np.linspace(0, 2.0, int(SR * 2.0), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


class TestZCR:
    def test_shape(self, sine_wave):
        assert extract_zcr(sine_wave, SR).shape == (2,)

    def test_no_nan(self, sine_wave):
        assert not np.any(np.isnan(extract_zcr(sine_wave, SR)))

    def test_non_negative_mean(self, sine_wave):
        assert extract_zcr(sine_wave, SR)[0] >= 0


class TestRMS:
    def test_shape(self, sine_wave):
        assert extract_rms(sine_wave, SR).shape == (2,)

    def test_positive_mean(self, sine_wave):
        assert extract_rms(sine_wave, SR)[0] > 0


class TestTonnetz:
    def test_shape(self, sine_wave):
        feat = extract_tonnetz(sine_wave, SR)
        assert feat.shape == (12,)  # 6 bins * 2 (mean + std)


class TestMelSpecStats:
    def test_shape_64_mels(self, sine_wave):
        feat = extract_mel_spectrogram_stats(sine_wave, SR, n_mels=64)
        assert feat.shape == (128,)  # 64 * 2

    def test_shape_32_mels(self, sine_wave):
        feat = extract_mel_spectrogram_stats(sine_wave, SR, n_mels=32)
        assert feat.shape == (64,)
