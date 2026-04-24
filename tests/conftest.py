"""conftest.py — shared pytest fixtures."""

import numpy as np
import pytest
import soundfile as sf
import tempfile
import os

import config


@pytest.fixture(scope="session")
def sample_rate():
    return config.SAMPLE_RATE


@pytest.fixture(scope="session")
def sine_wav_path(tmp_path_factory):
    """A 2-second 440 Hz sine wave saved as WAV."""
    sr = config.SAMPLE_RATE
    t = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False)
    wave = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    path = str(tmp_path_factory.mktemp("audio") / "sine_440.wav")
    sf.write(path, wave, sr)
    return path


@pytest.fixture(scope="session")
def noise_wav_path(tmp_path_factory):
    """A 2-second white-noise WAV."""
    sr = config.SAMPLE_RATE
    rng = np.random.default_rng(0)
    wave = rng.uniform(-0.3, 0.3, int(sr * 2.0)).astype(np.float32)
    path = str(tmp_path_factory.mktemp("audio") / "noise.wav")
    sf.write(path, wave, sr)
    return path
