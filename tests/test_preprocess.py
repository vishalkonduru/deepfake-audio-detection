"""Tests for preprocess.py audio conversion helper."""

import os
import tempfile
import numpy as np
import soundfile as sf
import pytest
from unittest.mock import patch

from preprocess import to_wav
import config


@pytest.fixture()
def wav_path():
    sr = config.SAMPLE_RATE
    t = np.linspace(0, 1, sr, endpoint=False)
    wave = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(tmp.name, wave, sr)
    yield tmp.name
    os.unlink(tmp.name)


def test_wav_passthrough(wav_path):
    """WAV files should be returned as-is without calling ffmpeg."""
    result = to_wav(wav_path)
    assert result == wav_path


def test_non_wav_without_ffmpeg_raises():
    """Non-WAV input without ffmpeg on PATH should raise RuntimeError."""
    with patch("preprocess.shutil.which", return_value=None):
        with pytest.raises(RuntimeError, match="ffmpeg"):
            to_wav("audio.mp3")
