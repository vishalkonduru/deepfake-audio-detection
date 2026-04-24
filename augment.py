"""Data augmentation helpers for training set expansion."""

import numpy as np
import librosa
from typing import Tuple


def add_gaussian_noise(y: np.ndarray, noise_factor: float = 0.005) -> np.ndarray:
    """Add Gaussian noise to waveform."""
    noise = np.random.randn(len(y))
    return y + noise_factor * noise


def time_stretch(y: np.ndarray, rate: float = 1.1) -> np.ndarray:
    """Time-stretch a waveform by *rate* (>1 = faster, <1 = slower)."""
    return librosa.effects.time_stretch(y, rate=rate)


def pitch_shift(y: np.ndarray, sr: int, n_steps: float = 2.0) -> np.ndarray:
    """Shift pitch by *n_steps* semitones."""
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)


def random_crop(y: np.ndarray, sr: int, duration: float = 1.0) -> np.ndarray:
    """Return a random crop of *duration* seconds. Pads if too short."""
    target = int(sr * duration)
    if len(y) >= target:
        start = np.random.randint(0, len(y) - target + 1)
        return y[start : start + target]
    return np.pad(y, (0, target - len(y)))


def augment(y: np.ndarray, sr: int) -> list:
    """Return a list of augmented variants of *y*."""
    variants = [
        add_gaussian_noise(y),
        time_stretch(y, rate=0.9),
        time_stretch(y, rate=1.1),
        pitch_shift(y, sr, n_steps=2),
        pitch_shift(y, sr, n_steps=-2),
    ]
    return variants
