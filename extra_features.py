"""Zero-Crossing Rate and RMS energy feature helpers."""

import numpy as np
import librosa

import config


def extract_zcr(y: np.ndarray, sr: int) -> np.ndarray:
    """Return mean and std of Zero-Crossing Rate."""
    zcr = librosa.feature.zero_crossing_rate(y)
    return np.array([float(np.mean(zcr)), float(np.std(zcr))])


def extract_rms(y: np.ndarray, sr: int) -> np.ndarray:
    """Return mean and std of RMS energy."""
    rms = librosa.feature.rms(y=y)
    return np.array([float(np.mean(rms)), float(np.std(rms))])


def extract_tonnetz(y: np.ndarray, sr: int) -> np.ndarray:
    """Return mean and std of Tonnetz (tonal centroid) features."""
    harmonic = librosa.effects.harmonic(y)
    tonnetz = librosa.feature.tonnetz(y=harmonic, sr=sr)
    return np.concatenate([np.mean(tonnetz, axis=1), np.std(tonnetz, axis=1)])


def extract_mel_spectrogram_stats(y: np.ndarray, sr: int, n_mels: int = 64) -> np.ndarray:
    """Return mean and std across mel-spectrogram frequency bins."""
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return np.concatenate([np.mean(mel_db, axis=1), np.std(mel_db, axis=1)])
