"""Unit tests for data augmentation helpers."""

import numpy as np
import pytest

import config
from augment import add_gaussian_noise, time_stretch, pitch_shift, random_crop, augment

SR = config.SAMPLE_RATE
DURATION = 1.0


@pytest.fixture(scope="module")
def sine_wave():
    t = np.linspace(0, DURATION, int(SR * DURATION), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


class TestAddGaussianNoise:
    def test_shape_preserved(self, sine_wave):
        out = add_gaussian_noise(sine_wave)
        assert out.shape == sine_wave.shape

    def test_values_differ(self, sine_wave):
        out = add_gaussian_noise(sine_wave)
        assert not np.allclose(out, sine_wave)


class TestTimeStretch:
    def test_output_is_1d(self, sine_wave):
        out = time_stretch(sine_wave, rate=1.1)
        assert out.ndim == 1


class TestPitchShift:
    def test_shape_preserved(self, sine_wave):
        out = pitch_shift(sine_wave, SR, n_steps=2)
        assert out.shape == sine_wave.shape


class TestRandomCrop:
    def test_output_length(self, sine_wave):
        out = random_crop(sine_wave, SR, duration=0.5)
        assert len(out) == int(SR * 0.5)

    def test_padding_short_input(self):
        short = np.zeros(100)
        out = random_crop(short, SR, duration=1.0)
        assert len(out) == SR


class TestAugment:
    def test_returns_five_variants(self, sine_wave):
        variants = augment(sine_wave, SR)
        assert len(variants) == 5

    def test_all_1d(self, sine_wave):
        for v in augment(sine_wave, SR):
            assert v.ndim == 1
