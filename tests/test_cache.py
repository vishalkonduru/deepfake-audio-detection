"""Unit tests for PredictionCache."""

import tempfile
import os
import pytest

from cache import PredictionCache


@pytest.fixture()
def wav_file():
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.write(b"RIFF" + b"\x00" * 100)
    tmp.flush()
    yield tmp.name
    os.unlink(tmp.name)


@pytest.fixture()
def cache():
    return PredictionCache(max_size=4)


def test_cache_miss_on_new_file(cache, wav_file):
    assert cache.get(wav_file) is None


def test_cache_hit_after_set(cache, wav_file):
    result = {"label": "REAL", "confidence": 0.9}
    cache.set(wav_file, result)
    assert cache.get(wav_file) == result


def test_hit_rate_updates(cache, wav_file):
    cache.set(wav_file, {"label": "FAKE"})
    cache.get(wav_file)  # hit
    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["hit_rate"] > 0


def test_lru_eviction(cache):
    files = []
    for i in range(5):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp.write(bytes([i] * 50))
        tmp.flush()
        files.append(tmp.name)
        cache.set(tmp.name, {"label": "REAL"})

    # Cache max_size=4 so oldest should be evicted
    assert cache.stats()["size"] == 4
    for f in files:
        os.unlink(f)


def test_clear_resets_stats(cache, wav_file):
    cache.set(wav_file, {"label": "REAL"})
    cache.get(wav_file)
    cache.clear()
    stats = cache.stats()
    assert stats["size"] == 0
    assert stats["hits"] == 0
