"""Simple in-process prediction cache to avoid redundant model inference."""

import hashlib
import os
from collections import OrderedDict
from threading import Lock
from typing import Optional


class PredictionCache:
    """Thread-safe LRU cache keyed on file SHA-256 hash."""

    def __init__(self, max_size: int = 256):
        self._cache: OrderedDict = OrderedDict()
        self._lock = Lock()
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def _file_hash(self, path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    def get(self, path: str) -> Optional[dict]:
        key = self._file_hash(path)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self.hits += 1
                return self._cache[key]
            self.misses += 1
            return None

    def set(self, path: str, result: dict) -> None:
        key = self._file_hash(path)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = result
            if len(self._cache) > self.max_size:
                self._cache.popitem(last=False)

    def stats(self) -> dict:
        with self._lock:
            total = self.hits + self.misses
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": self.hits / total if total else 0.0,
            }

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self.hits = 0
            self.misses = 0


# Module-level singleton
default_cache = PredictionCache()
