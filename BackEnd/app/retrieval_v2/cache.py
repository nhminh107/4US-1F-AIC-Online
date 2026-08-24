"""Task 13: In-memory LRU cache for query text embeddings and plans."""

from __future__ import annotations

import collections
import threading
from typing import Any


class QueryEmbeddingCache:
    """Thread-safe bounded LRU cache for encoded query vectors and plan artifacts."""

    def __init__(self, maxsize: int = 1000) -> None:
        self.maxsize = maxsize
        self._cache: collections.OrderedDict[tuple[str, str], Any] = collections.OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, model_version: str, normalized_query: str) -> Any | None:
        key = (model_version, normalized_query)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self.hits += 1
                return self._cache[key]
            self.misses += 1
            return None

    def put(self, model_version: str, normalized_query: str, value: Any) -> None:
        key = (model_version, normalized_query)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = value
            if len(self._cache) > self.maxsize:
                self._cache.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self.hits = 0
            self.misses = 0

    @property
    def hit_ratio(self) -> float:
        total = self.hits + self.misses
        return round(self.hits / total, 4) if total > 0 else 0.0


__all__ = ["QueryEmbeddingCache"]
