from __future__ import annotations

import os
import time
from typing import Any

# In-memory dict (POC). Redis via CACHE_BACKEND=redis.
_store: dict[str, tuple[Any, float]] = {}


def get(key: str) -> Any | None:
    if os.environ.get("CACHE_BACKEND", "memory") == "redis":
        return _redis_get(key)
    entry = _store.get(key)
    if entry and time.time() < entry[1]:
        return entry[0]
    return None


def set(key: str, value: Any, ttl: int = 3600) -> None:
    if os.environ.get("CACHE_BACKEND", "memory") == "redis":
        _redis_set(key, value, ttl)
        return
    _store[key] = (value, time.time() + ttl)


def delete(key: str) -> None:
    if os.environ.get("CACHE_BACKEND", "memory") == "redis":
        _redis_delete(key)
        return
    _store.pop(key, None)


# ── Redis backend ─────────────────────────────────────────────────────────────

_redis_client = None


def _get_redis():
    global _redis_client
    if _redis_client is None:
        import redis
        _redis_client = redis.from_url(os.environ["REDIS_URL"])
    return _redis_client


def _redis_get(key: str) -> Any | None:
    import pickle
    raw = _get_redis().get(key)
    return pickle.loads(raw) if raw else None


def _redis_set(key: str, value: Any, ttl: int) -> None:
    import pickle
    _get_redis().setex(key, ttl, pickle.dumps(value))


def _redis_delete(key: str) -> None:
    _get_redis().delete(key)
