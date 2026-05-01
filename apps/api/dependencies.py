from __future__ import annotations

import os
from functools import lru_cache

from packages.storage.base import StorageBackend


@lru_cache(maxsize=1)
def get_storage() -> StorageBackend:
    backend = os.environ.get("STORAGE_BACKEND", "local")
    if backend == "s3":
        from packages.storage.s3 import S3Storage
        return S3Storage()
    from packages.storage.local import LocalStorage
    return LocalStorage()
