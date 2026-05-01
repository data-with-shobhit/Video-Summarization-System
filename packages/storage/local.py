from __future__ import annotations

import os
from pathlib import Path

from packages.storage.base import StorageBackend


class LocalStorage(StorageBackend):
    """Local filesystem storage. POC default."""

    def __init__(self, base_path: str | None = None):
        root = base_path or os.environ.get("LOCAL_STORAGE_PATH", "./data")
        self._root = Path(root).resolve()  # absolute — prevents cwd-relative path bugs
        self._root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, key: str) -> Path:
        p = self._root / key
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def exists(self, key: str) -> bool:
        return self._resolve(key).exists()

    def read_bytes(self, key: str) -> bytes:
        return self._resolve(key).read_bytes()

    def write_bytes(self, key: str, data: bytes) -> None:
        self._resolve(key).write_bytes(data)

    def read_text(self, key: str) -> str:
        return self._resolve(key).read_text(encoding="utf-8")

    def write_text(self, key: str, text: str) -> None:
        self._resolve(key).write_text(text, encoding="utf-8")

    def local_path(self, key: str) -> Path:
        return self._resolve(key)

    def delete(self, key: str) -> None:
        p = self._resolve(key)
        if p.exists():
            p.unlink()
