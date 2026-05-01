from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class StorageBackend(ABC):
    """Abstract interface. Swap local↔S3 by changing STORAGE_BACKEND env var."""

    @abstractmethod
    def exists(self, key: str) -> bool: ...

    @abstractmethod
    def read_bytes(self, key: str) -> bytes: ...

    @abstractmethod
    def write_bytes(self, key: str, data: bytes) -> None: ...

    @abstractmethod
    def read_text(self, key: str) -> str: ...

    @abstractmethod
    def write_text(self, key: str, text: str) -> None: ...

    @abstractmethod
    def local_path(self, key: str) -> Path:
        """Return a local filesystem path suitable for ffmpeg/subprocess calls."""
        ...

    @abstractmethod
    def delete(self, key: str) -> None: ...

    def video_dir(self, video_hash: str) -> str:
        return f"videos/{video_hash}"

    def key(self, video_hash: str, filename: str) -> str:
        return f"videos/{video_hash}/{filename}"
