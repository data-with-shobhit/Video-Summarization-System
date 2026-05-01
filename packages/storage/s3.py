from __future__ import annotations

import os
import tempfile
from pathlib import Path

from packages.storage.base import StorageBackend


class S3Storage(StorageBackend):
    """S3 storage backend. Production target. Keys identical to LocalStorage."""

    def __init__(self, bucket: str | None = None, region: str | None = None):
        import boto3
        self._bucket = bucket or os.environ["S3_BUCKET"]
        region = region or os.environ.get("S3_REGION", "ap-south-1")
        self._s3 = boto3.client("s3", region_name=region)
        self._tmp_dir = tempfile.mkdtemp()

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError
        try:
            self._s3.head_object(Bucket=self._bucket, Key=key)
            return True
        except ClientError:
            return False

    def read_bytes(self, key: str) -> bytes:
        obj = self._s3.get_object(Bucket=self._bucket, Key=key)
        return obj["Body"].read()

    def write_bytes(self, key: str, data: bytes) -> None:
        self._s3.put_object(Bucket=self._bucket, Key=key, Body=data)

    def read_text(self, key: str) -> str:
        return self.read_bytes(key).decode("utf-8")

    def write_text(self, key: str, text: str) -> None:
        self.write_bytes(key, text.encode("utf-8"))

    def local_path(self, key: str) -> Path:
        """Download to temp dir for ffmpeg/subprocess use."""
        local = Path(self._tmp_dir) / key.replace("/", "_")
        local.parent.mkdir(parents=True, exist_ok=True)
        if not local.exists():
            data = self.read_bytes(key)
            local.write_bytes(data)
        return local

    def delete(self, key: str) -> None:
        self._s3.delete_object(Bucket=self._bucket, Key=key)
