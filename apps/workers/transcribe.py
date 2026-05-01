from __future__ import annotations

"""Transcribe worker: Groq Whisper → transcript.json"""

import json

from packages.ml.whisper import transcribe
from packages.storage.base import StorageBackend


def run(video_hash: str, audio_key: str, storage: StorageBackend) -> str:
    """
    Returns transcript_key.
    Idempotent — skips if transcript already exists.
    """
    base = storage.video_dir(video_hash)
    transcript_key = f"{base}/transcript.json"

    if storage.exists(transcript_key):
        return transcript_key

    audio_path = str(storage.local_path(audio_key))
    transcript = transcribe(audio_path)
    storage.write_text(transcript_key, transcript.model_dump_json(indent=2))

    return transcript_key
