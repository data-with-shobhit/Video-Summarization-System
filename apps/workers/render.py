from __future__ import annotations

"""Render worker: ffmpeg stitch → output_{ratio}.mp4"""

import json
import os
import subprocess
import tempfile
from pathlib import Path

from packages.storage.base import StorageBackend


def run(video_hash: str, ratio: int, storage: StorageBackend) -> str:
    base = storage.video_dir(video_hash)
    output_key = f"{base}/output_{ratio}.mp4"

    if storage.exists(output_key):
        return output_key

    selection = json.loads(storage.read_text(f"{base}/selection_{ratio}.json"))
    norm_path = str(storage.local_path(f"{base}/normalized.mp4"))
    output_path = str(storage.local_path(output_key))

    add_crossfades = os.environ.get("RENDER_ADD_CROSSFADES", "false").lower() == "true"
    # Stream copy is fast (~5s) but causes frame flickering and lip-sync drift at cut points.
    # Default is re-encode with veryfast preset (~15s) — clean cuts, no artifacts.
    use_stream_copy = os.environ.get("RENDER_USE_STREAM_COPY", "false").lower() == "true"

    if not add_crossfades and use_stream_copy:
        _render_stream_copy(selection, norm_path, output_path)
    else:
        _render_reencode(selection, norm_path, output_path)

    return output_key


def _render_stream_copy(
    selection: list[dict], input_path: str, output_path: str
) -> None:
    """Fast path: -c copy, ~5 sec. No re-encode."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    # ffmpeg concat demuxer requires forward slashes even on Windows
    safe_input = input_path.replace("\\", "/")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        concat_file = f.name
        for seg in selection:
            f.write(f"file '{safe_input}'\n")
            f.write(f"inpoint {seg['start_s']}\n")
            f.write(f"outpoint {seg['end_s']}\n")

    try:
        result = subprocess.run([
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", concat_file,
            "-c", "copy",
            output_path
        ], capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed:\n{result.stderr[-2000:]}")
    finally:
        os.unlink(concat_file)


def _render_reencode(
    selection: list[dict], input_path: str, output_path: str
) -> None:
    """Re-encode path for crossfades. ~45 sec. Only when user requests crossfades."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    crossfade_ms = int(os.environ.get("RENDER_CROSSFADE_MS", "200"))

    safe_input = input_path.replace("\\", "/")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        concat_file = f.name
        for seg in selection:
            f.write(f"file '{safe_input}'\n")
            f.write(f"inpoint {seg['start_s']}\n")
            f.write(f"outpoint {seg['end_s']}\n")

    preset = os.environ.get("RENDER_PRESET", "veryfast")
    crf = os.environ.get("RENDER_CRF", "23")
    try:
        result = subprocess.run([
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", concat_file,
            "-vsync", "cfr",           # constant frame rate — prevents flicker
            "-c:v", "libx264", "-preset", preset, "-crf", crf,
            "-c:a", "aac", "-ar", "44100",
            "-async", "1",             # resample audio to fix lip-sync drift at cuts
            output_path
        ], capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed:\n{result.stderr[-2000:]}")
    finally:
        os.unlink(concat_file)
