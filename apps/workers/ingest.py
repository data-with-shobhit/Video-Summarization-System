from __future__ import annotations

"""
Ingest worker: normalize video, demux audio, sample frames.
Outputs: normalized.mp4, audio.wav, audio_chunk_NN.wav, frames/*.jpg, scenes.json
"""

import json
import logging
import os
import subprocess
from pathlib import Path

from scenedetect import open_video, SceneManager, ContentDetector

from packages.storage.base import StorageBackend

log = logging.getLogger(__name__)


def run(video_hash: str, source_path: str, storage: StorageBackend) -> dict:
    """
    Returns dict with paths to all artifacts keyed by artifact name.
    All artifacts stored via storage backend.
    """
    base = storage.video_dir(video_hash)

    norm_key = f"{base}/normalized.mp4"
    audio_key = f"{base}/audio.wav"
    scenes_key = f"{base}/scenes.json"

    # 1. Normalize to constant 30fps h264 + aac
    norm_path = str(storage.local_path(norm_key))
    _norm_valid = storage.exists(norm_key) and Path(norm_path).exists() and Path(norm_path).stat().st_size > 10_000
    if not _norm_valid:
        if Path(norm_path).exists():
            log.warning(f"[ingest] deleting corrupt/partial normalized.mp4 ({Path(norm_path).stat().st_size} bytes)")
            Path(norm_path).unlink()
        _ffmpeg_normalize(source_path, norm_path)
        if not storage.exists(norm_key):
            storage.write_bytes(norm_key, Path(norm_path).read_bytes())

    # 2. Extract 16kHz mono WAV
    if not storage.exists(audio_key):
        norm_path = str(storage.local_path(norm_key))
        audio_path = str(storage.local_path(audio_key))
        _extract_audio(norm_path, audio_path)

    # 3. Scene detection
    if not storage.exists(scenes_key):
        norm_path = str(storage.local_path(norm_key))
        scenes = _detect_scenes(norm_path)
        storage.write_text(scenes_key, json.dumps(scenes))
    else:
        scenes = json.loads(storage.read_text(scenes_key))

    # 4. Sample frames at 1fps
    frames_dir = f"{base}/frames"
    frames_index_key = f"{frames_dir}/index.json"
    if not storage.exists(frames_index_key):
        norm_path = str(storage.local_path(norm_key))
        frames_local_dir = str(storage.local_path(frames_dir))
        frame_paths = _sample_frames(norm_path, frames_local_dir)
        storage.write_text(frames_index_key, json.dumps(frame_paths))

    return {
        "normalized_key": norm_key,
        "audio_key": audio_key,
        "scenes_key": scenes_key,
        "frames_dir": frames_dir,
    }


def _ffmpeg_normalize(input_path: str, output_path: str) -> None:
    """Constant 30fps, h264 video, aac audio."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run([
        "ffmpeg", "-y", "-i", input_path,
        "-vf", "fps=30",
        "-vsync", "cfr",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-ar", "44100",
        output_path
    ], check=False, capture_output=True)
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace").strip()
        log.error(f"ffmpeg failed (exit {result.returncode}): {stderr[-2000:]}")
        raise subprocess.CalledProcessError(result.returncode, result.args, result.stderr)


def _extract_audio(video_path: str, audio_path: str) -> None:
    """16kHz mono WAV for Groq Whisper."""
    Path(audio_path).parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run([
        "ffmpeg", "-y", "-i", video_path,
        "-ar", "16000", "-ac", "1",
        audio_path
    ], check=False, capture_output=True)
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace").strip()
        log.error(f"ffmpeg failed (exit {result.returncode}): {stderr[-2000:]}")
        raise subprocess.CalledProcessError(result.returncode, result.args, result.stderr)


def _detect_scenes(video_path: str) -> list[dict]:
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=30.0))
    scene_manager.detect_scenes(video, show_progress=False)
    scene_list = scene_manager.get_scene_list()

    fps = video.frame_rate
    scenes = []
    for i, (start, end) in enumerate(scene_list):
        start_s = start.get_frames() / fps
        end_s = end.get_frames() / fps
        scenes.append({
            "scene_index": i,
            "start_s": round(start_s, 3),
            "end_s": round(end_s, 3),
            "duration_s": round(end_s - start_s, 3),
        })

    # fallback: treat whole video as one scene if no cuts detected
    if not scenes:
        total_frames = video.duration.get_frames()
        total_s = total_frames / fps if fps else 0
        scenes = [{"scene_index": 0, "start_s": 0.0, "end_s": round(total_s, 3), "duration_s": round(total_s, 3)}]

    return scenes


def _sample_frames(video_path: str, frames_dir: str) -> list[str]:
    """1fps JPEG frames. Returns list of local paths."""
    Path(frames_dir).mkdir(parents=True, exist_ok=True)
    output_pattern = os.path.join(frames_dir, "frame_%05d.jpg")
    result = subprocess.run([
        "ffmpeg", "-y", "-i", video_path,
        "-vf", "fps=1",
        "-q:v", "3",
        output_pattern
    ], check=False, capture_output=True)
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace").strip()
        log.error(f"ffmpeg failed (exit {result.returncode}): {stderr[-2000:]}")
        raise subprocess.CalledProcessError(result.returncode, result.args, result.stderr)

    return sorted(str(p) for p in Path(frames_dir).glob("frame_*.jpg"))
