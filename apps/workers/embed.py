from __future__ import annotations

"""
Embed worker: MoViNet (video lane) + librosa (audio lane).
Outputs: movinet_features.npy, audio_features.json
"""

import json
import os

import librosa
import numpy as np

from packages.storage.base import StorageBackend


def run(video_hash: str, storage: StorageBackend, ratio: int | None = None) -> dict:
    base = storage.video_dir(video_hash)
    movinet_key = f"{base}/movinet_features.npy"
    audio_features_key = f"{base}/audio_features.json"
    scenes_key = f"{base}/scenes.json"
    audio_key = f"{base}/audio.wav"

    scenes = json.loads(storage.read_text(scenes_key))

    # Audio features (librosa)
    if not storage.exists(audio_features_key):
        audio_path = str(storage.local_path(audio_key))
        af = _extract_audio_features(audio_path, scenes)
        storage.write_text(audio_features_key, json.dumps(af))

    # MoViNet visual embeddings — skip when ratio is known and below threshold
    skip_below = int(os.environ.get("SKIP_VISUAL_EMBEDDINGS_BELOW_RATIO", "3"))
    skip_movinet = ratio is not None and ratio <= skip_below
    if skip_movinet:
        import logging
        logging.getLogger(__name__).info(
            f"[embed] skipping MoViNet — ratio={ratio} <= skip_below={skip_below}"
        )
    elif not storage.exists(movinet_key):
        frames_index_key = f"{base}/frames/index.json"
        if storage.exists(frames_index_key):
            frame_paths = json.loads(storage.read_text(frames_index_key))
            embeddings = _extract_movinet_features(frame_paths)
            buf = np.ndarray.tobytes(embeddings)
            storage.write_bytes(movinet_key, buf)

    return {
        "movinet_key": movinet_key,
        "audio_features_key": audio_features_key,
    }


def _extract_audio_features(audio_path: str, scenes: list[dict]) -> list[dict]:
    y, sr = librosa.load(audio_path, sr=16000, mono=True)

    # Pre-check: skip silent audio
    if np.mean(librosa.feature.rms(y=y)) < 0.01:
        return [_silence_features(s) for s in scenes]

    rms = librosa.feature.rms(y=y)[0]
    # F0 estimation
    f0, voiced_flag, _ = librosa.pyin(y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"))

    hop_length = 512
    frame_to_time = lambda f: librosa.frames_to_time(f, sr=sr, hop_length=hop_length)

    result = []
    for scene in scenes:
        start_f = librosa.time_to_frames(scene["start_s"], sr=sr, hop_length=hop_length)
        end_f = librosa.time_to_frames(scene["end_s"], sr=sr, hop_length=hop_length)
        end_f = min(end_f, len(rms) - 1)

        scene_rms = rms[start_f:end_f + 1] if end_f >= start_f else np.array([0.0])
        scene_f0 = f0[start_f:end_f + 1] if f0 is not None and end_f >= start_f else np.array([0.0])
        scene_f0 = scene_f0[~np.isnan(scene_f0)]

        max_rms = float(np.max(rms)) or 1.0
        result.append({
            "scene_index": scene["scene_index"],
            "start_s": scene["start_s"],
            "end_s": scene["end_s"],
            "rms_energy": float(np.mean(scene_rms)) / max_rms,
            "pitch_variance": float(np.std(scene_f0)) / 100.0 if len(scene_f0) > 1 else 0.0,
            "speech_rate_wpm": 0.0,  # filled by segment worker using transcript
            "silence_ratio": float(np.mean(scene_rms < 0.01)),
        })
    return result


def _silence_features(scene: dict) -> dict:
    return {
        "scene_index": scene["scene_index"],
        "start_s": scene["start_s"],
        "end_s": scene["end_s"],
        "rms_energy": 0.0,
        "pitch_variance": 0.0,
        "speech_rate_wpm": 0.0,
        "silence_ratio": 1.0,
    }


def _extract_movinet_features(frame_paths: list[str]) -> np.ndarray:
    from packages.ml.movinet import load_model, embed_frames
    import cv2

    model = load_model()
    frames = []
    for path in frame_paths:
        img = cv2.imread(path)
        if img is None:
            frames.append(np.zeros((172, 172, 3), dtype=np.float32))
            continue
        img = cv2.resize(img, (172, 172))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        frames.append(img)

    return embed_frames(frames, model)
