from __future__ import annotations

"""Score worker: GLM-4.7 (NVIDIA NIM) segment scoring → scores_{ratio}.json"""

import json

import numpy as np

from packages.core.scoring.llm import score_via_llm
from packages.core.scoring.composite import build_composite_scores
from packages.storage.base import StorageBackend


def run(video_hash: str, ratio: int, video_type: str, storage: StorageBackend) -> str:
    base = storage.video_dir(video_hash)
    scores_key = f"{base}/scores_{ratio}.json"

    if storage.exists(scores_key):
        return scores_key

    scenes = json.loads(storage.read_text(f"{base}/scenes.json"))
    audio_features = json.loads(storage.read_text(f"{base}/audio_features.json"))
    topic_segments = json.loads(storage.read_text(f"{base}/topic_segments.json"))

    # Build segment list for LLM (one per scene, text from transcript)
    transcript = json.loads(storage.read_text(f"{base}/transcript.json"))
    scene_texts = _build_scene_texts(scenes, transcript["segments"])
    segments_for_llm = [
        {"id": s["scene_index"], "text": scene_texts.get(s["scene_index"], ""), **s}
        for s in scenes
    ]

    # LLM scoring (skipped for 2-3x light band)
    if ratio <= 3:
        llm_scores = {s["scene_index"]: {"llm_score": 5.0, "reason": "light trim — rule-based, LLM not called"} for s in scenes}
    else:
        llm_scores = score_via_llm(segments_for_llm, video_type, ratio)

    # Load MoViNet features if available
    movinet_key = f"{base}/movinet_features.npy"
    movinet_features = None
    if storage.exists(movinet_key):
        raw = storage.read_bytes(movinet_key)
        movinet_features = np.frombuffer(raw, dtype=np.float32).reshape(-1, 600)

    composite = build_composite_scores(scenes, llm_scores, audio_features, movinet_features)
    scores_data = [s.model_dump() for s in composite]
    storage.write_text(scores_key, json.dumps(scores_data))

    return scores_key


def _build_scene_texts(scenes: list[dict], transcript_segments: list[dict]) -> dict[int, str]:
    result = {}
    for scene in scenes:
        words = []
        for seg in transcript_segments:
            if seg["end"] <= scene["start_s"]:
                continue
            if seg["start"] >= scene["end_s"]:
                break
            words.append(seg["text"])
        result[scene["scene_index"]] = " ".join(words).strip()
    return result
