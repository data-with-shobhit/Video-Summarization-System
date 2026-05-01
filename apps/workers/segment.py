from __future__ import annotations

"""
Segment worker: sentence-transformers + TextTiling → topic_segments.json
Also fills speech_rate_wpm into audio_features.json
"""

import json

from packages.core.segmentation.texttiling import find_boundaries, align_transcript_to_scenes
from packages.core.segmentation.labels import maybe_label_segments
from packages.storage.base import StorageBackend


def run(video_hash: str, storage: StorageBackend) -> str:
    base = storage.video_dir(video_hash)
    segments_key = f"{base}/topic_segments.json"

    if storage.exists(segments_key):
        return segments_key

    scenes = json.loads(storage.read_text(f"{base}/scenes.json"))
    transcript = json.loads(storage.read_text(f"{base}/transcript.json"))
    audio_features = json.loads(storage.read_text(f"{base}/audio_features.json"))

    # Align transcript to scenes
    chunks = align_transcript_to_scenes(transcript["segments"], scenes)

    # TextTiling boundary detection
    topic_segments = find_boundaries(chunks)

    # Add chapter labels (optional, DeepSeek Flash)
    topic_segments = maybe_label_segments(topic_segments)

    # Fill speech_rate_wpm from transcript into audio_features
    af_map = {f["scene_index"]: f for f in audio_features}
    for scene in scenes:
        idx = scene["scene_index"]
        words_in_scene = _count_words_in_scene(transcript["segments"], scene["start_s"], scene["end_s"])
        dur = scene["end_s"] - scene["start_s"]
        if idx in af_map and dur > 0:
            af_map[idx]["speech_rate_wpm"] = (words_in_scene / dur) * 60.0

    storage.write_text(f"{base}/audio_features.json", json.dumps(list(af_map.values())))
    storage.write_text(segments_key, json.dumps(topic_segments))

    return segments_key


def _count_words_in_scene(transcript_segments: list[dict], start_s: float, end_s: float) -> int:
    count = 0
    for seg in transcript_segments:
        if seg["end"] <= start_s:
            continue
        if seg["start"] >= end_s:
            break
        count += len(seg["text"].split())
    return count
