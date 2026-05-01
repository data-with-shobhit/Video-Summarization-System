from __future__ import annotations

"""Density worker: compute density score → density.json"""

import json

import numpy as np

from packages.core.density.scorer import compute_density_score
from packages.core.density.signals import (
    compute_silence_ratio, compute_filler_word_rate,
    compute_pacing_variance, compute_lexical_density,
)
from packages.core.models import DensitySignals
from packages.ml.embeddings import mean_pairwise_cosine_sim, embed_texts
from packages.ml.movinet import compute_visual_change_rate
from packages.storage.base import StorageBackend


def run(video_hash: str, storage: StorageBackend) -> dict:
    base = storage.video_dir(video_hash)
    density_key = f"{base}/density.json"

    if storage.exists(density_key):
        return json.loads(storage.read_text(density_key))

    transcript = json.loads(storage.read_text(f"{base}/transcript.json"))
    audio_features = json.loads(storage.read_text(f"{base}/audio_features.json"))
    topic_segments = json.loads(storage.read_text(f"{base}/topic_segments.json"))
    scenes = json.loads(storage.read_text(f"{base}/scenes.json"))

    full_text = " ".join(s["text"] for s in transcript["segments"])
    video_duration_s = scenes[-1]["end_s"] if scenes else 1800.0

    # Semantic redundancy from chunk embeddings
    chunk_texts = [seg.get("text", "") for seg in topic_segments]
    chunk_embs = embed_texts(chunk_texts) if chunk_texts else np.zeros((1, 384))
    semantic_redundancy = mean_pairwise_cosine_sim(chunk_embs)

    # Visual change rate from MoViNet
    movinet_key = f"{base}/movinet_features.npy"
    visual_change_rate = 0.0
    if storage.exists(movinet_key):
        raw = storage.read_bytes(movinet_key)
        embs = np.frombuffer(raw, dtype=np.float32).reshape(-1, 600)
        visual_change_rate = compute_visual_change_rate(embs)

    signals = DensitySignals(
        semantic_redundancy=semantic_redundancy,
        silence_ratio=compute_silence_ratio(audio_features),
        filler_word_rate=compute_filler_word_rate(full_text),
        pacing_variance=compute_pacing_variance(audio_features),
        lexical_density=compute_lexical_density(full_text),
        topic_count=len(topic_segments),
        visual_change_rate=visual_change_rate,
    )

    result = compute_density_score(signals, video_duration_s)
    result_dict = result.model_dump()
    storage.write_text(density_key, json.dumps(result_dict))

    return result_dict
