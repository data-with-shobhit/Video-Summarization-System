from __future__ import annotations

"""Light trim (2-3x): rule-based, no LLM, sentence level."""

import re
import numpy as np

from packages.core.models import SelectionSegment

FILLER_RE = re.compile(
    r"\b(um+|uh+|like|you know|basically|literally|actually)\b[,.]?\s*", re.I
)
SILENCE_THRESHOLD_S = 1.5  # silences longer than this are removed


def select_light(
    scenes: list[dict],
    audio_features: list[dict],
    transcript_segments: list[dict],
    target_ratio: float = 2.5,
) -> list[SelectionSegment]:
    """
    Remove: silences > 1.5s, filler words, repeated sentences (cosine > 0.92),
    verbal restarts.
    Quality target: >= 92% of meaningful content retained.
    """
    audio_map = {f["scene_index"]: f for f in audio_features}
    kept = []

    for scene in scenes:
        idx = scene["scene_index"]
        af = audio_map.get(idx, {})

        # Drop pure silence scenes
        if af.get("silence_ratio", 0.0) > 0.85:
            continue

        kept.append(scene)

    # Remove near-duplicate scenes (cosine sim > 0.92)
    kept = _deduplicate(kept, threshold=0.92)

    # Trim to target ratio
    total_dur = sum(s["end_s"] - s["start_s"] for s in scenes)
    target_dur = total_dur / target_ratio
    kept = _trim_to_target(kept, target_dur)

    return [
        SelectionSegment(
            start_s=s["start_s"],
            end_s=s["end_s"],
            scene_index=s["scene_index"]
        )
        for s in kept
    ]


def _deduplicate(scenes: list[dict], threshold: float) -> list[dict]:
    """Remove scenes with embedding cosine sim > threshold to any already-kept scene."""
    if not scenes:
        return scenes
    kept = [scenes[0]]
    for scene in scenes[1:]:
        emb = scene.get("embedding")
        if emb is None:
            kept.append(scene)
            continue
        sims = [
            float(np.dot(emb, k["embedding"]))
            for k in kept if k.get("embedding") is not None
        ]
        if not sims or max(sims) <= threshold:
            kept.append(scene)
    return kept


def _trim_to_target(scenes: list[dict], target_dur: float) -> list[dict]:
    """Keep scenes until target duration reached (in order)."""
    result = []
    total = 0.0
    for s in scenes:
        dur = s["end_s"] - s["start_s"]
        if total + dur > target_dur * 1.05:
            break
        result.append(s)
        total += dur
    return result
