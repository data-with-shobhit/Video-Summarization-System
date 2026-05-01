from __future__ import annotations

"""Highlight reel (7-10x): topic-cluster representative selection."""

import math
import os

import numpy as np

from packages.core.models import SelectionSegment


def select_highlight(
    scenes: list[dict],
    topic_segments: list[dict],
    composite_scores: list,
    ratio: int,
) -> list[SelectionSegment]:
    """
    1. Use TextTiling segments as clusters
    2. Rank by sum of composite scores within segment
    3. Select top K: K = ceil(topic_count * (1 - (ratio-7)/10))
    4. Best scene within each cluster (highest composite score)
    5. Return in chronological order

    DeepSeek V4-Pro mandatory at this band — distinguishes intro vs repeat within cluster.
    """
    score_map = {s.id: s.composite_score or 0.0 for s in composite_scores}
    topic_count = len(topic_segments)

    # K = number of topic segments to keep
    k = math.ceil(topic_count * (1.0 - (ratio - 7) / 10.0))
    k = max(k, 1)

    # Score each topic segment by sum of its scenes' composite scores
    seg_scores = []
    for topic in topic_segments:
        t_scenes = [
            s for s in scenes
            if s["start_s"] >= topic["start_s"] and s["end_s"] <= topic["end_s"]
        ]
        total_score = sum(score_map.get(s["scene_index"], 0.0) for s in t_scenes)
        seg_scores.append((total_score, topic, t_scenes))

    seg_scores.sort(key=lambda x: x[0], reverse=True)
    top_segments = seg_scores[:k]

    # Best scene per top segment
    selected = []
    for _, topic, t_scenes in top_segments:
        if not t_scenes:
            continue
        best = max(t_scenes, key=lambda s: score_map.get(s["scene_index"], 0.0))
        selected.append(best)

    selected.sort(key=lambda s: s["start_s"])

    return [
        SelectionSegment(
            start_s=s["start_s"],
            end_s=s["end_s"],
            scene_index=s["scene_index"],
            topic_segment_index=next(
                (i for i, t in enumerate(topic_segments)
                 if s["start_s"] >= t["start_s"] and s["end_s"] <= t["end_s"]),
                None
            )
        )
        for s in selected
    ]
