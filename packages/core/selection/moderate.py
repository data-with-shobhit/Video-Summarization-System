from __future__ import annotations

"""Moderate (4-6x): submodular greedy at scene level."""

import numpy as np

from packages.core.models import SelectionSegment


def select_moderate(
    scenes: list[dict],
    composite_scores: list,
    target_ratio: float,
    diversity_threshold: float = 0.82,
) -> list[SelectionSegment]:
    """
    Greedy submodular selection:
      composite_score = 0.45*llm + 0.30*audio + 0.25*visual
    Skip if: duration would exceed target OR too similar to already-selected.
    Return in chronological order.
    """
    total_dur = sum(s["end_s"] - s["start_s"] for s in scenes)
    target_dur = total_dur / target_ratio

    score_map = {s.id: s for s in composite_scores}
    # Sort descending by composite score
    sorted_scenes = sorted(
        scenes,
        key=lambda s: score_map.get(s["scene_index"], type("", (), {"composite_score": 0.0})()).composite_score or 0.0,
        reverse=True
    )

    selected = []
    selected_dur = 0.0
    selected_embs = []

    for scene in sorted_scenes:
        dur = scene["end_s"] - scene["start_s"]
        if selected_dur + dur > target_dur * 1.05:
            continue

        # Diversity check
        emb = scene.get("embedding")
        if emb is not None and selected_embs:
            sims = [float(np.dot(emb, se)) for se in selected_embs]
            if max(sims) > diversity_threshold:
                continue

        selected.append(scene)
        selected_dur += dur
        if emb is not None:
            selected_embs.append(emb)

    # Restore chronological order
    selected.sort(key=lambda s: s["start_s"])

    return [
        SelectionSegment(
            start_s=s["start_s"],
            end_s=s["end_s"],
            scene_index=s["scene_index"]
        )
        for s in selected
    ]
