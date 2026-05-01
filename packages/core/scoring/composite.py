from __future__ import annotations

import numpy as np

from packages.core.models import SegmentScore


def build_composite_scores(
    scenes: list[dict],
    llm_scores: dict[int, dict],
    audio_features: list[dict],
    movinet_features: np.ndarray | None = None,
) -> list[SegmentScore]:
    """
    composite = 0.45 * llm_score + 0.30 * audio_emphasis + 0.25 * visual_salience

    scenes: [{scene_index, start_s, end_s, ...}]
    llm_scores: {scene_index: {llm_score: float, reason: str}}
    audio_features: [{scene_index, rms_energy, pitch_variance, ...}]
    movinet_features: (N_frames, 600) — optional, used for visual salience
    """
    audio_map = {f["scene_index"]: f for f in audio_features}

    # Pre-compute visual salience per scene if movinet available
    visual_salience_map: dict[int, float] = {}
    if movinet_features is not None and len(movinet_features) > 1:
        diffs = np.linalg.norm(np.diff(movinet_features, axis=0), axis=1)
        # Each diff[i] = change between frame i and i+1
        # Map frames → scenes by scene index (assume 1fps, scene_index ~ start_s)
        for scene in scenes:
            start_f = int(scene["start_s"])
            end_f = int(scene["end_s"])
            if start_f >= len(diffs):
                visual_salience_map[scene["scene_index"]] = 0.0
                continue
            end_f = min(end_f, len(diffs))
            visual_salience_map[scene["scene_index"]] = float(np.mean(diffs[start_f:end_f])) if end_f > start_f else 0.0

        # Normalize to 0-1
        max_v = max(visual_salience_map.values()) or 1.0
        visual_salience_map = {k: v / max_v for k, v in visual_salience_map.items()}

    results = []
    for scene in scenes:
        idx = scene["scene_index"]
        llm_data = llm_scores.get(idx, {})
        llm_s = float(llm_data.get("llm_score", 5.0)) / 10.0  # normalize to 0-1

        af = audio_map.get(idx, {})
        audio_emphasis = _compute_audio_emphasis(af)

        visual_salience = visual_salience_map.get(idx, 0.5)

        composite = (
            0.45 * llm_s +
            0.30 * audio_emphasis +
            0.25 * visual_salience
        )

        results.append(SegmentScore(
            id=idx,
            score=round(composite * 10, 2),  # back to 0-10 range
            reason=llm_data.get("reason"),
            llm_score=round(llm_s * 10, 2),
            audio_emphasis=round(audio_emphasis, 4),
            visual_salience=round(visual_salience, 4),
            composite_score=round(composite, 4),
        ))

    return results


def _compute_audio_emphasis(af: dict) -> float:
    """Combine rms_energy + pitch_variance into 0-1 emphasis score."""
    if not af:
        return 0.5
    # Both already 0-1 normalized from librosa worker
    rms = float(af.get("rms_energy", 0.5))
    pitch_var = float(af.get("pitch_variance", 0.5))
    return float(np.clip((rms + pitch_var) / 2.0, 0.0, 1.0))
