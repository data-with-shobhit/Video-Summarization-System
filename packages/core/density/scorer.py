from __future__ import annotations

import numpy as np

from packages.core.models import DensityResult, DensitySignals
from packages.core.density.signals import normalize_topic_count


def compute_density_score(s: DensitySignals, video_duration_s: float = 1800.0) -> DensityResult:
    """
    Pure function. No I/O. Returns DensityResult with score + ratio recommendation.

    Score > 0.80 → too dense to compress → recommended_ratio = None
    """
    norm_topics = normalize_topic_count(s.topic_count, video_duration_s)

    compressibility = (
        0.30 * s.semantic_redundancy +
        0.20 * s.silence_ratio +
        0.15 * s.filler_word_rate
    )
    density = (
        0.20 * s.lexical_density +
        0.10 * norm_topics +
        0.05 * s.visual_change_rate
    )
    raw_score = float(np.clip(density - compressibility + 0.5, 0.0, 1.0))

    ratio, ci, message = _recommend_ratio(raw_score)

    return DensityResult(
        score=round(raw_score, 4),
        recommended_ratio=ratio,
        ratio_confidence_interval=ci,
        message=message,
        signals={
            "semantic_redundancy": round(s.semantic_redundancy, 4),
            "silence_ratio": round(s.silence_ratio, 4),
            "filler_word_rate": round(s.filler_word_rate, 4),
            "pacing_variance": round(s.pacing_variance, 4),
            "lexical_density": round(s.lexical_density, 4),
            "topic_count": s.topic_count,
            "visual_change_rate": round(s.visual_change_rate, 4),
        }
    )


def _recommend_ratio(score: float) -> tuple[int | None, int | None, str | None]:
    if score >= 0.80:
        return None, None, (
            "Content too dense for video compression. "
            "Consider a text summary instead."
        )
    if score >= 0.60:
        return 2, 1, None
    if score >= 0.40:
        return 3, 1, None   # center of 3-4× band
    if score >= 0.20:
        return 6, 2, None   # center of 5-7× band
    return 9, 2, None       # center of 8-10× band
