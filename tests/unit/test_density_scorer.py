from __future__ import annotations

import pytest

from packages.core.density.scorer import compute_density_score
from packages.core.models import DensitySignals


def _signals(**overrides) -> DensitySignals:
    defaults = dict(
        semantic_redundancy=0.5,
        silence_ratio=0.1,
        filler_word_rate=0.1,
        pacing_variance=0.2,
        lexical_density=0.6,
        topic_count=5,
        visual_change_rate=0.3,
    )
    defaults.update(overrides)
    return DensitySignals(**defaults)


def test_score_range():
    result = compute_density_score(_signals())
    assert 0.0 <= result.score <= 1.0


def test_high_redundancy_low_score():
    """Very redundant content → low density score → high ratio recommended."""
    result = compute_density_score(_signals(
        semantic_redundancy=0.95,
        silence_ratio=0.6,
        filler_word_rate=0.8,
        lexical_density=0.2,
        topic_count=1,
        visual_change_rate=0.0,
    ))
    assert result.score < 0.3
    assert result.recommended_ratio is not None
    assert result.recommended_ratio >= 5


def test_dense_content_no_compression():
    """Dense content → score > 0.80 → no ratio recommended."""
    result = compute_density_score(_signals(
        semantic_redundancy=0.0,
        silence_ratio=0.0,
        filler_word_rate=0.0,
        lexical_density=0.95,
        topic_count=15,
        visual_change_rate=0.9,
    ))
    assert result.score >= 0.80
    assert result.recommended_ratio is None
    assert result.message is not None


def test_ratio_bands():
    """Each density band maps to expected ratio."""
    # 0.60-0.80 → 2x
    r = compute_density_score(_signals(
        semantic_redundancy=0.0, silence_ratio=0.0, filler_word_rate=0.0,
        lexical_density=0.75, topic_count=8, visual_change_rate=0.5,
    ), video_duration_s=1800)
    assert r.recommended_ratio == 2

    # 0.0-0.20 → 9x
    r2 = compute_density_score(_signals(
        semantic_redundancy=0.9, silence_ratio=0.7, filler_word_rate=0.8,
        lexical_density=0.1, topic_count=1, visual_change_rate=0.0,
    ), video_duration_s=1800)
    assert r2.recommended_ratio == 9
