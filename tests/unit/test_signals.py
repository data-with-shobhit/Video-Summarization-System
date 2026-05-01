from __future__ import annotations

from packages.core.density.signals import (
    compute_filler_word_rate, compute_lexical_density,
    compute_pacing_variance, compute_silence_ratio,
    normalize_topic_count,
)


def test_silence_ratio_full_silence():
    af = [{"scene_index": 0, "start_s": 0.0, "end_s": 10.0, "silence_ratio": 1.0}]
    assert compute_silence_ratio(af) == 1.0


def test_silence_ratio_no_silence():
    af = [{"scene_index": 0, "start_s": 0.0, "end_s": 10.0, "silence_ratio": 0.0}]
    assert compute_silence_ratio(af) == 0.0


def test_filler_word_rate_heavy():
    text = "um uh like you know basically um uh"
    rate = compute_filler_word_rate(text)
    assert rate > 0.0


def test_filler_word_rate_clean():
    text = "The quick brown fox jumps over the lazy dog"
    rate = compute_filler_word_rate(text)
    assert rate == 0.0


def test_lexical_density_all_unique():
    text = "alpha beta gamma delta epsilon"
    assert compute_lexical_density(text) == 1.0


def test_lexical_density_all_same():
    text = "the the the the the"
    assert compute_lexical_density(text) == pytest.approx(1 / 5)


def test_normalize_topic_count_clamp():
    assert normalize_topic_count(0) == 0.0
    assert normalize_topic_count(100, video_duration_s=60.0) == 1.0


import pytest
