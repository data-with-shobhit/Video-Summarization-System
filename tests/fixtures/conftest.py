"""Shared fixtures for all tests."""
from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def sample_scenes():
    return [
        {"scene_index": i, "start_s": i * 60.0, "end_s": (i + 1) * 60.0, "duration_s": 60.0}
        for i in range(10)
    ]


@pytest.fixture
def sample_audio_features(sample_scenes):
    return [
        {
            "scene_index": s["scene_index"],
            "start_s": s["start_s"],
            "end_s": s["end_s"],
            "rms_energy": 0.5,
            "pitch_variance": 0.3,
            "speech_rate_wpm": 130.0,
            "silence_ratio": 0.05,
        }
        for s in sample_scenes
    ]


@pytest.fixture
def sample_topic_segments():
    return [
        {"segment_index": 0, "start_s": 0.0, "end_s": 300.0, "chunk_indices": list(range(5)), "text": "intro content", "label": "Introduction"},
        {"segment_index": 1, "start_s": 300.0, "end_s": 600.0, "chunk_indices": list(range(5, 10)), "text": "main content", "label": "Core Concepts"},
    ]


@pytest.fixture
def sample_movinet_features():
    return np.random.rand(600, 600).astype(np.float32)
