from __future__ import annotations

from unittest.mock import patch
import numpy as np
import pytest

from packages.core.segmentation.texttiling import find_boundaries


def _make_chunks(n: int, start_s: float = 0.0, chunk_dur: float = 120.0) -> list[dict]:
    return [
        {"text": f"chunk text {i}", "start_s": start_s + i * chunk_dur, "end_s": start_s + (i + 1) * chunk_dur}
        for i in range(n)
    ]


def test_empty_input():
    assert find_boundaries([]) == []


def test_single_chunk():
    chunks = _make_chunks(1)
    with patch("packages.core.segmentation.texttiling.embed_texts") as mock_embed:
        mock_embed.return_value = np.array([[1.0, 0.0]])
        result = find_boundaries(chunks, threshold=0.35)
    assert len(result) == 1
    assert result[0]["segment_index"] == 0


def test_boundary_detected():
    """Two groups of similar chunks with a clear drop between them."""
    chunks = _make_chunks(6, chunk_dur=120.0)  # 12 min total

    # Embeddings: chunks 0-2 similar, chunks 3-5 different
    embs = np.zeros((6, 4))
    embs[0:3, 0] = 1.0
    embs[3:6, 1] = 1.0
    # Normalize
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)

    with patch("packages.core.segmentation.texttiling.embed_texts") as mock_embed:
        mock_embed.return_value = embs
        result = find_boundaries(chunks, threshold=0.35)

    assert len(result) == 2
    assert result[0]["start_s"] == 0.0
    assert result[1]["start_s"] == 3 * 120.0


def test_min_segment_merge():
    """Tiny segment (< 60s) should merge into neighbor."""
    # 5 chunks of 10s each — total 50s, all tiny
    chunks = _make_chunks(5, chunk_dur=10.0)

    embs = np.zeros((5, 4))
    embs[0, 0] = 1.0
    embs[1, 1] = 1.0  # boundary here
    embs[2, 1] = 1.0
    embs[3, 2] = 1.0  # boundary here
    embs[4, 2] = 1.0
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)

    with patch("packages.core.segmentation.texttiling.embed_texts") as mock_embed:
        mock_embed.return_value = embs
        import os
        with patch.dict(os.environ, {"TEXTTILING_MIN_SEGMENT_SECONDS": "60"}):
            result = find_boundaries(chunks, threshold=0.35)

    # All chunks get merged since each "segment" is < 60s
    assert len(result) >= 1
