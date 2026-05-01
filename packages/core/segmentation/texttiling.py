from __future__ import annotations

import os
import numpy as np

from packages.ml.embeddings import embed_texts


def find_boundaries(chunks: list[dict], threshold: float | None = None) -> list[dict]:
    """
    Detect topic segment boundaries via cosine similarity drops.

    chunks: [{"text": str, "start_s": float, "end_s": float}, ...]
    threshold: cosine sim below this → topic change.
      - 0.35 for structured content (lectures, demos, tutorials)
      - 0.25 for casual content (interviews, podcasts)

    Returns topic segments: [{segment_index, start_s, end_s, chunk_indices, text}, ...]
    """
    if not chunks:
        return []

    if threshold is None:
        threshold = float(os.environ.get("TEXTTILING_THRESHOLD", "0.35"))

    min_seg_s = float(os.environ.get("TEXTTILING_MIN_SEGMENT_SECONDS", "60"))

    texts = [c["text"] for c in chunks]
    embs = embed_texts(texts)  # (N, 384) normalized

    # Find boundary indices where sim drops below threshold
    raw_boundaries = [0]
    for i in range(1, len(embs)):
        sim = float(np.dot(embs[i - 1], embs[i]))
        if sim < threshold:
            raw_boundaries.append(i)
    raw_boundaries.append(len(chunks))

    # Merge segments shorter than min_seg_s with neighbor
    segments = []
    for i in range(len(raw_boundaries) - 1):
        start_idx = raw_boundaries[i]
        end_idx = raw_boundaries[i + 1]
        seg_duration = chunks[end_idx - 1]["end_s"] - chunks[start_idx]["start_s"]

        if seg_duration < min_seg_s and segments:
            # Merge into previous segment
            prev = segments[-1]
            prev["end_s"] = chunks[end_idx - 1]["end_s"]
            prev["chunk_indices"].extend(range(start_idx, end_idx))
            prev["text"] += " " + " ".join(c["text"] for c in chunks[start_idx:end_idx])
        else:
            segments.append({
                "segment_index": len(segments),
                "start_s": chunks[start_idx]["start_s"],
                "end_s": chunks[end_idx - 1]["end_s"],
                "chunk_indices": list(range(start_idx, end_idx)),
                "text": " ".join(c["text"] for c in chunks[start_idx:end_idx])
            })

    # Re-index after merging
    for i, seg in enumerate(segments):
        seg["segment_index"] = i

    return segments


def align_transcript_to_scenes(
    transcript_segments: list[dict],
    scenes: list[dict]
) -> list[dict]:
    """
    Merge transcript words into one chunk per scene.
    Returns [{"text": str, "start_s": float, "end_s": float}, ...]
    """
    chunks = []
    for scene in scenes:
        s_start = scene["start_s"]
        s_end = scene["end_s"]
        words = []
        for seg in transcript_segments:
            if seg["end"] <= s_start:
                continue
            if seg["start"] >= s_end:
                break
            words.append(seg["text"])
        chunks.append({
            "text": " ".join(words).strip() or "[no speech]",
            "start_s": s_start,
            "end_s": s_end
        })
    return chunks
