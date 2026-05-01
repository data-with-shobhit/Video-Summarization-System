from __future__ import annotations

from packages.ml.nvidianim import score_segments


def score_via_llm(
    segments: list[dict],
    video_type: str,
    ratio: int
) -> dict[int, dict]:
    """
    Call GLM-4.7 scoring via NVIDIA NIM. Returns {segment_id: {"score": float, "reason": str}}.
    Uses thinking mode for ratio >= 7.
    """
    raw = score_segments(segments, video_type, ratio)
    return {
        item["id"]: {
            "llm_score": float(item.get("score", 5)),
            "reason": item.get("reason", "")
        }
        for item in raw
    }
