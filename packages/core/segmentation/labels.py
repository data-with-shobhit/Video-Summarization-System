from __future__ import annotations

import os

from packages.ml.nvidianim import label_all_segments


def maybe_label_segments(segments: list[dict]) -> list[dict]:
    """
    Add chapter labels to topic segments if SEGMENT_LABELING_ENABLED=true.
    Labels appear as title cards in rendered output.
    """
    enabled = os.environ.get("SEGMENT_LABELING_ENABLED", "true").lower() == "true"
    if not enabled:
        for i, seg in enumerate(segments):
            seg["label"] = f"Segment {i + 1}"
        return segments
    return label_all_segments(segments)
