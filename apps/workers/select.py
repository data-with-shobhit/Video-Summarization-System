from __future__ import annotations

"""Select worker: ratio-band selector → selection_{ratio}.json"""

import json

from packages.core.models import Ratioband, SegmentScore
from packages.core.selection.light import select_light
from packages.core.selection.moderate import select_moderate
from packages.core.selection.highlight import select_highlight
from packages.storage.base import StorageBackend


def _band(ratio: int) -> Ratioband:
    if ratio <= 3:
        return Ratioband.LIGHT
    if ratio <= 6:
        return Ratioband.MODERATE
    return Ratioband.HIGHLIGHT


def run(video_hash: str, ratio: int, storage: StorageBackend) -> str:
    base = storage.video_dir(video_hash)
    selection_key = f"{base}/selection_{ratio}.json"

    if storage.exists(selection_key):
        return selection_key

    scenes = json.loads(storage.read_text(f"{base}/scenes.json"))
    audio_features = json.loads(storage.read_text(f"{base}/audio_features.json"))
    topic_segments = json.loads(storage.read_text(f"{base}/topic_segments.json"))

    scores_key = f"{base}/scores_{ratio}.json"
    composite_scores = []
    if storage.exists(scores_key):
        raw_scores = json.loads(storage.read_text(scores_key))
        composite_scores = [SegmentScore(**s) for s in raw_scores]

    band = _band(ratio)

    if band == Ratioband.LIGHT:
        selection = select_light(scenes, audio_features, [], target_ratio=ratio)
    elif band == Ratioband.MODERATE:
        selection = select_moderate(scenes, composite_scores, target_ratio=ratio)
    else:
        selection = select_highlight(scenes, topic_segments, composite_scores, ratio)

    data = [s.model_dump() for s in selection]
    storage.write_text(selection_key, json.dumps(data))

    return selection_key
