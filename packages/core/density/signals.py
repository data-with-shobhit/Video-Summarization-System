from __future__ import annotations

import re
import numpy as np

FILLER_WORDS = {"um", "uh", "like", "you know", "basically", "literally", "actually", "so"}
_FILLER_RE = re.compile(r"\b(" + "|".join(re.escape(w) for w in FILLER_WORDS) + r")\b", re.I)


def compute_silence_ratio(audio_features: list[dict]) -> float:
    """Fraction of total duration that is silence (rms < 0.01)."""
    if not audio_features:
        return 0.0
    total_dur = sum(f["end_s"] - f["start_s"] for f in audio_features)
    if total_dur == 0:
        return 0.0
    silence_dur = sum(
        (f["end_s"] - f["start_s"]) * f.get("silence_ratio", 0.0)
        for f in audio_features
    )
    return float(np.clip(silence_dur / total_dur, 0.0, 1.0))


def compute_filler_word_rate(transcript_text: str) -> float:
    """Filler words per 100 words. 0.0 = none, higher = more filler."""
    words = transcript_text.split()
    if not words:
        return 0.0
    hits = len(_FILLER_RE.findall(transcript_text))
    rate = (hits / len(words)) * 100
    # normalize to 0-1 (cap at 20 fillers per 100 words = very bad)
    return float(np.clip(rate / 20.0, 0.0, 1.0))


def compute_pacing_variance(audio_features: list[dict]) -> float:
    """Std dev of WPM across scenes, normalized to 0-1."""
    wpms = [f.get("speech_rate_wpm", 0.0) for f in audio_features if f.get("speech_rate_wpm", 0) > 0]
    if len(wpms) < 2:
        return 0.0
    std = float(np.std(wpms))
    # normalize: 100 WPM std = very high variance → 1.0
    return float(np.clip(std / 100.0, 0.0, 1.0))


def compute_lexical_density(transcript_text: str) -> float:
    """Type-token ratio — unique words / total words. Higher = denser vocabulary."""
    words = [w.lower().strip(".,!?;:\"'") for w in transcript_text.split()]
    if not words:
        return 0.0
    return float(len(set(words)) / len(words))


def normalize_topic_count(topic_count: int, video_duration_s: float = 1800.0) -> float:
    """
    Normalize topic count relative to video duration.
    A 30-min video with 5 topics → moderate density (0.5).
    More topics per minute → higher density.
    """
    if video_duration_s <= 0:
        return 0.5
    topics_per_minute = topic_count / (video_duration_s / 60.0)
    # scale: 0.5 topics/min → 0.5, 1.0 topics/min → 1.0
    return float(np.clip(topics_per_minute / 1.0, 0.0, 1.0))
