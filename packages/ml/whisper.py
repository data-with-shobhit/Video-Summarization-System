from __future__ import annotations

import math
import os
import subprocess
import tempfile
import time
from datetime import date
from pathlib import Path

from groq import Groq, RateLimitError

from apps.logger import get_logger
from packages.core.models import Transcript, TranscriptSegment, Word

log = get_logger("ml.groq.whisper")

# Free tier hard limits
_FREE_DAILY_LIMIT_S  = 28_800   # 8 hours audio / day
_FREE_HOURLY_LIMIT_S = 7_200    # 2 hours audio / hour
_WARN_AT_PCT         = 0.80     # warn at 80% used

# In-memory usage tracker (resets on restart — good enough for POC)
_usage: dict[str, float] = {"date": "", "day_s": 0.0, "hour_s": 0.0, "hour_ts": 0.0}

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        key = os.environ.get("GROQ_API_KEY", "")
        if not key:
            log.error("GROQ_API_KEY not set")
            raise RuntimeError("GROQ_API_KEY missing")
        _client = Groq(api_key=key)
        log.info("Groq client initialised")
    return _client


def _track_usage(audio_s: float) -> None:
    today = str(date.today())
    now = time.time()

    # Reset daily counter on new day
    if _usage["date"] != today:
        _usage["date"] = today
        _usage["day_s"] = 0.0
        log.info(f"[GROQ] daily usage counter reset for {today}")

    # Reset hourly counter after 1 hour
    if now - _usage["hour_ts"] > 3600:
        _usage["hour_s"] = 0.0
        _usage["hour_ts"] = now

    _usage["day_s"] += audio_s
    _usage["hour_s"] += audio_s

    day_pct  = _usage["day_s"]  / _FREE_DAILY_LIMIT_S  * 100
    hour_pct = _usage["hour_s"] / _FREE_HOURLY_LIMIT_S * 100

    log.info(
        f"[GROQ] usage | "
        f"today={_usage['day_s']:.0f}s/{_FREE_DAILY_LIMIT_S}s ({day_pct:.1f}%) | "
        f"this_hour={_usage['hour_s']:.0f}s/{_FREE_HOURLY_LIMIT_S}s ({hour_pct:.1f}%)"
    )

    if _usage["day_s"] >= _FREE_DAILY_LIMIT_S * _WARN_AT_PCT:
        log.warning(
            f"[GROQ] ⚠ approaching daily limit — "
            f"{_usage['day_s']:.0f}/{_FREE_DAILY_LIMIT_S}s used ({day_pct:.1f}%)"
        )
    if _usage["hour_s"] >= _FREE_HOURLY_LIMIT_S * _WARN_AT_PCT:
        log.warning(
            f"[GROQ] ⚠ approaching hourly limit — "
            f"{_usage['hour_s']:.0f}/{_FREE_HOURLY_LIMIT_S}s used ({hour_pct:.1f}%)"
        )


def _get_audio_duration(audio_path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
        capture_output=True, text=True, check=True
    )
    return float(result.stdout.strip())


def _is_silent(audio_path: str, threshold: float = 0.01) -> bool:
    """Return True if chunk is near-silent — Groq would hallucinate on it."""
    try:
        import librosa
        import numpy as np
        y, _ = librosa.load(audio_path, sr=16000, mono=True, duration=30)
        return float(np.mean(librosa.feature.rms(y=y))) < threshold
    except Exception:
        return False


def chunk_audio(audio_path: str, max_minutes: int | None = None) -> list[tuple[str, float]]:
    if max_minutes is None:
        max_minutes = int(os.environ.get("GROQ_AUDIO_CHUNK_MINUTES", "10"))
    min_chunk_s = int(os.environ.get("GROQ_MIN_CHUNK_SECONDS", "600"))
    overlap_s = int(os.environ.get("GROQ_CHUNK_OVERLAP_SECONDS", "30"))
    max_chunk_s = max_minutes * 60

    total_s = _get_audio_duration(audio_path)
    log.debug(f"audio duration: {total_s:.1f}s | max_chunk: {max_chunk_s}s | overlap: {overlap_s}s")

    if total_s <= max_chunk_s:
        log.debug("audio fits in one chunk — no splitting needed")
        return [(audio_path, 0.0)]

    n_chunks = math.ceil(total_s / max_chunk_s)
    chunk_s = max(total_s / n_chunks, min_chunk_s)
    log.info(f"splitting audio into {n_chunks} chunks of ~{chunk_s:.0f}s each (overlap={overlap_s}s)")

    chunks = []
    tmp_dir = tempfile.mkdtemp()
    offset = 0.0
    i = 0
    while offset < total_s:
        # Extend chunk by overlap (except last chunk)
        is_last = (offset + chunk_s >= total_s)
        duration = chunk_s if is_last else chunk_s + overlap_s
        out = os.path.join(tmp_dir, f"chunk_{i:02d}.wav")
        subprocess.run(
            ["ffmpeg", "-y", "-i", audio_path,
             "-ss", str(offset), "-t", str(duration),
             "-ar", "16000", "-ac", "1", out],
            capture_output=True, check=True
        )
        size_kb = Path(out).stat().st_size // 1024
        log.debug(f"chunk {i}: offset={offset:.1f}s duration={duration:.0f}s size={size_kb}KB")
        chunks.append((out, offset))
        offset += chunk_s  # advance by chunk_s, not chunk_s+overlap
        i += 1

    return chunks


def _transcribe_chunk_with_retry(
    client: Groq,
    chunk_path: str,
    model: str,
    chunk_num: int,
    total_chunks: int,
    max_retries: int = 4,
) -> object:
    """Submit one chunk to Groq with exponential backoff on rate limit."""
    delay = 5
    for attempt in range(1, max_retries + 1):
        try:
            with open(chunk_path, "rb") as f:
                result = client.audio.transcriptions.create(
                    file=f,
                    model=model,
                    response_format="verbose_json",
                    timestamp_granularities=["word", "segment"]
                )
            return result
        except RateLimitError as e:
            if attempt == max_retries:
                log.error(
                    f"[GROQ] chunk {chunk_num}/{total_chunks} rate limited — "
                    f"all {max_retries} retries exhausted | error={e}"
                )
                raise
            log.warning(
                f"[GROQ] chunk {chunk_num}/{total_chunks} rate limited — "
                f"retry {attempt}/{max_retries} in {delay}s | error={e}"
            )
            time.sleep(delay)
            delay *= 2   # 5 → 10 → 20 → 40
        except Exception as e:
            log.error(f"[GROQ] chunk {chunk_num}/{total_chunks} failed | attempt={attempt} | error={e}")
            raise


def transcribe(audio_path: str) -> Transcript:
    client = _get_client()
    model = os.environ.get("GROQ_WHISPER_MODEL", "whisper-large-v3")
    chunks = chunk_audio(audio_path)

    # Check daily limit before sending
    audio_total_s = _get_audio_duration(audio_path)
    remaining_s = _FREE_DAILY_LIMIT_S - _usage["day_s"]
    if audio_total_s > remaining_s:
        log.error(
            f"[GROQ] daily limit would be exceeded — "
            f"audio={audio_total_s:.0f}s remaining={remaining_s:.0f}s"
        )
        raise RuntimeError(
            f"Groq free tier daily limit reached. "
            f"Used {_usage['day_s']:.0f}/{_FREE_DAILY_LIMIT_S}s today. "
            f"Resets at midnight."
        )

    log.info(
        f"[GROQ] starting transcription | model={model} | "
        f"chunks={len(chunks)} | audio={audio_total_s:.0f}s"
    )

    all_segments: list[TranscriptSegment] = []
    language = "en"
    seg_id = 0

    overlap_s = int(os.environ.get("GROQ_CHUNK_OVERLAP_SECONDS", "30"))
    # Track the absolute end time of the last accepted segment for dedup
    last_accepted_end_s = 0.0

    for i, (chunk_path, offset_s) in enumerate(chunks):
        chunk_size_kb = Path(chunk_path).stat().st_size // 1024
        chunk_dur_s = _get_audio_duration(chunk_path)

        # ── Silence skip — avoid Groq hallucination on silent chunks ──────────
        if _is_silent(chunk_path):
            log.info(f"[GROQ] chunk {i+1}/{len(chunks)} is silent — skipping (offset={offset_s:.1f}s)")
            continue

        log.info(
            f"[GROQ] sending chunk {i+1}/{len(chunks)} | "
            f"size={chunk_size_kb}KB | duration={chunk_dur_s:.0f}s | offset={offset_s:.1f}s"
        )

        t0 = time.time()
        result = _transcribe_chunk_with_retry(client, chunk_path, model, i+1, len(chunks))
        elapsed = round(time.time() - t0, 2)

        language = getattr(result, "language", "en")
        n_segs = len(result.segments) if result.segments else 0
        log.info(
            f"[GROQ] chunk {i+1} OK | elapsed={elapsed}s | "
            f"language={language} | segments={n_segs}"
        )

        _track_usage(chunk_dur_s)

        for seg in result.segments:
            _g = (lambda o, k, d=None: o.get(k, d) if isinstance(o, dict) else getattr(o, k, d))
            seg_start_abs = _g(seg, "start", 0.0) + offset_s
            seg_end_abs = _g(seg, "end", 0.0) + offset_s

            # ── Overlap dedup — skip segments that fall inside previous chunk's overlap zone
            if i > 0 and seg_start_abs < last_accepted_end_s - (overlap_s * 0.5):
                log.debug(f"[GROQ] dedup: skipping segment at {seg_start_abs:.1f}s (overlap zone)")
                continue

            words = []
            for w in (_g(seg, "words") or []):
                w_start = _g(w, "start", 0.0) + offset_s
                # Skip words in overlap zone of previous chunk
                if i > 0 and w_start < last_accepted_end_s - (overlap_s * 0.5):
                    continue
                words.append(Word(
                    word=_g(w, "word", ""),
                    start=w_start,
                    end=_g(w, "end", 0.0) + offset_s,
                ))
            all_segments.append(TranscriptSegment(
                id=seg_id,
                start=seg_start_abs,
                end=seg_end_abs,
                text=(_g(seg, "text") or "").strip(),
                words=words,
            ))
            last_accepted_end_s = max(last_accepted_end_s, seg_end_abs)
            seg_id += 1

    total_words = sum(len(s.text.split()) for s in all_segments)
    log.info(
        f"[GROQ] transcription complete | "
        f"language={language} | total_segments={len(all_segments)} | total_words={total_words}"
    )
    return Transcript(language=language, segments=all_segments)


def get_usage_status() -> dict:
    """Current free tier usage — call from health endpoint or UI."""
    day_pct  = _usage["day_s"]  / _FREE_DAILY_LIMIT_S  * 100
    hour_pct = _usage["hour_s"] / _FREE_HOURLY_LIMIT_S * 100
    return {
        "day_used_s":     round(_usage["day_s"]),
        "day_limit_s":    _FREE_DAILY_LIMIT_S,
        "day_pct":        round(day_pct, 1),
        "day_remaining_s": max(0, _FREE_DAILY_LIMIT_S - _usage["day_s"]),
        "hour_used_s":    round(_usage["hour_s"]),
        "hour_limit_s":   _FREE_HOURLY_LIMIT_S,
        "hour_pct":       round(hour_pct, 1),
    }
