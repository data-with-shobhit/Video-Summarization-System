from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any  # kept for _call_with_retry return type

from openai import OpenAI

from apps.logger import get_logger
import packages.storage.cache as _cache_store

log = get_logger("ml.nvidia.glm")

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        key = os.environ.get("NVIDIA_API_KEY", "")
        if not key:
            log.error("NVIDIA_API_KEY not set")
            raise RuntimeError("NVIDIA_API_KEY missing")
        _client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=key,
        )
        log.info("NVIDIA NIM client initialised (GLM-4.7)")
    return _client


def _cache_key(text: str, video_type: str, ratio: int) -> str:
    return "nim:" + hashlib.sha256(f"{text}|{video_type}|{ratio}".encode()).hexdigest()


def _cache_get(key: str) -> Any | None:
    return _cache_store.get(key)


def _cache_set(key: str, value: Any) -> None:
    ttl = int(os.environ.get("LLM_CACHE_TTL_SECONDS", "86400"))
    _cache_store.set(key, value, ttl=ttl)


_VIDEO_TYPE_GUIDANCE = {
    "lecture":    "Prioritise key concepts, definitions, and worked examples. Cut repetition and tangents.",
    "interview":  "Keep the sharpest exchanges and most revealing answers. Cut filler, long pauses, and off-topic detours.",
    "demo":       "Retain every product action and outcome. Cut setup chatter and repeated walkthroughs.",
    "podcast":    "Preserve the strongest insights and story moments. Cut small-talk, cross-talk, and tangents.",
    "tutorial":   "Keep every instructional step and its rationale. Cut meta-commentary and lengthy intros.",
    "short_film": "Preserve character-defining moments, turning points, and emotional beats. Cut any scene that does not advance character or plot.",
    "movie":      "Identify the core dramatic arc — inciting incident, escalation, climax, resolution. Keep pivotal scenes and cut subplots that don't serve the spine.",
    "episode":    "Retain the episode's main plot beats and character moments that matter to the series arc. Cut cold-open repetition, recap scenes, and filler B-plots.",
    "unknown":    "Identify the narrative spine and keep only segments that advance it. Cut filler and repetition.",
}


def _build_scoring_prompt(segments: list[dict], video_type: str, ratio: int) -> str:
    pct = round(100 / ratio)
    topic_summary = "; ".join(s.get("label", f"Segment {s['id']}") for s in segments[:10])
    segs_json = json.dumps(
        [{"id": s["id"], "text": s["text"][:300]} for s in segments], indent=2
    )
    guidance = _VIDEO_TYPE_GUIDANCE.get(video_type, _VIDEO_TYPE_GUIDANCE["unknown"])
    return (
        f"Content type: {video_type}. Target compression: {ratio}× (keep {pct}% of content).\n"
        f"Editorial guidance: {guidance}\n\n"
        f"Topics covered: {topic_summary}\n\n"
        f"Score each segment 0–10 (10 = essential to the story; 0 = pure filler). "
        f"Be ruthless — only the best {pct}% survives.\n\n"
        f"Output a JSON array. Each element:\n"
        f'- "id": segment id\n'
        f'- "score": integer 0–10\n'
        f'- "reason": one sentence max\n\n'
        f"Segments:\n{segs_json}"
    )


def score_segments(segments: list[dict], video_type: str, ratio: int) -> list[dict]:
    client = _get_client()
    model = os.environ.get("GLM_MODEL", "z-ai/glm4.7")
    use_thinking = ratio >= int(os.environ.get("GLM_PRO_RATIO_THRESHOLD", "7"))

    log.info(
        f"[NVIDIA] score_segments | model={model} | ratio={ratio}x | "
        f"thinking={use_thinking} | total_segments={len(segments)}"
    )

    results = []
    uncached = []
    for seg in segments:
        key = _cache_key(seg.get("text", ""), video_type, ratio)
        cached = _cache_get(key)
        if cached:
            log.debug(f"[NVIDIA] cache hit for segment id={seg.get('id')}")
            results.append(cached)
        else:
            uncached.append(seg)

    if not uncached:
        log.info(f"[NVIDIA] all {len(results)} segments served from cache")
        return results

    log.info(f"[NVIDIA] {len(uncached)} segments need scoring ({len(results)} from cache)")

    system = (
        "You are an award-winning film editor and content curator with 20 years of experience "
        "cutting documentaries, lectures, and online content. Your job: score transcript segments "
        "for importance so the best moments survive a compressed edit. Return only a JSON array."
    )
    if use_thinking:
        system += (
            " Think like a Sundance editor making a final cut — find the narrative spine, "
            "the emotional peaks, and the irreplaceable insights. Be precise and ruthless."
        )

    max_segs = int(os.environ.get("LLM_MAX_SEGMENTS_PER_CALL", "20"))
    batch_results = []

    for i in range(0, len(uncached), max_segs):
        batch = uncached[i: i + max_segs]
        prompt = _build_scoring_prompt(batch, video_type, ratio)
        batch_num = i // max_segs + 1
        total_batches = (len(uncached) + max_segs - 1) // max_segs

        log.info(
            f"[NVIDIA] batch {batch_num}/{total_batches} | "
            f"segments={len(batch)} | thinking={use_thinking}"
        )

        raw = _call_with_retry(client, model, system, prompt, use_thinking, batch_num, total_batches)

        parsed = _parse_scores(raw)
        if len(parsed) != len(batch):
            log.warning(
                f"[NVIDIA] score count mismatch: expected={len(batch)} got={len(parsed)}"
            )
        batch_results.extend(parsed)

        for seg, scored in zip(batch, parsed):
            key = _cache_key(seg.get("text", ""), video_type, ratio)
            _cache_set(key, scored)

    log.info(f"[NVIDIA] scoring complete | total_scored={len(batch_results)}")
    return results + batch_results


def _call_with_retry(client, model, system, prompt, use_thinking, batch_num, total_batches, max_retries=3) -> str:
    delay = 5
    for attempt in range(1, max_retries + 1):
        t0 = time.time()
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=2048,
                stream=True,
                extra_body={"chat_template_kwargs": {
                    "enable_thinking": use_thinking,
                    "clear_thinking": True,
                }},
            )
            raw = _collect_stream(stream)
            elapsed = round(time.time() - t0, 2)
            log.info(f"[NVIDIA] batch {batch_num}/{total_batches} OK | elapsed={elapsed}s | response_chars={len(raw)}")
            log.debug(f"[NVIDIA] raw response: {raw[:500]}")
            return raw
        except Exception as e:
            elapsed = round(time.time() - t0, 2)
            if attempt == max_retries:
                log.error(f"[NVIDIA] batch {batch_num} FAILED after {max_retries} attempts | elapsed={elapsed}s | error={e}")
                raise
            log.warning(f"[NVIDIA] batch {batch_num} attempt {attempt}/{max_retries} failed — retry in {delay}s | error={e}")
            time.sleep(delay)
            delay *= 2


def _collect_stream(stream) -> str:
    parts = []
    for chunk in stream:
        if not getattr(chunk, "choices", None) or not chunk.choices:
            continue
        delta = getattr(chunk.choices[0], "delta", None)
        if delta and getattr(delta, "content", None):
            parts.append(delta.content)
    return "".join(parts)


def _parse_scores(raw: str) -> list[dict]:
    # Strip markdown code fences if present
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # drop first line (```json or ```) and last line (```)
        inner = lines[1:] if lines[-1].strip() == "```" else lines[1:]
        if inner and inner[-1].strip() == "```":
            inner = inner[:-1]
        text = "\n".join(inner).strip()
    try:
        data = json.loads(text)
        if isinstance(data, list):
            items = data
        else:
            items = next((v for v in data.values() if isinstance(v, list)), [])
        for item in items:
            if not item.get("reason"):
                item["reason"] = f"score={item.get('score', '?')} (no reason provided by model)"
        if not items:
            log.warning(f"[NVIDIA] could not find list in response: {raw[:200]}")
        return items
    except json.JSONDecodeError as e:
        log.error(f"[NVIDIA] JSON parse error: {e} | raw={raw[:300]}")
        return []


def label_segment(segment_text: str) -> str:
    client = _get_client()
    model = os.environ.get("GLM_MODEL", "z-ai/glm4.7")

    log.debug(f"[NVIDIA] label_segment | chars={len(segment_text)}")
    t0 = time.time()
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content":
                f"Describe this video section in 3-5 words (title case):\n\n{segment_text[:400]}"}],
            max_tokens=30,
            temperature=0,
            stream=True,
            extra_body={"chat_template_kwargs": {"enable_thinking": False, "clear_thinking": True}},
        )
        label = _collect_stream(stream).strip()
        elapsed = round(time.time() - t0, 2)
        log.debug(f"[NVIDIA] label OK | elapsed={elapsed}s | label={label!r}")
        return label
    except Exception as e:
        elapsed = round(time.time() - t0, 2)
        log.error(f"[NVIDIA] label FAILED | elapsed={elapsed}s | error={e}")
        raise


def label_all_segments(segments: list[dict]) -> list[dict]:
    log.info(f"[NVIDIA] labelling {len(segments)} segments")
    for i, seg in enumerate(segments):
        seg["label"] = label_segment(seg.get("text", ""))
        log.debug(f"[NVIDIA] segment {i} label: {seg['label']!r}")
    log.info("[NVIDIA] all segments labelled")
    return segments
