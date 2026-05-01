# CLAUDE.md — Video Summarization System

> This file is the single source of truth for AI agents, developers, and contributors
> working on this codebase. Read it fully before writing any code or making any changes.

---

## Project overview

This system takes any uploaded video of arbitrary duration and produces a semantically
compressed output video at a user-chosen or system-recommended compression ratio between
2× and 10×. Unlike text-summary tools, the output is a shorter watchable video that
preserves the most meaningful content from the original.

The system analyzes the video across three modalities — visual, audio, and transcript —
computes a density score that quantifies how information-dense the content is, and uses
that score to recommend the optimal compression ratio. Users can accept the recommendation
or choose their own ratio. The system then selects the best segments using a strategy
appropriate to the chosen ratio band and stitches them into the output video.

**This is not a transcription tool. The output is always a video file.**

---

## Current environment constraints

| Constraint | Current (POC) | Production target |
|---|---|---|
| GPU | None — CPU only | Not required (optional for render acceleration) |
| Storage | Local filesystem (`./data/`) | S3 (ap-south-1) |
| Transcription | Groq API — Whisper large-v3 | Groq API (keep — cheaper than self-hosted) |
| Visual model | MoViNet-A2-Stream (CPU) | MoViNet-A2-Stream (stays CPU) |
| Topic segmentation | TextTiling + sentence-transformers | Same (no change needed) |
| LLM scoring | DeepSeek V4 via NVIDIA NIM API | DeepSeek V4-Flash (4–6×) + V4-Pro (7–10×) |
| Queue | None (single process) | Temporal or Celery + Redis |
| Database | SQLite | Postgres RDS |
| Cache | In-memory dict | Redis ElastiCache |

**The code must be written so each row can be swapped by changing config, not code.**

---

## Repository structure

```
videosum/
├── apps/
│   ├── api/
│   │   ├── main.py
│   │   ├── routers/
│   │   │   ├── jobs.py         # POST /jobs, GET /jobs/{id}, POST /jobs/{id}/confirm-ratio
│   │   │   └── feedback.py     # POST /jobs/{id}/feedback
│   │   └── dependencies.py
│   └── workers/
│       ├── ingest.py           # ffmpeg normalize, demux, frame sample
│       ├── transcribe.py       # Groq Whisper API
│       ├── embed.py            # MoViNet (video lane) + librosa (audio lane)
│       ├── segment.py          # sentence-transformers + TextTiling + DeepSeek labels
│       ├── density.py          # density score computation
│       ├── score.py            # DeepSeek V4 segment scoring
│       ├── select.py           # ratio-band selector
│       └── render.py           # ffmpeg stitch
├── packages/
│   ├── core/
│   │   ├── density/
│   │   │   ├── scorer.py       # compute_density_score() — pure function
│   │   │   └── signals.py      # individual signal extractors
│   │   ├── segmentation/
│   │   │   ├── texttiling.py   # cosine similarity boundary detection
│   │   │   └── labels.py       # DeepSeek V4-Flash segment labeling
│   │   ├── scoring/
│   │   │   ├── llm.py          # DeepSeek V4 via NVIDIA NIM
│   │   │   └── composite.py    # merge LLM + audio + visual scores
│   │   ├── selection/
│   │   │   ├── light.py        # 2–3× rule-based filler/silence removal
│   │   │   ├── moderate.py     # 4–6× submodular greedy
│   │   │   └── highlight.py    # 7–10× topic-cluster representative
│   │   └── models.py           # shared Pydantic types
│   ├── storage/
│   │   ├── base.py             # StorageBackend abstract interface
│   │   ├── local.py            # local filesystem — POC
│   │   ├── s3.py               # S3 — production
│   │   ├── db.py               # SQLite (POC) / Postgres (prod)
│   │   └── cache.py            # in-memory dict (POC) / Redis (prod)
│   └── ml/
│       ├── whisper.py          # Groq Whisper large-v3 wrapper
│       ├── movinet.py          # MoViNet-A2-Stream CPU wrapper
│       ├── embeddings.py       # sentence-transformers all-MiniLM-L6-v2
│       └── deepseek.py         # DeepSeek V4 via NVIDIA NIM (scoring + labeling)
├── scripts/
│   └── download_movinet.py
├── infra/
│   ├── terraform/
│   ├── k8s/
│   └── docker/
└── tests/
    ├── unit/
    ├── integration/
    └── fixtures/
```

**Critical rule:** `packages/core` has zero infra dependencies. Pure Python + numpy + pydantic only.

---

## Architecture

### Full pipeline

```
Video upload
    │
    ▼
ffmpeg normalize (30fps constant, 16kHz mono WAV, 1fps frame sample)
    │                              ~20 sec · CPU · free
    ├─────────────────────────────────────────────┐
    │  VIDEO LANE                                 │  AUDIO LANE
    ▼                                             ▼
PySceneDetect                            Groq Whisper large-v3
(scene boundaries)                       (word-level transcript)
~25 sec · CPU · free                     ~6 sec · API · $0.055/video
    │                                             │
    ▼                                             ▼
MoViNet-A2-Stream                        librosa
(temporal embeddings 600-dim)            (volume, pitch, speech rate)
~30 sec · CPU · free                     ~15 sec · CPU · free
    │                                             │
    └─────────────────┬───────────────────────────┘
                      │
                      ▼
          sentence-transformers (all-MiniLM-L6-v2)
          384-dim embeddings per scene chunk
          ~10 sec · CPU · free
                      │
                      ▼
          TextTiling (cosine similarity boundary detection)
          → topic segments with timestamps
          → topic count (replaces BERTopic)
          ~2 sec · CPU · free
                      │
                      ▼
          Density scorer
          → score 0.0–1.0 + recommended ratio + confidence interval
          ~2 sec · CPU · free
                      │
                      ▼
          ┌── API returns recommendation to user ──┐
          │   user confirms ratio                  │
          ▼                                        │
  DeepSeek V4 via NVIDIA NIM           (optional) DeepSeek V4-Flash
  Score each segment 0–10              Label each topic segment
  Flash for 4–6×, Pro for 7–10×       "Introduction", "Demo", "Q&A"
  ~20 sec · API · $0.003–0.013/video   ~2 sec · API · $0.001/video
          │
          ▼
  Selector (light / moderate / highlight)
  < 1 sec · CPU · free
          │
          ▼
  ffmpeg render
  Stream copy (-c copy) when no crossfades needed: ~5 sec
  Re-encode only when adding crossfades: ~45 sec
  CPU · free
          │
          ▼
  Output video saved to ./data/{hash}/output_{ratio}.mp4
```

### Two-phase design

**Phase 1 — Analyze** (~2.5 min): Runs automatically on upload. Returns density score,
recommended ratio with confidence interval, topic segments with timestamps, and per-signal
breakdown for the UI. No user input required.

**Phase 2 — Summarize** (~75 sec): Only starts after user confirms ratio via
`POST /jobs/{id}/confirm-ratio`. Never render at the wrong ratio.

---

## AI model stack

### Groq — Whisper large-v3

**What**: Speech-to-text with word-level timestamps.
**When**: Phase 1, audio lane. Once per unique video hash.
**Why Groq**: 299× real-time. 30-min video → ~6 sec. Eliminates transcription as bottleneck
(was 25–40 min self-hosted on CPU). $0.111/hr = $0.055/30-min video.
**Alternative**: Whisper large-v3-turbo on Groq at $0.04/hr ($0.02/video) for English-only.
**File size constraint**: 100MB/request. Normalize to 16kHz mono WAV first (reduces 6×).
Chunk into ≤24-min segments. Use 2 chunks for a 30-min video.
**Minimum charge**: 10 seconds/request. Keep chunks ≥10 min.
**Batch API**: 50% discount available for async (non-real-time) workloads. Use at scale.

```python
# packages/ml/whisper.py
from groq import Groq
client = Groq(api_key=os.environ["GROQ_API_KEY"])

def transcribe(audio_path: str) -> dict:
    chunks = chunk_audio(audio_path, max_minutes=24)
    segments = []
    for chunk_path, offset_s in chunks:
        with open(chunk_path, "rb") as f:
            result = client.audio.transcriptions.create(
                file=f,
                model=os.environ.get("GROQ_WHISPER_MODEL", "whisper-large-v3"),
                response_format="verbose_json",
                timestamp_granularities=["word", "segment"]
            )
        for seg in result.segments:
            seg.start += offset_s
            seg.end += offset_s
            segments.append(seg)
    return {"segments": segments, "language": result.language}
```

---

### MoViNet-A2-Stream — temporal visual embeddings

**What**: Encodes motion, transitions, and action continuity across frames.
600-dim feature vector per frame with temporal context from all prior frames.
**When**: Phase 1, video lane. 1 frame/sec.
**Why MoViNet over CLIP on CPU**:
- CLIP: each frame processed in total isolation — zero temporal awareness. Runs at
  8–12fps on CPU.
- MoViNet-A2-Stream: causal (2+1)D convolutions with stream buffer — each frame
  processed with memory of all prior frames. Runs at ~60fps on CPU.
- MoViNet is 5× faster AND temporally aware. No contest on CPU.
**Why not X-CLIP**: GPU model, slow on CPU (~15fps). Not worth it without GPU.
**Why not TimeSformer**: Requires A10G GPU, 12GB VRAM. Overkill.
**Hardware**: CPU only. ~400MB RAM. ~30 sec for 1800 frames.

```python
# packages/ml/movinet.py
import tensorflow as tf
import numpy as np
# MUST use streaming model (movinet_a2_stream), NOT base model
# Base uses tf.nn.conv3d — slow on CPU
# Stream uses (2+1)D conv = fast on CPU via AVX/AVX-512

MODEL_PATH = os.environ.get("MOVINET_MODEL_PATH", "./models/movinet_a2_stream")

def load_model():
    model = tf.saved_model.load(MODEL_PATH)
    # Pre-warm to avoid cold-start on first video
    dummy = tf.zeros([1, 1, 172, 172, 3])
    model({"image": dummy, **model.init_states(tf.shape(dummy))})
    return model

def embed_frames(frames: list, model) -> np.ndarray:
    states = model.init_states(tf.shape(frames[0]))
    embeddings = []
    for frame in frames:
        outputs, states = model({"image": tf.expand_dims(frame, 0), **states})
        embeddings.append(outputs["logits"].numpy()[0])
    return np.array(embeddings)  # shape: (N_frames, 600)
```

---

### sentence-transformers + TextTiling — topic segmentation

**What**: Converts transcript chunks to 384-dim embeddings, then detects topic
boundaries via cosine similarity drops between adjacent chunks.
**When**: Phase 1, after transcription. Runs on scene-aligned transcript chunks.
**Why TextTiling over BERTopic**:
BERTopic is designed for large document corpora — it clusters globally and does not
give timestamps. For a single video you need topic *segmentation* (where do topics
change?) not topic *modeling* (what topics exist across many documents?).

TextTiling answers exactly the right question: where does the content shift?
It uses embeddings you already computed (sentence-transformers), adds ~2 sec of
cosine similarity math, and returns segments with timestamps directly mapped to
the video timeline. Zero additional dependencies.

**Output**: list of `{start_s, end_s, chunk_indices}` — each entry is one topic segment.
**Topic count** = `len(segments)` — replaces BERTopic's topic count in the density scorer.

```python
# packages/core/segmentation/texttiling.py
import numpy as np
from sentence_transformers import SentenceTransformer

_model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_chunks(chunks: list[str]) -> np.ndarray:
    return _model.encode(chunks, normalize_embeddings=True)

def find_boundaries(chunks: list[dict], threshold: float = 0.35) -> list[dict]:
    """
    chunks: [{"text": str, "start_s": float, "end_s": float}, ...]
    Returns topic segments with timestamps.
    threshold: cosine sim drop below this = topic change. Tune per content type:
      - lectures/structured: 0.35
      - casual/unstructured: 0.25 (more sensitive)
    """
    texts = [c["text"] for c in chunks]
    embs = embed_chunks(texts)

    boundaries = [0]
    for i in range(1, len(embs)):
        sim = float(np.dot(embs[i-1], embs[i]))  # already normalized
        if sim < threshold:
            boundaries.append(i)
    boundaries.append(len(chunks))

    segments = []
    for i in range(len(boundaries) - 1):
        start_idx = boundaries[i]
        end_idx = boundaries[i + 1]
        segments.append({
            "segment_index": i,
            "start_s": chunks[start_idx]["start_s"],
            "end_s": chunks[end_idx - 1]["end_s"],
            "chunk_indices": list(range(start_idx, end_idx)),
            "text": " ".join(c["text"] for c in chunks[start_idx:end_idx])
        })
    return segments
```

**Threshold tuning**:
- Start with 0.35 for structured content (lectures, demos, tutorials)
- Lower to 0.25 for casual content (interviews, podcasts) to catch subtler shifts
- If getting too many tiny segments (< 1 min each): raise threshold
- If missing obvious topic changes: lower threshold
- Log detected segment count per video — outliers (< 2 or > 15 segments for a 30-min video) indicate threshold needs tuning for that content type

---

### DeepSeek V4-Flash — topic segment labeling (optional)

**What**: Reads each topic segment's text and returns a 3–5 word human-readable label.
**When**: Phase 2, after TextTiling segments are computed. Optional — only fires if
chapter labels are enabled (they appear as title cards in the output video).
**Why**: Turns "Segment 1, Segment 2" into "Introduction", "Core concepts", "Live demo",
"Q&A". Uses Flash (cheapest model) because labeling is trivial.
**Cost**: ~5 segments × 20 output tokens × $0.28/M = **$0.001 per video**.

```python
# packages/core/segmentation/labels.py
from packages.ml.deepseek import client

def label_segment(segment_text: str) -> str:
    r = client.chat.completions.create(
        model=os.environ.get("DEEPSEEK_FLASH_MODEL", "deepseek-ai/deepseek-v4-flash"),
        messages=[{"role": "user", "content":
            f"Describe this video section in 3-5 words (title case):\n\n{segment_text[:400]}"}],
        max_tokens=20,
        temperature=0
    )
    return r.choices[0].message.content.strip()

def label_all_segments(segments: list[dict]) -> list[dict]:
    for seg in segments:
        seg["label"] = label_segment(seg["text"])
    return segments
# Output: [{"label": "Introduction and overview", "start_s": 0, "end_s": 180}, ...]
```

---

### DeepSeek V4 via NVIDIA NIM — segment scoring

**What**: Scores each transcript segment 0–10 for importance. At 7–10×, selects the
best representative scene per topic cluster.
**When**: Phase 2, score activity.
**Endpoint**: `https://integrate.api.nvidia.com/v1` (OpenAI-compatible)
**Free tier**: ~1,000 requests/day at build.nvidia.com. 2 calls/video = 500 videos/day ceiling.

**Model routing**:

| Ratio | Model | Mode | Cost/video |
|---|---|---|---|
| 2–3× | Not called | — | $0.000 |
| 4–6× | deepseek-v4-flash | Non-think | $0.003 |
| 7–10× | deepseek-v4-pro | Think High | $0.013 |
| Segment labeling | deepseek-v4-flash | Non-think | $0.001 |

**Why DeepSeek V4 over Claude**: V4-Flash is ~10× cheaper than Claude Haiku for
structured JSON scoring. V4-Pro matches Claude Opus 4.6 quality at lower cost.
Open weights — self-hostable on 2× A100 80GB for Flash when volume justifies it.

```python
# packages/ml/deepseek.py
from openai import OpenAI

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ["NVIDIA_API_KEY"]
)

def score_segments(segments: list, video_type: str, ratio: int) -> list:
    use_pro = ratio >= int(os.environ.get("DEEPSEEK_PRO_RATIO_THRESHOLD", "7"))
    model = (
        os.environ.get("DEEPSEEK_PRO_MODEL", "deepseek-ai/deepseek-v4-pro")
        if use_pro else
        os.environ.get("DEEPSEEK_FLASH_MODEL", "deepseek-ai/deepseek-v4-flash")
    )
    system = "Score video transcript segments for summarization. Return only JSON."
    if use_pro:
        system += " Think carefully — high compression editorial decisions."

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": _build_prompt(segments, video_type, ratio)}
        ],
        temperature=0.1,
        response_format={"type": "json_object"},
        max_tokens=1000
    )
    return _parse_scores(resp.choices[0].message.content)
```

**Scoring prompt contract** (do not change without updating `tests/evals/llm_scoring.py`):
```
Given this {video_type} video at {ratio}× compression, score each segment.

Output JSON array. Each element:
- "id": segment id
- "score": integer 0–10 (10 = essential; 0 = pure filler)
- "reason": one sentence max

Context: {topic_summary}
Be strict — only {round(100/ratio)}% of content survives.

Segments:
{segments_json}
```

**Caching**: `sha256(segment_text + video_type + str(ratio))` → TTL 24h.
15–30% cache hit rate expected on repeat/similar content.

---

### librosa — audio feature extraction

**What**: RMS energy (volume), pitch variance (F0), speech rate (WPM from Groq
timestamps), silence ratio — per scene segment.
**When**: Phase 1, audio lane, after Groq transcription.
**Why**: Prosodic emphasis is invisible in transcript text. Loud + high-pitch = emphasis.
**Hardware**: Pure CPU. ~15 sec for 30-min video.

---

## Topic segmentation — full flow

```
Groq transcript (word-level timestamps)
        │
        ▼
Align chunks to PySceneDetect boundaries
(one chunk per scene, ~24 chunks for 30 min)
        │
        ▼
sentence-transformers → 384-dim embeddings
        │
        ▼
TextTiling cosine similarity boundary detection
        │
        ▼
Topic segments with timestamps:
  [0–3 min: intro text...]
  [3–14 min: core content text...]
  [14–22 min: demo text...]
  [22–30 min: wrap-up text...]
        │
        ├──→ topic_count → density scorer
        │
        └──→ (optional) DeepSeek V4-Flash label each segment
               "Introduction" / "Core concepts" / "Live demo" / "Q&A"
               → chapter title cards in rendered output
```

**TextTiling vs BERTopic — why the switch**:

| | TextTiling | BERTopic |
|---|---|---|
| Designed for | Single document segmentation | Multi-document corpus analysis |
| Gives timestamps | Yes — directly | No |
| Dependencies | sentence-transformers (already in stack) | Additional: hdbscan, umap, etc. |
| Cold-start | ~2 sec | ~8–15 sec |
| Accuracy on single video | High | Medium (needs many docs to cluster well) |
| Topic count | Number of boundaries found | Cluster count |

---

## Density scoring

```python
@dataclass
class DensitySignals:
    # sentence-transformers: mean pairwise cosine sim of chunk embeddings
    semantic_redundancy: float

    # librosa
    silence_ratio: float
    filler_word_rate: float    # um/uh/like/you know per 100 words
    pacing_variance: float     # std dev of WPM across scenes

    # Groq transcript
    lexical_density: float     # type-token ratio

    # TextTiling
    topic_count: int           # number of distinct topic segments found

    # MoViNet
    visual_change_rate: float  # mean frame-to-frame feature distance

def compute_density_score(s: DensitySignals) -> float:
    compressibility = (
        0.30 * s.semantic_redundancy +
        0.20 * s.silence_ratio +
        0.15 * s.filler_word_rate
    )
    density = (
        0.20 * s.lexical_density +
        0.10 * normalize_topic_count(s.topic_count) +
        0.05 * s.visual_change_rate
    )
    return float(np.clip(density - compressibility + 0.5, 0.0, 1.0))
```

### Ratio recommendation

| Density score | Recommended ratio | Confidence interval |
|---|---|---|
| 0.0 – 0.20 | 8–10× | ±2 |
| 0.20 – 0.40 | 5–7× | ±2 |
| 0.40 – 0.60 | 3–4× | ±1 |
| 0.60 – 0.80 | 2× | ±1 |
| 0.80 – 1.00 | Do not compress | — |

Score > 0.80: return `recommended_ratio: null` with message
`"Content too dense for video compression. Consider a text summary instead."`

---

## Selection strategies

### Light trim (2–3×) — `packages/core/selection/light.py`

Rule-based. No LLM. Sentence level.
Removes: silences > 1.5s, filler words (um/uh/like/you know), repeated sentences
(cosine sim > 0.92), verbal restarts. Quality target: ≥ 92%.

### Moderate (4–6×) — `packages/core/selection/moderate.py`

Submodular greedy at scene level.
```
composite_score = 0.45 × llm_score + 0.30 × audio_emphasis + 0.25 × visual_salience

Sort scenes by composite_score descending
For each scene c:
  if total_duration(S) + duration(c) > target: skip
  if max_cosine_sim(embed(c), selected) > 0.82: skip  # diversity
  S.append(c)
Return S in chronological order
```

### Highlight reel (7–10×) — `packages/core/selection/highlight.py`

Topic-cluster representative selection.
1. Use TextTiling segments as clusters
2. Rank by sum of composite scores within segment
3. Select top K: `K = ceil(topic_count × (1 - (ratio-7)/10))`
4. DeepSeek V4-Pro (Think High) picks best scene within each cluster
5. Return in chronological order

DeepSeek V4-Pro mandatory here — within a segment all scenes are semantically similar.
Flash cannot reliably distinguish "introduces concept" from "repeats concept."

---

## Storage — local filesystem (POC)

Keyed by `video_hash` (SHA-256 of raw video bytes). Same key = dedup.

```
./data/videos/{sha256_hash}/
  source.{ext}              # original upload
  normalized.mp4            # constant 30fps re-encode
  audio.wav                 # 16kHz mono WAV
  audio_chunk_00.wav        # Groq chunk 0 (≤24 min)
  audio_chunk_01.wav        # Groq chunk 1 (if >24 min)
  frames/                   # 1fps JPEGs
  transcript.json           # Groq word-level output, offset-corrected
  scenes.json               # PySceneDetect boundaries
  movinet_features.npy      # shape (N_frames, 600)
  chunk_embeddings.npy      # shape (N_scenes, 384)
  audio_features.json       # librosa per scene
  topic_segments.json       # TextTiling output with labels
  density.json              # score + signals + recommendation
  scores_{ratio}.json       # DeepSeek composite scores
  selection_{ratio}.json    # (start_s, end_s) list
  output_{ratio}.mp4        # final output
```

Migration to S3: set `STORAGE_BACKEND=s3`. Keys identical. Zero code changes.

---

## Cost per 30-min video

### POC (local machine, free tiers)

| Component | Model | Cost |
|---|---|---|
| Transcription | Groq Whisper large-v3 | $0.055 |
| Segment scoring (ratio 4–6×) | DeepSeek V4-Flash | $0.003 |
| Segment scoring (ratio 7–10×) | DeepSeek V4-Pro | $0.013 |
| Segment labeling (chapter titles) | DeepSeek V4-Flash | $0.001 |
| NVIDIA NIM API | Free tier | $0.000 |
| Local storage + CPU | — | $0.000 |
| **Total (ratio 4–6×)** | | **$0.059** |
| **Total (ratio 7–10×)** | | **$0.069** |

Groq = 78% of cost. Dominant lever at all scales.

### Production estimate (adds cloud infra)

| Component | Cost/video | Notes |
|---|---|---|
| All API costs | $0.059–0.069 | Same as POC |
| S3 storage + transfer | $0.030 | ~2GB artifacts, 7-day TTL |
| EC2 t3.medium spot CPU | $0.001 | ~3.5 min @ $0.015/hr |
| Postgres + Redis amortized | $0.055 | $165/mo ÷ 100 videos/day |
| **Total at 100/day** | **~$0.15/video** | **~$450/month** |

### Cost at scale

| Volume | Per video | Monthly | Notes |
|---|---|---|---|
| 10/day | $0.12 | ~$36 | POC — single machine |
| 50/day | $0.11 | ~$165 | Add queue + EC2 |
| 100/day | $0.15 | ~$450 | K8s viable |
| 300/day | $0.13 | ~$1,170 | Cache hits reduce LLM cost |
| 500/day | $0.12 | ~$1,800 | Consider Groq batch API |

**Single biggest cost lever**: Groq Batch API = 50% discount on transcription for
async jobs. Switches $0.055 → $0.028/video. Drops total from $0.15 → $0.09 at 100/day.
Zero code changes — just a flag in the API call.

**Self-hosting break-even**: DeepSeek V4-Flash on 2× A100 80GB breaks even vs NVIDIA NIM
at ~800 videos/day. Don't self-host below this volume.

---

## Pipeline wall-clock time (CPU-only, 30-min video, 8-core machine)

| Stage | Tool | Time | Bottleneck? |
|---|---|---|---|
| ffmpeg normalize + demux | ffmpeg | ~20 sec | No |
| Audio chunking | ffmpeg | ~5 sec | No |
| Groq transcription | Groq Whisper large-v3 | **~6 sec** | No |
| Scene detection | PySceneDetect | ~25 sec | No |
| MoViNet embedding | MoViNet-A2-Stream | ~30 sec | No |
| librosa features | librosa | ~15 sec | No |
| sentence-transformers | all-MiniLM-L6-v2 | ~10 sec | No |
| TextTiling | cosine sim math | ~2 sec | No |
| Density scoring | Python | ~2 sec | No |
| **Phase 1 total** | | **~2.5 min** | |
| DeepSeek scoring | NVIDIA NIM | ~20 sec | No |
| Segment labeling | NVIDIA NIM | ~5 sec | No |
| Selection | Python | <1 sec | No |
| ffmpeg render (stream copy) | ffmpeg | **~5 sec** | No |
| ffmpeg render (re-encode) | ffmpeg | ~45 sec | Mild |
| **Phase 2 total** | | **~35 sec** | |
| **Total (stream copy)** | | **~3 min** | |
| **Total (re-encode)** | | **~3.5 min** | |

### Render optimization — critical

Default ffmpeg concat re-encodes every frame = 45 sec. Use stream copy when possible:

```python
# packages/workers/render.py

def render(selection: list[dict], input_path: str, output_path: str,
           add_crossfades: bool = False):
    if not add_crossfades:
        # Stream copy — no re-encode, ~5 sec
        # Only works when input codec = output codec (always true for normalized.mp4)
        ffmpeg_concat(selection, input_path, output_path, codec="copy")
    else:
        # Re-encode only for crossfade transitions, ~45 sec
        ffmpeg_concat_with_transitions(selection, input_path, output_path)
```

Default to stream copy. Only re-encode if user explicitly requests crossfades.
At POC scale this saves 40 sec per video. At 100 videos/day it saves 66 CPU-hours/day.

---

## Scalability ceilings and fixes

### Groq free tier

**Ceiling**: 7,200 seconds of audio/hour = 240 videos/hour maximum.
**Fix for POC**: Use multiple Groq API keys (rotate per request).
**Fix for production**: Upgrade to Groq paid tier (~$50/month removes ceiling).
**Fix at scale**: Groq Batch API (50% discount, 24hr processing window).

### NVIDIA NIM free tier

**Ceiling**: ~1,000 requests/day ≈ 500 videos/day at 2 calls/video.
**Fix for POC**: Sufficient for development (500 videos/day is a lot for POC).
**Fix for production**: NIM paid tier removes ceiling.
**Fix at scale (800+/day)**: Self-host DeepSeek V4-Flash on 2× A100 80GB.

### ffmpeg render

**Ceiling**: ~45 sec per video with re-encode = 80 videos/day on one machine.
**Fix**: Use stream copy (5 sec) as default — 720 videos/day on one machine.
**Fix at scale**: GPU-accelerated NVENC encoding (3–4× faster than CPU re-encode)
but only needed at 500+ videos/day. Not required in current stack.

### MoViNet + sentence-transformers (CPU)

**Ceiling**: None practical. Both scale horizontally — add more CPU worker processes.
1 worker handles ~20 videos/hour. 10 workers = 200 videos/hour. Linear scaling.

### Everything else

PySceneDetect, librosa, TextTiling, density scorer, selector — all pure CPU, all
stateless, all scale by running more processes. No bottlenecks before 2000+/day.

---

## Environment variables

```bash
# === Transcription (Groq) ===
GROQ_API_KEY=gsk_...
GROQ_WHISPER_MODEL=whisper-large-v3       # or whisper-large-v3-turbo (English only)
GROQ_AUDIO_CHUNK_MINUTES=24
GROQ_MIN_CHUNK_SECONDS=600
GROQ_USE_BATCH=false                      # set true for 50% discount on async jobs

# === Visual model (MoViNet) ===
MOVINET_MODEL_PATH=./models/movinet_a2_stream
MOVINET_FRAME_SAMPLE_RATE=1

# === Topic segmentation (TextTiling) ===
TEXTTILING_THRESHOLD=0.35                 # lower = more sensitive to topic changes
TEXTTILING_MIN_SEGMENT_SECONDS=60        # ignore boundaries < 1 min apart
SEGMENT_LABELING_ENABLED=true            # DeepSeek labels each segment

# === LLM scoring (DeepSeek V4 via NVIDIA NIM) ===
NVIDIA_API_KEY=nvapi-...
DEEPSEEK_FLASH_MODEL=deepseek-ai/deepseek-v4-flash
DEEPSEEK_PRO_MODEL=deepseek-ai/deepseek-v4-pro
DEEPSEEK_PRO_RATIO_THRESHOLD=7
LLM_CACHE_TTL_SECONDS=86400
LLM_MAX_SEGMENTS_PER_CALL=20
LLM_CONFIDENCE_FALLBACK_THRESHOLD=0.6

# === Render ===
RENDER_USE_STREAM_COPY=true              # use -c copy (fast) vs re-encode (slow)
RENDER_ADD_CROSSFADES=false             # forces re-encode if true
RENDER_CROSSFADE_MS=200

# === Storage ===
STORAGE_BACKEND=local                    # "local" | "s3"
LOCAL_STORAGE_PATH=./data
S3_BUCKET=videosum-artifacts
S3_REGION=ap-south-1

# === Database ===
DATABASE_BACKEND=sqlite                  # "sqlite" | "postgres"
SQLITE_PATH=./data/videosum.db
DATABASE_URL=postgresql://...

# === Cache ===
CACHE_BACKEND=memory                     # "memory" | "redis"
REDIS_URL=redis://...

# === Pipeline ===
SKIP_VISUAL_EMBEDDINGS_BELOW_RATIO=3
MAX_VIDEO_DURATION_SECONDS=7200
MAX_UPLOAD_SIZE_GB=10
```

---

## Common failure modes

**Groq 413 — file too large**
Chunk exceeds 100MB. Lower `GROQ_AUDIO_CHUNK_MINUTES` to 20.
Always normalize to 16kHz mono WAV before chunking:
`ffmpeg -i input.mp4 -ar 16000 -ac 1 audio.wav`

**Groq timestamp discontinuity at chunk boundaries**
Apply 30-second overlap between chunks. Deduplicate words with matching text
within the overlap zone.

**TextTiling creates too many tiny segments (< 1 min)**
Threshold too low. Raise `TEXTTILING_THRESHOLD` from 0.35 to 0.45.
Also enforce `TEXTTILING_MIN_SEGMENT_SECONDS=60` to merge tiny segments with neighbors.

**TextTiling misses obvious topic changes**
Threshold too high. Lower from 0.35 to 0.25.
Check: are chunks too short (< 30 words)? Short chunks have unstable embeddings.
Merge chunks shorter than 30 words with their neighbors before running TextTiling.

**MoViNet slow on first video**
TF SavedModel cold-start ~3 sec. Pre-warm at startup (implemented in `load_model()`).
In production keep 1 warm worker minimum.

**DeepSeek V4-Flash scores all cluster around 5**
Chunks too short (< 30 words) or too similar. Merge before scoring.
If clustering persists, escalate to V4-Pro for that job.

**NVIDIA NIM 429 rate limit**
Free tier limit hit. Retry with exponential backoff (1s, 2s, 4s).
Do not swap to a lower-quality model. Queue and wait.
For production: upgrade NIM tier.

**Render output has wrong duration (stream copy mode)**
Input has variable frame rate. Stream copy cannot fix VFR.
Normalize to constant 30fps during ingest:
`ffmpeg -i input.mp4 -vf fps=30 -c:v libx264 -c:a aac normalized.mp4`
This is already done in the ingest worker. If this error appears, ingest worker
failed silently — check `normalized.mp4` exists before proceeding to render.

**Render audio sync drift**
Only happens with re-encode mode. Ensure `-vsync cfr` flag in ffmpeg call.

**Groq hallucinating on silence**
Pre-check with librosa: skip chunks where `np.mean(librosa.feature.rms(y=audio)) < 0.01`.
Mark skipped ranges as silence in transcript.

---

## Observability (POC — structured stdout)

```python
import json, logging

def log_stage(job_id, stage, model, duration_ms, cost_usd, status, extra=None):
    logging.info(json.dumps({
        "job_id": job_id, "stage": stage, "model": model,
        "duration_ms": round(duration_ms), "cost_usd": round(cost_usd, 5),
        "status": status, **(extra or {})
    }))

# Extra fields worth logging per stage:
# transcribe: language, language_probability, n_chunks
# texttiling: n_segments, threshold_used, segment_durations
# score: ratio, model_used, score_distribution_mean, score_distribution_std
# render: codec_mode ("copy" or "encode"), output_duration_s
```

Track weekly: Groq cost/video, DeepSeek cost/video, total wall-clock, TextTiling
segment count distribution, LLM score distribution per ratio band.

---

## Development setup

```bash
# Prerequisites: Python 3.11+, ffmpeg

git clone https://github.com/your-org/videosum
cd videosum

pip install -e "packages/core[dev]"
pip install -e "packages/ml[dev]"
pip install -e "packages/storage[dev]"
pip install -e "apps/api[dev]"

# TensorFlow CPU (no CUDA needed)
pip install tensorflow-cpu

# Download MoViNet-A2-Stream weights (~25MB)
python scripts/download_movinet.py --model a2 --output ./models/movinet_a2_stream

# Configure
cp .env.example .env
# Fill in: GROQ_API_KEY, NVIDIA_API_KEY

# Test
pytest tests/unit tests/integration -v

# Run
uvicorn apps.api.main:app --reload --port 8000

# Submit test job
curl -X POST http://localhost:8000/jobs \
  -H "Content-Type: application/json" \
  -d '{"video_path": "./tests/fixtures/sample.mp4", "ratio": "auto"}'
```

---

## Upgrade path — POC to production

1. **Local → S3**: set `STORAGE_BACKEND=s3`. Zero code changes.
2. **SQLite → Postgres**: set `DATABASE_BACKEND=postgres`. Run migrations.
3. **Memory cache → Redis**: set `CACHE_BACKEND=redis`.
4. **Single process → queue**: add Celery (simpler) or Temporal (more robust).
5. **NIM free → NIM paid**: same endpoint, higher rate limits.
6. **Groq on-demand → Groq batch**: set `GROQ_USE_BATCH=true`. 50% cost reduction.
7. **NVIDIA NIM → self-hosted DeepSeek Flash**: at 800+ videos/day on 2× A100 80GB.

---

## Competitive context

Every existing tool (NoteGPT, Eightify, Vimeo AI, Otter.ai, Mindgrasp, Notta)
outputs text. None output a shorter video. Vimeo AI produces a video recap but only
within their platform, no ratio control, no density analysis.

This system's moat:
1. Video-to-video output — not text
2. Content-density-aware ratio recommendation
3. Three selection strategies matched to ratio band
4. TextTiling-based topic segmentation that maps directly to video timeline
5. Chapter labels baked into the output video (human-readable, not "Segment 1")
6. Feedback loop — every rated job improves the density scorer over time

Log all signals, scores, chosen ratios, TextTiling segment counts, and user ratings
from day one. That dataset compounds. It is the long-term moat.

---

*Last updated: 2026-04-30*

*Changes in this version:*
- *Topic segmentation: BERTopic removed → replaced with TextTiling (cosine similarity boundary detection using existing sentence-transformer embeddings). Simpler, faster, gives timestamps directly, zero new dependencies.*
- *Added DeepSeek V4-Flash segment labeling for chapter title cards ($0.001/video optional).*
- *Added full cost breakdown tables: POC ($0.059–0.069/video), production ($0.15/video at 100/day), scale table up to 500/day.*
- *Added scalability ceilings section: Groq free tier (240 videos/hr), NVIDIA NIM free tier (500 videos/day), render bottleneck.*
- *Added render optimization: stream copy default (~5 sec) vs re-encode (~45 sec). Critical for throughput.*
- *Added Groq Batch API as cost lever (50% discount, drops total to ~$0.09/video).*
- *Added self-hosting break-even: DeepSeek V4-Flash on 2× A100 80GB at 800+ videos/day.*
- *Updated TextTiling threshold tuning guidance.*
- *Updated render failure mode: VFR → stream copy issue and fix.*
