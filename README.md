# VideoSum — AI Video Summarization System

Takes any uploaded video and produces a shorter watchable video at a user-chosen compression ratio (2×–10×). Output is always a video file — not a text summary.

---

## What it does

1. Analyzes the video across three modalities: visual (MoViNet), audio (librosa), transcript (Groq Whisper)
2. Computes a density score (0–1) that quantifies how information-dense the content is
3. Recommends the optimal compression ratio based on density
4. User confirms ratio → system selects best segments and stitches output video

---

## Architecture

```
Browser / Streamlit UI
        │
        ▼
FastAPI (port 8000)
        │
        ├── Postgres (job state, metrics, exports, feedback)
        ├── Redis (Celery broker + LLM score cache)
        └── Celery Worker
                │
                ├── Phase 1 (~2.5 min) — analyze
                └── Phase 2 (~35 sec)  — summarize
```

### Phase 1 — Analyze (auto on upload)

| Stage | Tool | What happens |
|-------|------|-------------|
| INGEST | ffmpeg | Normalize to 30fps h264, extract 16kHz mono WAV, sample 1fps frames |
| TRANSCRIBE | Groq Whisper large-v3 | Word-level transcript with timestamps |
| EMBED | MoViNet-A2-Stream + librosa | 600-dim temporal embeddings + audio features per scene |
| SEGMENT | sentence-transformers + TextTiling | Topic boundary detection → segments with timestamps |
| DENSITY | Python | Density score 0–1 → recommended ratio |

Returns: density score, recommended ratio, topic segments, signal breakdown.

### Phase 2 — Summarize (starts after user confirms ratio)

| Stage | Tool | What happens |
|-------|------|-------------|
| SCORE | GLM-4.7 via NVIDIA NIM | Score each scene 0–10 for importance |
| SELECT | Python | Pick best scenes by ratio band strategy |
| RENDER | ffmpeg stream copy | Stitch output video (~5 sec, zero quality loss) |

---

## Three selection strategies

| Ratio | Strategy | LLM |
|-------|----------|-----|
| 2–3× | Rule-based: remove silences, fillers, repeated sentences | Not called |
| 4–6× | Submodular greedy: composite score + diversity check | GLM-4.7 standard |
| 7–10× | Topic-cluster representative: one best scene per topic | GLM-4.7 thinking |

---

## AI Model Stack

| Model | Role | Cost/30-min video |
|-------|------|------------------|
| Groq Whisper large-v3 | Transcription (99 languages) | $0.055 |
| MoViNet-A2-Stream | Temporal visual embeddings (CPU) | $0.000 |
| all-MiniLM-L6-v2 | Sentence embeddings for TextTiling | $0.000 |
| GLM-4.7 via NVIDIA NIM | Scene scoring + segment labeling | $0.001–$0.013 |
| **Total** | | **$0.058–$0.069** |

---

## Density Score Formula

```
compressibility = 0.30 × semantic_redundancy
               + 0.20 × silence_ratio
               + 0.15 × filler_word_rate

density        = 0.20 × lexical_density
               + 0.10 × topic_count (normalized)
               + 0.05 × visual_change_rate

score = clip(density - compressibility + 0.5, 0.0, 1.0)
```

| Score | Recommended ratio |
|-------|------------------|
| 0.0–0.20 | 8–10× |
| 0.20–0.40 | 5–7× |
| 0.40–0.60 | 3–4× |
| 0.60–0.80 | 2× |
| 0.80–1.00 | Do not compress |

---

## Setup

```bash
# Prerequisites: Python 3.11+, ffmpeg, Redis, Postgres

git clone <repo>
cd videosum

pip install -r requirements.txt
# TensorFlow CPU (large download ~700MB — install separately)
pip install tensorflow-cpu tensorflow-hub

# Download MoViNet weights (~25MB)
python scripts/download_movinet.py --model a2 --output ./models/movinet_a2_stream

cp .env.example .env
# Fill in: GROQ_API_KEY, NVIDIA_API_KEY, DATABASE_URL, REDIS_URL
```

---

## Running

```bash
# Start API
uvicorn apps.api.main:app --reload --port 8000

# Start Celery worker
python start_worker.py

# Start UI
streamlit run app_ui.py
```

---

## Environment Variables

```bash
GROQ_API_KEY=gsk_...
NVIDIA_API_KEY=nvapi-...
GLM_MODEL=z-ai/glm4.7
DATABASE_BACKEND=postgres          # sqlite | postgres
DATABASE_URL=postgresql://...
CACHE_BACKEND=redis                # memory | redis
REDIS_URL=redis://...
STORAGE_BACKEND=local              # local | s3
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0
```

Full variable reference in [.env.example](.env.example).

---

## Detailed Documentation

- [ProjectFlow.md](ProjectFlow.md) — complete pipeline flow, every decision, DB schema, API endpoints
- [ProjectReadme.md](ProjectReadme.md) — scaling, observability, cost tables, upgrade path

---

## Upgrade Path

```
SQLite    → Postgres    : DATABASE_BACKEND=postgres
Local     → S3          : STORAGE_BACKEND=s3
Memory    → Redis cache : CACHE_BACKEND=redis
Single    → Worker pool : separate phase1/phase2 Celery workers
NIM free  → NIM paid    : same endpoint, higher rate limits
On-demand → Batch API   : GROQ_USE_BATCH=true (50% cost saving)
NIM       → Self-hosted : at 800+ videos/day on 2× A100 80GB
```
