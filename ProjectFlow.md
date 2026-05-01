# VideoSum — Complete Project Flow

---

## What the system does

Takes any uploaded video and produces a semantically compressed output video at a
user-chosen compression ratio (2×–10×). The output is always a video file — not text.
Three modalities analyzed: visual (MoViNet), audio (librosa), transcript (Groq Whisper).
All three feed into a composite score that drives selection.

---

## Architecture overview

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
                ├── Phase 1 queue (CPU-heavy)
                └── Phase 2 queue (API-heavy)
```

---

## Three lanes of analysis

### Audio lane — librosa
- Input: `audio.wav` (16kHz mono)
- Extracts per PySceneDetect scene:
  - RMS energy (volume)
  - Pitch variance F0 (prosodic emphasis)
  - Silence ratio
  - Speech rate WPM (from Groq timestamps)
- Output: `audio_features.json`
- Used in: density scorer (Phase 1) + composite scoring (Phase 2)

### Video lane — MoViNet-A2-Stream
- Input: 1fps JPEG frames sampled from `normalized.mp4`
- Architecture: (2+1)D CNN with causal stream buffer
  - Spatial features: what is in each frame
  - Temporal features: how frames change over time with memory of all prior frames
- Not a Transformer — designed for CPU, no GPU needed
- Produces: 600-dim embedding per frame
- Output: `movinet_features.npy` shape (N_frames, 600)
- Computes: `visual_change_rate` — mean frame-to-frame L2 distance (one global number)
- Used in: density scorer only (Phase 1, weighted 0.05)

### LLM lane — GLM-4.7 via NVIDIA NIM
- Input: transcript text per PySceneDetect scene (text only, no frames)
- Output: score 0–10 + reason per scene
- Used in: composite scoring (Phase 2, weighted 0.45)

---

## Two segmentation systems (independent)

### PySceneDetect — visual cuts
- Detects pixel-level changes between frames on `normalized.mp4` (full 30fps)
- Output: list of `{scene_index, start_s, end_s, duration_s}`
- No labels — just timestamps
- 30-min video: 50–200 scenes depending on camera cut frequency
- **Master clock** — everything else aligns to these boundaries

### TextTiling — topic boundaries
- Input: one text chunk per PySceneDetect scene (words from Groq transcript)
- Embeds each chunk with `all-MiniLM-L6-v2` → 384-dim vector
- Computes cosine similarity between adjacent chunks
- Similarity drops below 0.35 → topic boundary → new segment
- Output: `topic_segments.json` with timestamps + GLM-4.7 labels
- 30-min video: 4–15 topic segments
- Used for: chapter labels (all ratios) + coverage enforcement (7–10× only)

**Key distinction:**
- PySceneDetect: "where did the camera cut?"
- TextTiling: "where did the topic change?"
- These are completely independent. 15 PySceneDetect scenes can exist inside 1 TextTiling segment.

---

## Phase 1 — Analyze (~2.5 min, cached after first run)

### INGEST — ffmpeg
Three outputs from one source video:

1. `normalized.mp4` — constant 30fps h264 + aac
   - Why: stream copy render (5s) only works with consistent codec
   - Why: PySceneDetect needs constant frame rate for accurate cut detection
   - Full 30fps kept — nothing dropped

2. `audio.wav` — 16kHz mono WAV
   - Why 16kHz: Whisper minimum requirement, reduces file size 6× vs stereo 44kHz
   - Why mono: halves data, Groq charges by audio duration not file size

3. `frames/` — 1fps JPEG frames
   - Why 1fps not 30fps: adjacent frames at 30fps are nearly identical
   - 30-min video = 1800 frames at 1fps vs 54,000 at 30fps
   - MoViNet on 1800 frames = 30s. On 54,000 frames = 15 min on CPU

### TRANSCRIBE — Groq Whisper large-v3
- Sends `audio.wav` to Groq API
- Returns word-level timestamps for entire video
- Why Groq: self-hosted Whisper = 25–40 min on CPU. Groq = 6 sec. $0.055/video
- Why large-v3: 99 languages. Turbo = English only at half the cost
- Chunking: videos >24 min split into ≤24-min chunks with 30s overlap
- Overlap deduplication: words in overlap zone of previous chunk are skipped
- Output: `transcript.json`

### EMBED — MoViNet + librosa
Two things happen in parallel:

**MoViNet (video lane):**
- Loads 1fps frames from `frames/`
- Runs each frame through MoViNet-A2-Stream with stream buffer
- Stream buffer = each frame processed with memory of all prior frames
- Output: `movinet_features.npy` shape (N_frames, 600)
- Skipped if ratio ≤ 3 (light trim doesn't need visual features)

**librosa (audio lane):**
- Loads `audio.wav`
- Computes RMS energy, pitch variance, silence ratio per PySceneDetect scene
- Speech rate filled later by segment worker using Groq word timestamps
- Output: `audio_features.json`

### SEGMENT — sentence-transformers + TextTiling
Step by step:

1. Groq transcript words aligned to PySceneDetect scene boundaries
   → one text chunk per scene (variable length, ~30–300 words)

2. Each chunk → `all-MiniLM-L6-v2` → 384-dim embedding

3. Cosine similarity computed between adjacent chunk embeddings:
   - Above 0.35 → same topic → continue current segment
   - Below 0.35 → topic changed → start new segment

4. GLM-4.7 flash labels each TextTiling segment (3–5 words)
   → "Introduction", "Live Demo", "Q&A Session"

5. Output: `topic_segments.json`

**Threshold tuning:**
- 0.35 default → structured content (lectures, demos)
- 0.25 → casual content (interviews, podcasts) — more sensitive
- Raise if getting too many tiny segments (<1 min each)
- Lower if missing obvious topic changes

### DENSITY SCORER — pure Python
Computes single number 0.0–1.0 from 7 signals:

```
compressibility = 0.30 × semantic_redundancy    (sentence-transformers)
               + 0.20 × silence_ratio            (librosa)
               + 0.15 × filler_word_rate          (Groq transcript)

density        = 0.20 × lexical_density           (Groq transcript)
               + 0.10 × normalize(topic_count)    (TextTiling)
               + 0.05 × visual_change_rate        (MoViNet)

score = clip(density - compressibility + 0.5, 0.0, 1.0)
```

Score → recommended ratio:

| Score | Recommended ratio |
|-------|------------------|
| 0.0–0.20 | 8–10× |
| 0.20–0.40 | 5–7× |
| 0.40–0.60 | 3–4× |
| 0.60–0.80 | 2× |
| 0.80–1.00 | Do not compress |

**Phase 1 ends here.**
API returns density score + recommendation + signal breakdown to UI.
User sees the analysis and picks a ratio.

---

## User picks ratio → Phase 2 begins

---

## Phase 2 — Summarize (~35 sec)

### SCORE — GLM-4.7 via NVIDIA NIM
**What gets sent to LLM:**

For each PySceneDetect scene:
- `id`: scene index
- `text`: all Groq transcript words within that scene's time window (max 300 chars)

All scenes sent in ONE API call. Not one call per scene.

**System prompt includes:**
- Content type (interview/lecture/demo etc.)
- Target compression ratio
- Genre-specific editorial guidance
- Topic labels from TextTiling (context only)

**LLM returns:**
```json
[
  {"id": 0, "score": 8, "reason": "Introduces core concept — essential"},
  {"id": 1, "score": 3, "reason": "Repeats earlier explanation — safe to cut"},
  ...
]
```

Score 0–10 per scene. Reason stored in `scores_{ratio}.json`.

**Model routing:**
- Ratio ≤ 3 → LLM not called. All scores default to 5.0 (rule-based trim)
- Ratio 4–6 → GLM-4.7 standard mode
- Ratio 7–10 → GLM-4.7 thinking mode (harder editorial decisions)

**Composite score:**
```
composite = 0.45 × llm_score        (semantic importance)
          + 0.30 × audio_emphasis   (librosa RMS + pitch per scene)
          + 0.25 × visual_salience  (librosa RMS proxy per scene)
```

Why these weights:
- LLM highest (0.45) — understands meaning, context, narrative
- Audio second (0.30) — prosodic emphasis is real signal, language-agnostic
- Visual lowest (0.25) — weakest proxy, talking head = no visual change but high info

### SELECT — three strategies by ratio band

**Light trim 2–3× — rule-based, no LLM:**
- Removes silences >1.5s
- Removes filler words (um/uh/like/you know)
- Removes repeated sentences (cosine sim >0.92)
- ~1s to run, zero API cost

**Moderate 4–6× — submodular greedy:**
```
Sort all scenes by composite_score descending
For each scene:
  if adding it exceeds target duration → skip
  if cosine sim > 0.82 with any already-selected scene → skip (diversity)
  else → select it
Return selected scenes in chronological order
```
Diversity check critical — without it selector collapses onto single best topic.

**Highlight reel 7–10× — topic-cluster representative:**
```
TextTiling segments = clusters (e.g. 5 topic clusters)
Rank clusters by sum of composite scores
Select top K clusters: K = ceil(topic_count × (1 - (ratio-7)/10))
Within each cluster → GLM-4.7 thinking picks single best scene
Return K scenes in chronological order
```
TextTiling enforces full video coverage — one scene per topic.
Without this, 7× compression would only show the highest-scoring 2 minutes.

### RENDER — ffmpeg stream copy
- Default: stream copy (`-c copy`) — no re-encoding, ~5s, zero quality loss
- Only works because input is already normalized h264 (from INGEST step)
- Re-encode only if crossfades requested (~45s, off by default)
- Output: `output_{ratio}.mp4`

---

## Caching strategy

Everything keyed by SHA-256 hash of raw video bytes:

| Artifact | Cache scope |
|----------|------------|
| normalized.mp4 | Permanent |
| transcript.json | Permanent |
| movinet_features.npy | Permanent |
| audio_features.json | Permanent |
| topic_segments.json | Permanent |
| density.json | Permanent |
| scores_{ratio}.json | Per ratio, permanent |
| selection_{ratio}.json | Per ratio, permanent |
| output_{ratio}.mp4 | Per ratio, permanent |
| LLM scores (Redis) | 24h TTL |

Same video uploaded twice = zero reprocessing. Phase 1 cached forever.
Phase 2 re-runs only if new ratio requested.

---

## Database schema

### jobs
```
job_id          TEXT PRIMARY KEY
video_hash      TEXT
name            TEXT
status          TEXT  (pending/analyzing/awaiting_ratio/summarizing/done/failed)
ratio           INTEGER
video_type      TEXT  (lecture/interview/demo/podcast/tutorial/short_film/movie/episode/unknown)
video_path      TEXT  (original upload path)
density_result  TEXT  (JSON blob)
output_path     TEXT
error           TEXT
created_at      TEXT
updated_at      TEXT
```

### job_metrics
One row per pipeline stage per job. Powers the scorecard.
```
job_id          TEXT
stage           TEXT  (INGEST/TRANSCRIBE/EMBED/SEGMENT/DENSITY/SCORE/SELECT/RENDER)
model           TEXT  (ffmpeg/whisper-large-v3/MoViNet-A2-Stream/GLM-4.7/etc.)
elapsed_s       REAL
cost_usd        REAL
input_tokens    INTEGER
output_tokens   INTEGER
audio_s         REAL
extra           TEXT  (JSON blob — stage-specific details)
```

### exports
One row per render. Stores eval metrics.
```
job_id              TEXT
video_hash          TEXT
ratio               INTEGER
output_path         TEXT
original_duration_s REAL
output_duration_s   REAL
file_size_mb        REAL
render_elapsed_s    REAL
eval_metrics        TEXT  (JSON: ratio_accuracy, topic_coverage, score_gap, distribution)
```

### feedback
```
job_id              TEXT
rating              INTEGER  (1–5)
comment             TEXT
actual_ratio_used   INTEGER
```

---

## API endpoints

```
POST /jobs
  Body: {video_path, ratio, name, video_type}
  → hash video, create job in DB, dispatch phase1 to Celery
  → returns job_id immediately (<200ms)

GET /jobs/{id}
  → returns current status + density_result if available

GET /jobs/{id}/report
  → returns job_metrics + exports + scores_by_ratio (scorecard data)

POST /jobs/{id}/confirm-ratio
  Body: {ratio}
  → dispatches phase2 to Celery

POST /jobs/{id}/retry
  → resumes from furthest completed checkpoint:
    has density.json → restore to awaiting_ratio (free)
    has scores → re-run select + render only
    nothing → re-run phase 1 (needs original video file)

POST /jobs/{id}/re-summarize
  Body: {ratio}
  → re-runs phase 2 at new ratio (phase 1 fully cached)

POST /jobs/{id}/feedback
  Body: {rating, comment}

GET /health
  → API status + Groq free tier usage
```

---

## Celery queue design

```
API process                    Worker process
    │                               │
    ├─ phase1_task.delay()          ├─ pulls from phase1 queue
    │   args: job_id,               │   → _run_phase1()
    │         video_hash,           │
    │         video_path            │
    │                               │
    └─ phase2_task.delay()          └─ pulls from phase2 queue
        args: job_id,                   → _run_phase2()
              video_hash,
              ratio
```

**Worker settings:**
- `--pool=threads` — keeps heartbeat alive during long ffmpeg/MoViNet calls
- `--concurrency=1` — CPU bottleneck, one task at a time
- `--queues=phase1,phase2` — single worker handles both queues
- `task_acks_late=True` — task acknowledged only after completion, not on receipt
- `task_track_started=True` — status visible in monitoring

---

## End-to-end request flow

```
1. User uploads video via Streamlit UI

2. POST /jobs
   → SHA-256 hash computed
   → job row inserted (status=analyzing)
   → phase1_task dispatched to Redis
   → job_id returned to UI in <200ms

3. Celery worker picks up phase1_task
   → INGEST: normalize + extract audio + sample frames
   → TRANSCRIBE: Groq Whisper → transcript.json
   → EMBED: MoViNet + librosa → features
   → SEGMENT: TextTiling → topic_segments.json
   → DENSITY: compute score → density.json
   → upsert_job(status=awaiting_ratio)
   → each stage writes row to job_metrics

4. UI polls GET /jobs/{id} every 3s
   → sees awaiting_ratio
   → displays density score, signals, recommendation

5. User picks ratio (e.g. 5×), clicks Confirm

6. POST /jobs/{id}/confirm-ratio
   → phase2_task dispatched to Redis

7. Celery worker picks up phase2_task
   → SCORE: GLM-4.7 scores all scenes → scores_5.json
   → SELECT: submodular greedy → selection_5.json
   → RENDER: ffmpeg stream copy → output_5.mp4
   → upsert_job(status=done)
   → insert_export() with eval metrics

8. UI polls → sees done
   → shows video player + download button + report card
```

---

## Cost per 30-min video

| Stage | Model | Cost |
|-------|-------|------|
| TRANSCRIBE | Groq Whisper large-v3 | $0.055 |
| SCORE (ratio 4–6×) | GLM-4.7 standard | $0.003 |
| SCORE (ratio 7–10×) | GLM-4.7 thinking | $0.013 |
| SEGMENT labels | GLM-4.7 flash | $0.001 |
| Everything else | CPU | $0.000 |
| **Total** | | **$0.058–$0.069** |

Groq = 80% of total cost. Biggest lever: Groq Batch API = 50% discount.

---

## Upgrade path

```
SQLite    → Postgres    : DATABASE_BACKEND=postgres
Local     → S3          : STORAGE_BACKEND=s3
Memory    → Redis cache : CACHE_BACKEND=redis
Polling   → SSE         : add EventSourceResponse endpoint
Single    → Worker pool : separate phase1/phase2 K8s Deployments
NIM free  → NIM paid    : same endpoint, higher rate limits
On-demand → Batch API   : GROQ_USE_BATCH=true (50% saving)
NIM       → Self-hosted : at 800+ videos/day on 2× A100 80GB
```

---

*Last updated: 2026-05-01*
