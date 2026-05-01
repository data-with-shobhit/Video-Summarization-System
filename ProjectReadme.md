# VideoSum — System Design, Scaling & Observability

---

## What the system does

Takes any uploaded video and produces a semantically compressed output video at a
user-chosen compression ratio (2×–10×). Unlike text-summary tools, the output is
always a shorter watchable video. The system analyzes content across three modalities
(visual, audio, transcript), computes a density score, recommends a ratio, and renders
the output using the best selection strategy for that compression band.

---

## Example outputs

### Sam Altman × Nikhil Kamath — podcast interview

| | Link |
|--|------|
| Original | [YouTube](https://www.youtube.com/watch?v=SfOaZIGJ_gs) |
| Compressed | [Google Drive](https://drive.google.com/file/d/1AT5nb-7XKq69_Am6ccjLvQ5rzLzzbl9t/view?usp=sharing) |

---

### Marques Brownlee — Pixel 9 review

| | Link |
|--|------|
| Original | [YouTube](https://www.youtube.com/watch?v=EGkGRs6YhoM) |
| Compressed | [Google Drive](https://drive.google.com/file/d/1swzh5upX9DLwIeYQIrvEIuWnYPKn_k7r/view?usp=drive_link) |

---

## Pipeline overview

```
Upload
  │
  ▼
Phase 1 — Analyze (~2.5 min, fully cached after first run)
  INGEST → TRANSCRIBE → EMBED → SEGMENT → DENSITY
  Output: density score + recommended ratio + signal breakdown
  
  ↓ user confirms ratio ↓

Phase 2 — Summarize (~35 sec)
  SCORE → SELECT → RENDER
  Output: compressed video file
```

---

## Every step — decision rationale

### INGEST — ffmpeg
Normalizes raw upload to constant 30fps h264 + 16kHz mono WAV + 1fps JPEG frames.

- **Why constant 30fps:** ffmpeg stream copy (5s render) only works when input codec matches
  output. Variable frame rate breaks it. Normalize once upfront, save 40s on every render.
- **Why 16kHz mono WAV:** Groq charges by audio duration. Mono halves data. 16kHz is
  Whisper's minimum — anything higher is wasted bandwidth. Reduces file size 6× vs stereo 44kHz.
- **Why 1fps frames:** MoViNet needs temporal context across frames, not high-frequency
  sampling. 1fps captures motion patterns without storage overhead.

### TRANSCRIBE — Groq Whisper large-v3
Speech to text with word-level timestamps.

- **Why Groq over self-hosted:** Self-hosted Whisper large-v3 on CPU = 25–40 min for a
  30-min video. Groq = 6 sec. $0.055/video is cheaper than running a GPU server at any
  volume below ~5,000 videos/day.
- **Why large-v3 over turbo:** large-v3 = 99 languages. Turbo = English only, half the
  cost. Use turbo only for confirmed English-only pipelines.
- **Chunking:** Videos >24 min split into chunks with 30s overlap. Overlap deduplicates
  words at boundaries so no content is lost at cut points.

### EMBED — MoViNet-A2-Stream + librosa
Visual temporal embeddings + audio feature extraction.

- **Why MoViNet over CLIP:** CLIP processes each frame in total isolation — zero temporal
  awareness. MoViNet-A2-Stream uses causal (2+1)D convolutions with a stream buffer —
  each frame is processed with memory of all prior frames. 5× faster on CPU AND
  temporally aware. CLIP would miss "this scene is the climax of a 3-minute buildup."
- **Why librosa for audio:** RMS energy + pitch variance (F0) captures prosodic emphasis
  invisible in the transcript. Loud + high-pitch = the speaker is making an important point.
  Pure transcript analysis misses this entirely.

### SEGMENT — sentence-transformers + TextTiling
Topic boundary detection with timestamps.

- **Why TextTiling over BERTopic:** BERTopic is designed for large document corpora —
  it clusters globally across thousands of documents. For a single video you need
  segmentation (where do topics change?) not modeling (what topics exist?). TextTiling
  uses the sentence-transformer embeddings already computed, adds 2s of cosine similarity
  math, and returns segments with timestamps mapped directly to the video timeline.
  Zero extra dependencies, zero extra cold-start time.
- **Why topic count matters:** At 7–10× compression, the selector picks one representative
  scene per topic. Without segmentation, high compression would just pick the top 10%
  of scenes by score — all from the most exciting 2-minute window. TextTiling forces
  coverage of the entire video.

### DENSITY SCORER — pure Python
Single number 0.0–1.0 quantifying how information-dense the content is.

```
compressibility = 0.30 × semantic_redundancy
               + 0.20 × silence_ratio
               + 0.15 × filler_word_rate

density = 0.20 × lexical_density
        + 0.10 × normalize(topic_count)
        + 0.05 × visual_change_rate

score = clip(density - compressibility + 0.5, 0, 1)
```

- **Why this formula:** Compressibility signals (redundancy, silence, fillers) pull the
  score down — high compressibility = safe to cut more. Density signals (vocabulary
  richness, topic diversity, visual dynamism) pull it up — dense content = cut less.
- **Why this matters for the product:** No existing tool tells you how much to compress.
  You get a fixed output. Here the system gives a calibrated recommendation based on
  actual content analysis, with a confidence interval.

### SCORE — GLM-4.7 via NVIDIA NIM
Per-scene importance scoring 0–10.

- **Why LLM for scoring:** Audio tells you where it's energetic. Visual tells you where
  there's motion. Neither tells you where the important content is. Only language
  understanding can distinguish "introduces key concept for the first time" from
  "repeats the same point for the third time."
- **Why GLM-4.7 over Claude/GPT:** ~10× cheaper than Claude Haiku for structured JSON
  scoring. Open weights — self-hostable on 2× A100 80GB at 800+/day. Free tier covers
  500 videos/day before any payment.
- **Model routing:** ratio ≤ 3 → skip LLM (rule-based trim doesn't need editorial
  judgment). ratio 4–6 → standard mode. ratio 7–10 → thinking mode (harder editorial
  decisions at high compression require deeper reasoning).
- **Genre-aware prompting:** System prompt changes by content type. Interview gets
  "keep the sharpest exchanges." Tutorial gets "keep every instructional step."
  Same model, different editorial lens, meaningfully better output.

### SELECT — three strategies by ratio band

**Light trim (2–3×) — rule-based:**
Removes silences >1.5s, filler words, repeated sentences (cosine sim >0.92).
No LLM. Zero API cost. ~1s to run.

**Moderate (4–6×) — submodular greedy:**
```
Sort by composite_score descending
For each scene:
  if exceeds target duration → skip
  if too similar to already selected (cosine sim > 0.82) → skip
  else → select
Return in chronological order
```
The diversity check is critical — without it you'd pick 5 scenes from the same
2-minute window.

**Highlight reel (7–10×) — topic-cluster representative:**
1. TextTiling segments = clusters
2. Rank clusters by total score
3. Select top K clusters: `K = ceil(topic_count × (1 - (ratio-7)/10))`
4. GLM-4.7 thinking picks best single scene per cluster
5. Return chronologically

### RENDER — ffmpeg stream copy
Stitches selected segments into output video.

- **Why stream copy by default:** No re-encoding. ~5s for any length video. Quality
  identical to input. Saves 40s per video = 66 CPU-hours/day at 100 videos/day.
- **When to re-encode:** Only if user requests crossfade transitions (~45s, off by default).

---

## Queuing architecture

### Current (POC)
```
API → Celery → Redis broker → Worker (threads, concurrency=1)
```
Two queues: `phase1`, `phase2`. Worker pulls from both.
`task_acks_late=True` — task only acknowledged after completion. If worker dies,
task requeues after `visibility_timeout` (3600s).

### Production target
```
                    ┌─────────────────────────────┐
Load Balancer       │   API cluster (3+ replicas)  │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │   Redis (ElastiCache)         │
                    │   - Celery broker             │
                    │   - Result backend            │
                    │   - LLM score cache (24h TTL) │
                    └──────────────┬──────────────┘
                          ┌────────┴────────┐
                          ▼                 ▼
               Phase1 workers (N)   Phase2 workers (M)
               (CPU-heavy: 8 core   (API-heavy: 2 core
                t3.2xlarge spot)     t3.medium spot)
```

**Why separate phase1/phase2 worker pools:**
- Phase 1 is CPU-bound (MoViNet, librosa, ffmpeg). Needs 8 cores, runs 2.5 min.
- Phase 2 is API-bound (waits on NVIDIA NIM). Needs 2 cores, mostly I/O wait.
- Mixing them on the same machine means CPU workers starve API workers during peak.

**Queue depth monitoring:**
- Alert if phase1 queue depth > 10 (workers falling behind)
- Alert if phase2 queue depth > 50 (NVIDIA NIM rate limit likely)
- Auto-scale phase1 workers when queue depth > 5

**Dead letter queue:**
- Failed tasks after 3 retries → DLQ
- DLQ triggers PagerDuty alert + Slack notification
- Manual inspection before replay

### Why not Temporal
Celery is simpler for this workflow — two linear phases, no complex branching.
Temporal becomes valuable when you need: sub-workflow retry (retry just the
Groq call, not the whole ingest), workflow versioning, or cross-service
saga coordination. At current scale, Celery overhead is zero and debugging
is simpler.

---

## Real-time & low latency

### What "low latency" means here
Phase 1 is inherently 2.5 min (Groq is 6s, MoViNet is 30s, unavoidable).
"Low latency" means:
1. **Upload response < 200ms** — job accepted immediately, processing async
2. **Progress updates every 3s** — user sees live stage progression
3. **Phase 2 starts < 1s after ratio confirm** — no queuing delay
4. **Download available < 5s after phase 2 completes** — stream copy render

### Current polling (POC)
Streamlit UI polls `GET /jobs/{id}` every 3s. Simple, works, not scalable
beyond ~100 concurrent users (each polling = 1 req/3s = 2,000 req/min for
100 users).

### Production: Server-Sent Events (SSE)
Replace polling with SSE push:
```python
@router.get("/{job_id}/stream")
async def stream_status(job_id: str):
    async def event_generator():
        while True:
            job = get_job(job_id)
            yield {"data": json.dumps({"status": job["status"]})}
            if job["status"] in ("done", "failed"):
                break
            await asyncio.sleep(2)
    return EventSourceResponse(event_generator())
```
Benefits: single persistent connection per job, server pushes updates,
no polling overhead.

### Model pre-warming
MoViNet has a 3s cold-start on first video (TF SavedModel initialization).
In production, pre-warm on worker startup:
```python
# Called once at worker boot
movinet_model = load_model()  # pre-warms, stays in memory
```
Keeps the model resident in RAM. First video pays 3s cold-start, every
subsequent video pays 0s.

### LLM score caching
Cache key: `sha256(segment_text + video_type + str(ratio))`
TTL: 24 hours. Expected cache hit rate: 15–30% on repeat/similar content
(conference talks, product demos with similar intros).
Saves $0.003–$0.013 per cache hit.

### Deduplication by video hash
SHA-256 of raw video bytes. Same video uploaded twice = zero re-processing.
Phase 1 artifacts cached indefinitely. Phase 2 scores cached 24h.
At 100 videos/day with 20% re-upload rate: saves 20 × $0.055 = $1.10/day
in Groq costs.

---

## Observability

### Three pillars

#### 1. Structured logging (implemented)
Every stage emits a JSON log line:
```json
{
  "job_id": "cf281d84",
  "stage": "TRANSCRIBE",
  "model": "whisper-large-v3",
  "duration_ms": 48410,
  "cost_usd": 0.04435,
  "status": "done",
  "language": "en",
  "segments": 312,
  "total_words": 4821
}
```
Shipped to: CloudWatch Logs (AWS) → Athena for SQL queries.

#### 2. Metrics (to implement)
Key metrics to track per stage:

| Metric | Alert threshold |
|--------|----------------|
| phase1_duration_p95 | > 5 min |
| phase2_duration_p95 | > 3 min |
| groq_cost_per_video_p95 | > $0.15 |
| llm_score_cache_hit_rate | < 10% |
| queue_depth_phase1 | > 10 |
| queue_depth_phase2 | > 50 |
| job_failure_rate_5m | > 5% |
| render_codec_mode | stream_copy vs re-encode ratio |

Ship to: CloudWatch Metrics → Grafana dashboard.

#### 3. Distributed tracing (to implement)
Trace ID generated at job creation, propagated through Celery tasks.
```python
trace_id = job_id  # already unique, reuse as trace ID
```
Every log line, every DB write, every API call includes `trace_id`.
In production: OpenTelemetry → Jaeger or AWS X-Ray.

### Weekly metrics to review
- Groq cost/video (target: < $0.06)
- GLM cost/video by ratio band
- TextTiling segment count distribution (outliers = threshold needs tuning)
- LLM score distribution per ratio band (all scores clustering at 5 = model confusion)
- User-selected ratio vs recommended ratio (how often users override)
- User ratings (1–5) by content type and ratio band
- Cache hit rate by content type

### Alerting strategy
```
P0 (page immediately):
  - Job failure rate > 10% over 5 min
  - Groq API returning 5xx
  - Worker queue stalled (no tasks processed in 10 min)

P1 (Slack alert, fix within 1hr):
  - NVIDIA NIM 429 rate limit sustained > 15 min
  - Phase 1 p95 > 8 min (MoViNet slowdown)
  - DB connection pool exhausted

P2 (review next day):
  - LLM cache hit rate dropping
  - Cost per video trending up
  - TextTiling producing > 15 segments consistently
```

---

## Scaling ceilings and fixes

| Bottleneck | Ceiling | Fix |
|-----------|---------|-----|
| Groq free tier | 240 videos/hr | Upgrade to paid ($50/mo removes ceiling) |
| NVIDIA NIM free tier | 500 videos/day | Upgrade NIM tier |
| ffmpeg render (re-encode) | 80 videos/day/machine | Use stream copy (720/day) or GPU NVENC |
| MoViNet CPU | 20 videos/hr/worker | Add more CPU workers (linear scale) |
| Redis connections | ~100 concurrent | ElastiCache r6g.large handles 65,000 conn |
| Postgres | ~500 req/s single node | RDS read replicas for dashboard queries |

### Self-hosting break-even
GLM-4.7-flash on 2× A100 80GB: ~$3/hr on-demand, handles ~500 req/hr.
NVIDIA NIM paid tier: ~$0.003/request.
Break-even: 3 / 0.003 = 1,000 req/hr = **500 videos/hr** before self-hosting wins.

### Groq Batch API
50% discount on transcription. Async — 24hr processing window.
Switches $0.055 → $0.028/video.
Zero code changes — just `GROQ_USE_BATCH=true` in `.env`.
Worth enabling at any volume for non-real-time jobs (nightly batch processing).

---

## Storage architecture

### POC — local filesystem
```
./data/videos/{sha256_hash}/
  source.{ext}              # original upload
  normalized.mp4            # constant 30fps
  audio.wav                 # 16kHz mono
  transcript.json           # Groq output
  scenes.json               # PySceneDetect boundaries
  movinet_features.npy      # (N_frames, 600) float32
  audio_features.json       # librosa per scene
  topic_segments.json       # TextTiling output + labels
  density.json              # score + signals + recommendation
  scores_{ratio}.json       # GLM composite scores
  selection_{ratio}.json    # selected (start_s, end_s) pairs
  output_{ratio}.mp4        # final output
```

### Production — S3
Set `STORAGE_BACKEND=s3`. Keys identical. Zero code changes.
```
s3://videosum-artifacts/videos/{sha256_hash}/...
```
- Lifecycle policy: delete artifacts after 7 days (output videos kept 30 days)
- Transfer acceleration for uploads from non-AWS regions
- S3 presigned URLs for direct browser download (bypasses API server)

### Why hash-keyed storage
Same video uploaded twice = same hash = zero reprocessing.
Dedup is automatic and free. Useful for: users re-uploading after browser crash,
batch processing pipelines sending same content multiple times.

---

## Database

### POC — SQLite
Single file, zero ops. Sufficient up to ~10 concurrent users.

### Production — Postgres RDS
`DATABASE_BACKEND=postgres`. Run `init_db()` migrations.

**Tables:**
- `jobs` — job lifecycle, status, density result, video_path
- `job_metrics` — per-stage timing, model, cost (the scorecard data)
- `exports` — compression history + eval metrics per ratio
- `feedback` — user ratings 1–5 + comments

**Why separate metrics table:**
Enables SQL queries like:
```sql
-- Average Groq cost per video this week
SELECT AVG(cost_usd) FROM job_metrics
WHERE stage = 'TRANSCRIBE' AND created_at > NOW() - INTERVAL '7 days';

-- Which content types have highest LLM score variance
SELECT j.video_type, STDDEV(cost_usd) as cost_std
FROM job_metrics jm JOIN jobs j USING (job_id)
WHERE jm.stage = 'SCORE'
GROUP BY j.video_type;
```

**Read replicas:**
Dashboard queries (queue_dashboard, report cards) hit read replica.
Write path (job updates, metric inserts) hits primary.
Prevents dashboard load from affecting job processing.

---

## Cost at scale

| Volume | Groq | GLM | Infra | Total/video | Monthly |
|--------|------|-----|-------|-------------|---------|
| 10/day | $0.055 | $0.008 | $0.00 | $0.063 | ~$19 |
| 100/day | $0.055 | $0.008 | $0.087 | $0.150 | ~$450 |
| 300/day | $0.028* | $0.006 | $0.062 | $0.096 | ~$864 |
| 500/day | $0.028* | $0.005 | $0.055 | $0.088 | ~$1,320 |

*Groq Batch API (50% discount, async jobs)

**Single biggest cost lever:** Enable `GROQ_USE_BATCH=true`. Drops total from
$0.150 → $0.093 per video at 100/day. Saves $1,700/month at 100 videos/day.

---

## Upgrade path — POC to production

```
Step 1: Local → S3
  STORAGE_BACKEND=s3
  Zero code changes.

Step 2: SQLite → Postgres
  DATABASE_BACKEND=postgres
  Run init_db() migrations.

Step 3: Memory cache → Redis
  CACHE_BACKEND=redis
  Zero code changes.

Step 4: Single worker → worker pool
  Deploy phase1 workers (CPU-optimized) and phase2 workers (compute-optimized)
  as separate K8s Deployments with independent HPA scaling.

Step 5: Polling → SSE
  Add EventSourceResponse endpoint.
  Update Streamlit UI to use SSE instead of polling loop.

Step 6: Groq on-demand → Groq Batch
  GROQ_USE_BATCH=true
  50% cost reduction on transcription.

Step 7: NVIDIA NIM free → NIM paid
  Same endpoint, higher rate limits.

Step 8 (800+/day): NVIDIA NIM → self-hosted GLM-4.7-flash
  2× A100 80GB. Break-even at 800+ videos/day.
```

---

## Core package — `packages/core`

Zero infrastructure dependencies. Pure Python + numpy + pydantic only.
No imports from `apps/`, `packages/ml/`, or `packages/storage/`.
Workers import from core — never the other way around.
This means every function here is unit-testable with no mocks, no DB, no API keys.

---

### `density/signals.py` — 5 signal extractors

These are the raw inputs to the density scorer. Each returns a float 0–1.

#### `compute_silence_ratio(audio_features)`
```
silence_duration = sum of (scene_duration × scene_silence_ratio) for each scene
output = silence_duration / total_duration
```
Source: librosa per-scene `silence_ratio` field (fraction of frames with rms < 0.01).
High value → lots of dead air → safe to compress more.

#### `compute_filler_word_rate(transcript_text)`
```
filler words = {um, uh, like, you know, basically, literally, actually, so}
rate = (filler_count / total_words) × 100
output = clip(rate / 20.0, 0, 1)    # 20 fillers/100 words = 1.0
```
Source: Groq transcript text. High value → casual, padded speech → compress more.

#### `compute_pacing_variance(audio_features)`
```
wpms = [scene.speech_rate_wpm for scene if wpm > 0]
output = clip(std(wpms) / 100.0, 0, 1)    # 100 WPM std = 1.0
```
Source: WPM filled by segment worker using Groq word timestamps.
High variance → dynamic pacing → likely denser content.

#### `compute_lexical_density(transcript_text)`
```
output = unique_words / total_words    # type-token ratio
```
Source: Groq transcript. Higher = richer vocabulary = denser content.
A lecture on ML: ~0.55. Casual podcast: ~0.35.

#### `normalize_topic_count(topic_count, video_duration_s)`
```
topics_per_minute = topic_count / (duration / 60)
output = clip(topics_per_minute / 1.0, 0, 1)    # 1 topic/min = 1.0
```
Source: TextTiling segment count. More topics per minute = denser content.

---

### `density/scorer.py` — `compute_density_score(signals)`

Pure function. Takes all 7 signals, returns `DensityResult`.

```python
compressibility = 0.30 × semantic_redundancy   # pulls score DOWN (safe to cut)
               + 0.20 × silence_ratio
               + 0.15 × filler_word_rate

density        = 0.20 × lexical_density        # pulls score UP (cut less)
               + 0.10 × normalize(topic_count)
               + 0.05 × visual_change_rate

score = clip(density - compressibility + 0.5, 0.0, 1.0)
```

**Why 0.5 offset:** Centers the scale. If all signals are average (0.5), score = 0.5 → 3–4× recommendation. Without offset, average content would score near 0 and always get max compression.

**Ratio recommendation:**
```
score ≥ 0.80 → None  (do not compress — content too dense)
score ≥ 0.60 → 2×   (±1)
score ≥ 0.40 → 3×   (±1, center of 3–4× band)
score ≥ 0.20 → 6×   (±2, center of 5–7× band)
score <  0.20 → 9×  (±2, center of 8–10× band)
```

The confidence interval (±1 or ±2) tells the UI what range is safe to offer the user.

---

### `segmentation/texttiling.py`

#### `align_transcript_to_scenes(transcript_segments, scenes)`
Prereq step before TextTiling can run. Merges Groq word-level transcript into
one text chunk per PySceneDetect scene. Each chunk = all words spoken during that scene.
Returns `[{"text": str, "start_s": float, "end_s": float}, ...]`.

#### `find_boundaries(chunks, threshold=0.35)`
```
1. embed each chunk → 384-dim vector (all-MiniLM-L6-v2, already normalized)
2. for each adjacent pair: sim = dot(emb[i-1], emb[i])
3. sim < threshold → topic boundary → start new segment
4. merge any segment shorter than TEXTTILING_MIN_SEGMENT_SECONDS (default 60s)
   into its neighbor to avoid tiny fragments
5. re-index and return segments with timestamps
```

**Threshold tuning:**
- 0.35 → structured content (lectures, demos, tutorials)
- 0.25 → casual content (interviews, podcasts) — more sensitive
- Too many tiny segments → raise threshold
- Missing obvious topic shifts → lower threshold

#### `segmentation/labels.py` — `maybe_label_segments(segments)`
Calls `label_all_segments()` in nvidianim.py if `SEGMENT_LABELING_ENABLED=true`.
Each segment gets a 3–5 word human-readable label ("Introduction", "Live Demo", "Q&A").
Falls back to "Segment 1/2/3" if disabled or API fails.

---

### `selection/light.py` — 2–3× rule-based

No LLM. ~1s to run. Zero API cost.

```
For each scene:
  if silence_ratio > 0.85 → drop (pure silence)
  else → keep

Deduplicate: cosine sim > 0.92 with any kept scene → drop (near-identical content)

Trim to target: walk kept scenes in order, stop when target duration reached
```

Quality target: ≥ 92% of meaningful content retained. Designed for talks where
the content is good but padded with dead air and verbal filler.

---

### `selection/moderate.py` — 4–6× submodular greedy

```
composite_score = 0.45 × llm_score       (normalized 0–1)
               + 0.30 × audio_emphasis   (rms_energy + pitch_variance) / 2
               + 0.25 × visual_salience  (MoViNet frame-to-frame L2, normalized)

sort all scenes by composite_score descending

for each scene:
  if selected_duration + scene_duration > target × 1.05 → skip
  if max cosine_sim(scene, any selected scene) > 0.82 → skip  (diversity)
  else → select

return selected in chronological order
```

**Why the diversity check matters:** Without it, the top 10 scenes by score are
often from the same 2-minute "peak" window. The diversity threshold forces the
selector to spread across the whole video.

**Why 0.82 not 0.92:** More aggressive than light trim. At 4–6× you're making
real editorial cuts — near-similar content (0.82 sim) is redundant enough to drop.

---

### `selection/highlight.py` — 7–10× topic-cluster representative

```
K = ceil(topic_count × (1 - (ratio - 7) / 10))

examples:
  5 topics at 7×  → K = ceil(5 × 1.0)  = 5  (keep all topics)
  5 topics at 8×  → K = ceil(5 × 0.9)  = 5
  5 topics at 9×  → K = ceil(5 × 0.8)  = 4
  5 topics at 10× → K = ceil(5 × 0.7)  = 4

for each TextTiling segment:
  segment_score = sum of composite scores of all scenes inside it

rank segments by segment_score descending, take top K

for each of the K segments:
  best_scene = scene with highest composite_score inside this segment

return K scenes in chronological order
```

**Why topic-cluster at 7–10×:** Greedy composite scoring at extreme compression
always collapses to the top 2 minutes. TextTiling forces one scene per topic
cluster — guarantees the output covers the full video arc even at 10×.

---

### `scoring/composite.py` — `build_composite_scores(scenes, llm_scores, audio_features, movinet_features)`

Assembles the final per-scene composite score from three lanes:

```python
llm_s          = llm_scores[scene_index]["llm_score"] / 10.0   # normalize to 0-1

audio_emphasis = clip((rms_energy + pitch_variance) / 2, 0, 1)  # from librosa

visual_salience:
  if movinet_features available:
    diffs = L2 distance between consecutive frame embeddings
    visual_salience = mean(diffs[start_frame:end_frame]), normalized by max
  else:
    visual_salience = 0.5  # neutral fallback

composite = 0.45 × llm_s + 0.30 × audio_emphasis + 0.25 × visual_salience
```

Output stored as `scores_{ratio}.json`. Used by both moderate and highlight selectors.

---

### `models.py` — shared Pydantic types

| Type | Used by |
|------|---------|
| `DensitySignals` | Input to `compute_density_score` (7 fields) |
| `DensityResult` | Output: score, ratio, CI, signals dict, message |
| `Word` | Groq word with start/end timestamps |
| `TranscriptSegment` | One Whisper segment with words list |
| `Transcript` | Full transcript: language + segments |
| `Scene` | PySceneDetect boundary: scene_index, start_s, end_s, duration_s |
| `TopicSegment` | TextTiling output: timestamps + chunk_indices + label |
| `SegmentScore` | Composite score: llm_score, audio_emphasis, visual_salience, composite_score, reason |
| `SelectionSegment` | Selector output: start_s, end_s, scene_index, topic_segment_index |
| `AudioFeatures` | librosa per-scene: rms_energy, pitch_variance, speech_rate_wpm, silence_ratio |
| `JobStatus` | Enum: pending/analyzing/awaiting_ratio/summarizing/done/failed |
| `VideoType` | Enum: lecture/interview/demo/podcast/tutorial/short_film/movie/episode/unknown |
| `Ratioband` | Enum: light/moderate/highlight |
| `JobCreate`, `JobResponse` | API request/response shapes |
| `ConfirmRatioRequest`, `ResummarizeRequest`, `FeedbackRequest` | API body types |

---

## Competitive moat

Every existing tool (NoteGPT, Eightify, Vimeo AI, Otter.ai) outputs **text**.
None output a shorter video. Vimeo AI produces a video recap but only within
their platform, no ratio control, no density analysis.

This system's moat:
1. **Video-to-video output** — not text
2. **Content-density-aware ratio recommendation** — no other tool does this
3. **Three selection strategies** matched to compression band
4. **TextTiling topic segmentation** mapped directly to video timeline
5. **Chapter labels** baked into output (human-readable, not "Segment 1")
6. **Feedback loop** — every rated job improves the density scorer over time

The feedback dataset is the long-term moat. Log all signals, scores, ratios,
segment counts, and user ratings from day one. At 500 videos/day that is
180,000 labeled training examples per year. Fine-tune the density scorer
weights and LLM prompts on that data. Competitors starting later start with
zero data.

---

*Last updated: 2026-05-01*
