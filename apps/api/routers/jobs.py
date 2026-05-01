from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from apps.api.dependencies import get_storage
from apps.logger import get_job_logger, get_logger
from packages.core.models import ConfirmRatioRequest, JobCreate, JobResponse, JobStatus, ResummarizeRequest
from packages.storage.base import StorageBackend
from packages.storage.db import get_job, get_job_report, init_db, insert_export, insert_metric, upsert_job

router = APIRouter(prefix="/jobs", tags=["jobs"])
log = get_logger(__name__)


@router.post("", response_model=JobResponse, status_code=202)
async def create_job(
    body: JobCreate,
    background_tasks: BackgroundTasks,
    storage: StorageBackend = Depends(get_storage),
):
    video_path = body.video_path
    if not Path(video_path).exists():
        raise HTTPException(status_code=400, detail=f"File not found: {video_path}")

    video_hash = _hash_file(video_path)
    job_id = str(uuid.uuid4())
    file_size_mb = round(Path(video_path).stat().st_size / 1024 / 1024, 1)

    name = body.name or Path(video_path).stem
    video_type = body.video_type.value
    log.info(f"New job {job_id} | name={name!r} | type={video_type} | file={video_path} | size={file_size_mb}MB | hash={video_hash[:12]}...")
    initial_ratio = None if body.ratio == "auto" else int(body.ratio)
    upsert_job(job_id, video_hash=video_hash, name=name, video_type=video_type, status=JobStatus.ANALYZING, ratio=None, video_path=video_path)
    _dispatch_phase1(job_id, video_hash, video_path, storage, background_tasks, ratio=initial_ratio)

    return JobResponse(job_id=job_id, status=JobStatus.ANALYZING, video_hash=video_hash, name=name)


@router.get("/{job_id}", response_model=JobResponse)
async def get_job_status(job_id: str):
    row = get_job(job_id)
    if not row:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobResponse(
        job_id=job_id,
        status=row["status"],
        video_hash=row.get("video_hash"),
        density_result=row.get("density_result"),
        output_path=row.get("output_path"),
        error=row.get("error"),
    )


@router.get("/{job_id}/report")
async def get_report(job_id: str, storage: StorageBackend = Depends(get_storage)):
    row = get_job(job_id)
    if not row:
        raise HTTPException(status_code=404, detail="Job not found")
    report = get_job_report(job_id)
    # Attach per-segment scores for each export ratio
    scores_by_ratio = {}
    for exp in report.get("exports", []):
        ratio = exp.get("ratio")
        if ratio:
            key = f"videos/{row['video_hash']}/scores_{ratio}.json"
            try:
                scores_by_ratio[ratio] = json.loads(storage.read_text(key))
            except Exception:
                pass
    report["scores_by_ratio"] = scores_by_ratio
    return report


@router.post("/{job_id}/re-summarize", response_model=JobResponse)
async def re_summarize(
    job_id: str,
    body: ResummarizeRequest,
    background_tasks: BackgroundTasks,
    storage: StorageBackend = Depends(get_storage),
):
    """Re-run phase 2 at a different ratio. Phase 1 data is reused from cache."""
    row = get_job(job_id)
    if not row:
        raise HTTPException(status_code=404, detail="Job not found")
    if row["status"] not in (JobStatus.DONE, JobStatus.AWAITING_RATIO, JobStatus.FAILED):
        raise HTTPException(status_code=400, detail=f"Cannot re-summarize job with status={row['status']}")

    video_hash = row["video_hash"]
    base = storage.video_dir(video_hash)
    if not storage.exists(f"{base}/density.json"):
        raise HTTPException(status_code=400, detail="Phase 1 not complete — run analysis first")

    ratio = body.ratio
    log.info(f"Job {job_id} re-summarize at {ratio}x")
    upsert_job(job_id, status=JobStatus.SUMMARIZING, ratio=ratio, output_path=None, error=None)
    _dispatch_phase2(job_id, video_hash, ratio, storage, background_tasks)
    return JobResponse(job_id=job_id, status=JobStatus.SUMMARIZING, video_hash=video_hash)


@router.post("/{job_id}/retry", response_model=JobResponse)
async def retry_job(
    job_id: str,
    background_tasks: BackgroundTasks,
    storage: StorageBackend = Depends(get_storage),
):
    row = get_job(job_id)
    if not row:
        raise HTTPException(status_code=404, detail="Job not found")
    if row["status"] not in (JobStatus.FAILED, JobStatus.ANALYZING, JobStatus.SUMMARIZING):
        raise HTTPException(status_code=400, detail=f"Job cannot be retried (status={row['status']})")

    video_hash = row["video_hash"]
    base = storage.video_dir(video_hash)

    # Determine furthest completed point and resume from there
    has_density  = storage.exists(f"{base}/density.json")
    has_transcript = storage.exists(f"{base}/transcript.json")
    ratio = row.get("ratio")
    has_scores = ratio and storage.exists(f"{base}/scores_{ratio}.json")

    if has_density:
        # Phase 1 complete — restore to awaiting_ratio
        import json
        density_result = json.loads(storage.read_text(f"{base}/density.json"))
        upsert_job(job_id, status=JobStatus.AWAITING_RATIO, density_result=density_result, error=None)
        log.info(f"Job {job_id} retry: density cached → restored to awaiting_ratio")
        return JobResponse(job_id=job_id, status=JobStatus.AWAITING_RATIO,
                           video_hash=video_hash, density_result=density_result)

    if ratio and has_scores:
        # Phase 2 partial — re-run phase 2 (select + render will be re-run, score cached)
        upsert_job(job_id, status=JobStatus.SUMMARIZING, error=None)
        _dispatch_phase2(job_id, video_hash, ratio, storage, background_tasks)
        log.info(f"Job {job_id} retry: scores cached → re-running phase 2")
        return JobResponse(job_id=job_id, status=JobStatus.SUMMARIZING, video_hash=video_hash)

    # Phase 1 incomplete — re-run from beginning (workers skip cached artifacts)
    video_path = row.get("video_path") or ""
    if not video_path or not Path(video_path).exists():
        raise HTTPException(
            status_code=400,
            detail="Original video file is gone (temp file deleted). Please re-upload the video."
        )
    upsert_job(job_id, status=JobStatus.ANALYZING, error=None)
    _dispatch_phase1(job_id, video_hash, video_path, storage, background_tasks)
    log.info(f"Job {job_id} retry: re-running phase 1 (transcript_cached={has_transcript})")
    return JobResponse(job_id=job_id, status=JobStatus.ANALYZING, video_hash=video_hash)


@router.post("/{job_id}/confirm-ratio", response_model=JobResponse)
async def confirm_ratio(
    job_id: str,
    body: ConfirmRatioRequest,
    background_tasks: BackgroundTasks,
    storage: StorageBackend = Depends(get_storage),
):
    row = get_job(job_id)
    if not row:
        raise HTTPException(status_code=404, detail="Job not found")
    if row["status"] != JobStatus.AWAITING_RATIO:
        raise HTTPException(
            status_code=400,
            detail=f"Job not ready for ratio confirmation (status={row['status']})"
        )

    ratio = body.ratio
    log.info(f"Job {job_id} ratio confirmed: {ratio}x")
    upsert_job(job_id, status=JobStatus.SUMMARIZING, ratio=ratio)
    _dispatch_phase2(job_id, row["video_hash"], ratio, storage, background_tasks)

    return JobResponse(job_id=job_id, status=JobStatus.SUMMARIZING, video_hash=row["video_hash"])


# ── Helpers ───────────────────────────────────────────────────────────────────

def _hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _step(jlog, name: str) -> float:
    jlog.info(f"{'='*60}")
    jlog.info(f"STEP START: {name}")
    jlog.info(f"{'='*60}")
    return time.time()


def _step_done(jlog, name: str, t0: float, job_id: str | None = None,
               model: str | None = None, audio_s: float | None = None,
               input_tokens: int | None = None, output_tokens: int | None = None,
               cost_usd: float | None = None, **details) -> float:
    elapsed = round(time.time() - t0, 2)
    detail_str = " | ".join(f"{k}={v}" for k, v in details.items())
    jlog.info(f"STEP DONE:  {name} | elapsed={elapsed}s{' | ' + detail_str if detail_str else ''}")
    if job_id:
        insert_metric(
            job_id=job_id, stage=name, elapsed_s=elapsed, model=model,
            audio_s=audio_s, input_tokens=input_tokens, output_tokens=output_tokens,
            cost_usd=cost_usd, extra=details if details else None,
        )
    return elapsed


# ── Evaluation metrics ────────────────────────────────────────────────────────

def _compute_eval_metrics(
    scores: list[dict],
    selection: list[dict],
    topic_segs: list[dict],
    orig_dur: float | None,
    output_dur: float,
    ratio: int,
) -> dict:
    import statistics

    # 1. Ratio accuracy
    actual_ratio = round(orig_dur / output_dur, 2) if orig_dur and output_dur else None
    ratio_error_pct = round(abs(actual_ratio - ratio) / ratio * 100, 1) if actual_ratio else None

    # 2. Topic coverage — time-overlap between selection and topic segments
    def _overlaps(ts_start, ts_end, sel_list):
        return any(s["start_s"] < ts_end and s["end_s"] > ts_start for s in sel_list)

    covered = sum(1 for ts in topic_segs if _overlaps(ts["start_s"], ts["end_s"], selection))
    coverage_pct = round(covered / len(topic_segs) * 100, 1) if topic_segs else None

    # 3 & 4. Score gap + distribution
    # scores id = scene_index; selection has scene_index field
    selected_scene_ids = {s.get("scene_index") for s in selection if s.get("scene_index") is not None}
    selected_scores, rejected_scores = [], []
    for s in scores:
        val = float(s.get("score") or s.get("composite_score") or 5)
        if s.get("id") in selected_scene_ids:
            selected_scores.append(val)
        else:
            rejected_scores.append(val)

    # Fallback: if scene_index not in selection, use topic-segment time overlap
    if not selected_scores and not rejected_scores:
        score_ids_covered = set()
        for ts in topic_segs:
            if _overlaps(ts["start_s"], ts["end_s"], selection):
                score_ids_covered.add(ts.get("segment_index", -1))
        for s in scores:
            val = float(s.get("score") or 5)
            if s.get("id") in score_ids_covered:
                selected_scores.append(val)
            else:
                rejected_scores.append(val)

    all_scores = [float(s.get("score") or 5) for s in scores]
    mean_all = round(sum(all_scores) / len(all_scores), 2) if all_scores else None
    std_all = round(statistics.stdev(all_scores), 2) if len(all_scores) > 1 else 0.0
    mean_sel = round(sum(selected_scores) / len(selected_scores), 2) if selected_scores else None
    mean_rej = round(sum(rejected_scores) / len(rejected_scores), 2) if rejected_scores else None
    score_gap = round(mean_sel - mean_rej, 2) if mean_sel is not None and mean_rej is not None else None

    return {
        "actual_ratio": actual_ratio,
        "target_ratio": ratio,
        "ratio_error_pct": ratio_error_pct,
        "topic_coverage_pct": coverage_pct,
        "topics_covered": covered,
        "topics_total": len(topic_segs),
        "score_gap": score_gap,
        "mean_selected_score": mean_sel,
        "mean_rejected_score": mean_rej,
        "score_distribution_mean": mean_all,
        "score_distribution_std": std_all,
    }


# ── Celery / BackgroundTasks dispatch ────────────────────────────────────────

def _dispatch_phase1(job_id, video_hash, video_path, storage, background_tasks, ratio=None):
    if os.environ.get("CELERY_BROKER_URL"):
        from apps.workers.pipeline_tasks import phase1_task
        phase1_task.delay(job_id, video_hash, video_path, ratio)
        log.info(f"Job {job_id} dispatched to Celery phase1 queue")
    else:
        background_tasks.add_task(_run_phase1, job_id, video_hash, video_path, storage, ratio)


def _dispatch_phase2(job_id, video_hash, ratio, storage, background_tasks):
    if os.environ.get("CELERY_BROKER_URL"):
        from apps.workers.pipeline_tasks import phase2_task
        phase2_task.delay(job_id, video_hash, ratio)
        log.info(f"Job {job_id} dispatched to Celery phase2 queue")
    else:
        background_tasks.add_task(_run_phase2, job_id, video_hash, ratio, storage)


# ── Phase 1 ───────────────────────────────────────────────────────────────────

def _run_phase1(job_id: str, video_hash: str, video_path: str, storage: StorageBackend, ratio: int | None = None):
    jlog = get_job_logger(job_id)
    phase_t0 = time.time()

    jlog.info(f"{'#'*60}")
    jlog.info(f"JOB {job_id} — PHASE 1 START")
    jlog.info(f"video_path = {video_path}")
    jlog.info(f"video_hash = {video_hash}")
    jlog.info(f"{'#'*60}")

    if not video_path or not Path(video_path).exists():
        msg = "Original video file is gone (temp file deleted). Please re-upload the video."
        jlog.error(msg)
        upsert_job(job_id, status=JobStatus.FAILED, error=msg)
        return

    try:
        from apps.workers import ingest, transcribe, embed, segment, density as density_worker

        # ── Ingest ────────────────────────────────────────────────
        t = _step(jlog, "INGEST (ffmpeg normalize + audio extract + scene detect + frame sample)")
        jlog.debug(f"input file: {video_path}")
        result = ingest.run(video_hash, video_path, storage)
        norm_size = Path(str(storage.local_path(result["normalized_key"]))).stat().st_size // 1024
        audio_size = Path(str(storage.local_path(result["audio_key"]))).stat().st_size // 1024
        scenes = json.loads(storage.read_text(result["scenes_key"]))
        _step_done(jlog, "INGEST", t, job_id=job_id,
                   model="ffmpeg (CPU)",
                   normalized_kb=norm_size,
                   audio_kb=audio_size,
                   scenes_detected=len(scenes))
        jlog.debug(f"scenes sample: {json.dumps(scenes[:3], indent=2)}")

        # ── Transcribe ────────────────────────────────────────────
        t = _step(jlog, "TRANSCRIBE (Groq Whisper large-v3)")
        audio_key = f"videos/{video_hash}/audio.wav"
        jlog.debug(f"audio key: {audio_key}")
        transcript_key = transcribe.run(video_hash, audio_key, storage)
        transcript = json.loads(storage.read_text(transcript_key))
        n_segs = len(transcript.get("segments", []))
        total_words = sum(len(s["text"].split()) for s in transcript.get("segments", []))
        # Groq cost: $0.111/hr of audio = $0.0000308/sec
        audio_duration_s = transcript.get("segments", [{}])[-1].get("end", 0) if transcript.get("segments") else 0
        groq_cost = round(audio_duration_s * 0.0000308, 5)
        _step_done(jlog, "TRANSCRIBE", t, job_id=job_id, model="whisper-large-v3",
                   audio_s=audio_duration_s, cost_usd=groq_cost,
                   language=transcript.get("language"),
                   segments=n_segs,
                   total_words=total_words)
        jlog.debug(f"transcript sample: {json.dumps(transcript.get('segments', [])[:2], indent=2)}")

        # ── Embed ─────────────────────────────────────────────────
        t = _step(jlog, "EMBED (librosa audio features + MoViNet visual)")
        embed_result = embed.run(video_hash, storage, ratio=ratio)
        audio_features = json.loads(storage.read_text(f"videos/{video_hash}/audio_features.json"))
        _step_done(jlog, "EMBED", t, job_id=job_id,
                   model="MoViNet-A2-Stream + librosa (CPU)",
                   audio_scenes=len(audio_features),
                   movinet_available=storage.exists(f"videos/{video_hash}/movinet_features.npy"))
        jlog.debug(f"audio features sample: {json.dumps(audio_features[:2], indent=2)}")

        # ── Segment ───────────────────────────────────────────────
        t = _step(jlog, "SEGMENT (sentence-transformers + TextTiling)")
        segment.run(video_hash, storage)
        topic_segs = json.loads(storage.read_text(f"videos/{video_hash}/topic_segments.json"))
        _step_done(jlog, "SEGMENT", t, job_id=job_id,
                   model="all-MiniLM-L6-v2 + TextTiling (CPU)",
                   topic_segments=len(topic_segs))
        for seg in topic_segs:
            jlog.debug(
                f"  segment {seg['segment_index']}: "
                f"{seg['start_s']:.1f}s – {seg['end_s']:.1f}s | "
                f"label={seg.get('label', 'n/a')} | "
                f"words={len(seg.get('text','').split())}"
            )

        # ── Density ───────────────────────────────────────────────
        t = _step(jlog, "DENSITY SCORE (compute_density_score)")
        density_result = density_worker.run(video_hash, storage)
        _step_done(jlog, "DENSITY", t, job_id=job_id,
                   model="density scorer (CPU)",
                   score=density_result.get("score"),
                   recommended_ratio=density_result.get("recommended_ratio"),
                   ci=density_result.get("ratio_confidence_interval"))
        jlog.info(f"density signals: {json.dumps(density_result.get('signals', {}), indent=2)}")

        # ── Done ──────────────────────────────────────────────────
        phase_elapsed = round(time.time() - phase_t0, 2)
        jlog.info(f"{'#'*60}")
        jlog.info(f"PHASE 1 COMPLETE | total={phase_elapsed}s | status=awaiting_ratio")
        jlog.info(f"{'#'*60}")

        upsert_job(job_id, status=JobStatus.AWAITING_RATIO, density_result=density_result)

    except Exception as e:
        jlog.exception(f"PHASE 1 FAILED: {e}")
        log.error(f"Job {job_id} phase1 failed: {e}")
        upsert_job(job_id, status=JobStatus.FAILED, error=str(e))


# ── Phase 2 ───────────────────────────────────────────────────────────────────

def _run_phase2(job_id: str, video_hash: str, ratio: int, storage: StorageBackend):
    jlog = get_job_logger(job_id)
    phase_t0 = time.time()

    jlog.info(f"{'#'*60}")
    jlog.info(f"JOB {job_id} — PHASE 2 START | ratio={ratio}x")
    jlog.info(f"{'#'*60}")

    try:
        from apps.workers import score as score_worker, select as select_worker, render as render_worker

        job_row = get_job(job_id)
        video_type = (job_row or {}).get("video_type") or "unknown"

        # ── Score ─────────────────────────────────────────────────
        model_name = "glm-4.9b-chat" if ratio < 7 else "z-ai/glm4.7"
        # GLM cost estimate: ~$0.28/M input tokens, ~$1.10/M output tokens (NIM flash rates)
        t = _step(jlog, f"SCORE (GLM-4.7 via NVIDIA NIM | ratio={ratio}x | thinking={'yes' if ratio >= 7 else 'no'})")
        score_worker.run(video_hash, ratio, video_type, storage)
        scores = json.loads(storage.read_text(f"videos/{video_hash}/scores_{ratio}.json"))
        score_vals = [s.get("score", 0) for s in scores]
        # Estimate tokens: ~150 input tokens/segment + ~30 output tokens/segment
        est_input_tok = len(scores) * 150
        est_output_tok = len(scores) * 30
        est_cost = round(est_input_tok * 0.00000028 + est_output_tok * 0.0000011, 5)
        _step_done(jlog, "SCORE", t, job_id=job_id, model=model_name,
                   input_tokens=est_input_tok, output_tokens=est_output_tok, cost_usd=est_cost,
                   segments_scored=len(scores),
                   mean_score=round(sum(score_vals) / len(score_vals), 2) if score_vals else 0,
                   min_score=min(score_vals, default=0),
                   max_score=max(score_vals, default=0))
        jlog.debug(f"scores: {json.dumps(scores[:5], indent=2)}")

        # ── Select ────────────────────────────────────────────────
        t = _step(jlog, f"SELECT (ratio_band={'light' if ratio<=3 else 'moderate' if ratio<=6 else 'highlight'})")
        select_worker.run(video_hash, ratio, storage)
        selection = json.loads(storage.read_text(f"videos/{video_hash}/selection_{ratio}.json"))
        total_dur = sum(s["end_s"] - s["start_s"] for s in selection)
        _step_done(jlog, "SELECT", t, job_id=job_id,
                   model="rule-based (CPU)" if ratio <= 3 else "submodular greedy (CPU)" if ratio <= 6 else "topic-cluster (CPU)",
                   segments_selected=len(selection),
                   total_duration_s=round(total_dur, 1))
        for seg in selection:
            jlog.debug(f"  selected: {seg['start_s']:.1f}s – {seg['end_s']:.1f}s")

        # ── Render ────────────────────────────────────────────────
        t = _step(jlog, "RENDER (ffmpeg concat)")
        render_t0 = time.time()
        output_key = render_worker.run(video_hash, ratio, storage)
        output_path = str(storage.local_path(output_key))
        output_size_mb = round(Path(output_path).stat().st_size / 1024 / 1024, 1)
        render_elapsed = round(time.time() - render_t0, 2)
        _step_done(jlog, "RENDER", t, job_id=job_id,
                   model="ffmpeg stream-copy (CPU)",
                   output_path=output_path,
                   output_size_mb=output_size_mb)

        # ── Record export + eval metrics ──────────────────────────
        try:
            transcript_data = json.loads(storage.read_text(f"videos/{video_hash}/transcript.json"))
            segs = transcript_data.get("segments", [])
            orig_dur = segs[-1]["end"] if segs else None
        except Exception:
            orig_dur = None

        topic_segs_for_eval = json.loads(storage.read_text(f"videos/{video_hash}/topic_segments.json"))
        eval_metrics = _compute_eval_metrics(scores, selection, topic_segs_for_eval, orig_dur, total_dur, ratio)
        jlog.info(f"EVAL METRICS: {json.dumps(eval_metrics)}")

        insert_export(
            job_id=job_id, video_hash=video_hash, ratio=ratio,
            output_path=output_path, original_duration_s=orig_dur,
            output_duration_s=round(total_dur, 1),
            file_size_mb=output_size_mb, render_elapsed_s=render_elapsed,
            eval_metrics=eval_metrics,
        )

        phase_elapsed = round(time.time() - phase_t0, 2)
        jlog.info(f"{'#'*60}")
        jlog.info(f"PHASE 2 COMPLETE | total={phase_elapsed}s | output={output_path}")
        jlog.info(f"{'#'*60}")

        upsert_job(job_id, status=JobStatus.DONE, output_path=output_path)

    except Exception as e:
        jlog.exception(f"PHASE 2 FAILED: {e}")
        log.error(f"Job {job_id} phase2 failed: {e}")
        upsert_job(job_id, status=JobStatus.FAILED, error=str(e))
