"""
Streamlit UI for VideoSum.
Run: streamlit run app_ui.py
Talks directly to the API at http://localhost:8000
"""
from __future__ import annotations

import os
import time
import tempfile
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")

import requests
import streamlit as st

API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="VideoSum",
    page_icon="🎬",
    layout="centered",
)

# ── Sidebar — Groq free tier usage ───────────────────────────────────────────
with st.sidebar:
    st.markdown("### Groq free tier usage")
    try:
        health = requests.get(f"{API_BASE}/health", timeout=2).json()
        usage = health.get("groq_usage", {})
        if usage:
            day_pct  = usage.get("day_pct", 0)
            hour_pct = usage.get("hour_pct", 0)

            st.caption("Today")
            st.progress(min(day_pct / 100, 1.0),
                text=f"{usage['day_used_s']}s / {usage['day_limit_s']}s ({day_pct}%)")

            st.caption("This hour")
            st.progress(min(hour_pct / 100, 1.0),
                text=f"{usage['hour_used_s']}s / {usage['hour_limit_s']}s ({hour_pct}%)")

            remaining_videos = int(usage["day_remaining_s"] / 1800)
            st.caption(f"~{remaining_videos} × 30-min videos remaining today")

            if day_pct >= 80:
                st.warning("Approaching daily limit")
        else:
            st.caption("No usage data yet")
    except Exception:
        st.caption("API offline")

    # ── Recent jobs ───────────────────────────────────────────────────────
    st.divider()
    st.markdown("### Recent jobs")
    try:
        from pathlib import Path
        _STATUS_ICON = {"analyzing":"🔄","awaiting_ratio":"⏳","summarizing":"⚙️","done":"✅","failed":"❌","pending":"🕐"}
        _backend = os.environ.get("DATABASE_BACKEND", "sqlite")
        if _backend == "postgres":
            import psycopg2
            from psycopg2.extras import RealDictCursor
            _conn = psycopg2.connect(os.environ["DATABASE_URL"])
            _cur = _conn.cursor(cursor_factory=RealDictCursor)
            _cur.execute("SELECT job_id, status, name FROM jobs ORDER BY created_at DESC LIMIT 10")
            _rows = [(r["job_id"], r["status"], r["name"]) for r in _cur.fetchall()]
            _cur.close(); _conn.close()
        else:
            import sqlite3
            _db = Path(os.environ.get("SQLITE_PATH", "./data/videosum.db"))
            if not _db.exists():
                _rows = []
            else:
                _conn = sqlite3.connect(str(_db))
                _rows = _conn.execute(
                    "SELECT job_id, status, name FROM jobs ORDER BY created_at DESC LIMIT 10"
                ).fetchall()
                _conn.close()
        if _rows:
            for _jid, _st, _name in _rows:
                _icon = _STATUS_ICON.get(_st, "•")
                _phase = "awaiting_ratio" if _st == "awaiting_ratio" else ("done" if _st == "done" else "analyzing")
                _url = f"http://localhost:8501/?job_id={_jid}&phase={_phase}"
                _label = _name or _jid[:8]
                st.markdown(f"{_icon} [{_label}]({_url}) — {_st}")
        else:
            st.caption("No jobs yet")
    except Exception as _ex:
        st.caption(f"Could not load jobs: {_ex}")

# ── Session state defaults — restore from URL on refresh ─────────────────────
_qp = st.query_params
for k, v in {
    "job_id": _qp.get("job_id"),
    "phase": _qp.get("phase", "upload"),
    "density": None,
    "output_path": None,
    "error": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# If we recovered a job from URL, fetch its latest state from the API
if st.session_state.job_id and st.session_state.phase not in ("upload", "done", "failed"):
    try:
        _job = requests.get(f"{API_BASE}/jobs/{st.session_state.job_id}", timeout=3).json()
        _status = _job.get("status")
        if _status == "awaiting_ratio":
            st.session_state.phase = "awaiting_ratio"
            st.session_state.density = _job.get("density_result")
        elif _status == "done":
            st.session_state.phase = "done"
            st.session_state.output_path = _job.get("output_path")
        elif _status == "failed":
            st.session_state.phase = "failed"
            st.session_state.error = _job.get("error")
        # analyzing / summarizing → keep polling phase as-is
    except Exception:
        pass


# ── Helpers ───────────────────────────────────────────────────────────────────

def api_post(path: str, **kwargs) -> dict:
    r = requests.post(f"{API_BASE}{path}", **kwargs)
    r.raise_for_status()
    return r.json()


def api_get(path: str) -> dict:
    r = requests.get(f"{API_BASE}{path}")
    r.raise_for_status()
    return r.json()


def poll_job(job_id: str) -> dict:
    return api_get(f"/jobs/{job_id}")


def reset():
    for k in ("job_id", "density", "output_path", "error"):
        st.session_state[k] = None
    st.session_state["phase"] = "upload"
    st.query_params.clear()


# ── UI ────────────────────────────────────────────────────────────────────────

st.title("VideoSum")
st.caption("Upload a video. Get a shorter video.")

# Always show reset button + job ID except on upload screen
if st.session_state.phase != "upload":
    _col1, _col2 = st.columns([1, 3])
    with _col1:
        if st.button("Start over", type="secondary"):
            reset()
            st.rerun()
    with _col2:
        if st.session_state.job_id:
            st.caption(f"Job ID: `{st.session_state.job_id}`")

# ─ Phase: upload ─────────────────────────────────────────────────────────────
if st.session_state.phase == "upload":
    st.subheader("Upload video")

    uploaded = st.file_uploader(
        "Choose a video file",
        type=["mp4", "mov", "avi", "mkv", "webm"],
        help="Max 10GB. Any duration."
    )

    if uploaded:
        # Save to temp file so API can read it
        suffix = Path(uploaded.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        st.video(tmp_path)

        job_name = st.text_input("Job name (optional)", placeholder="e.g. Product Demo v2, Team Meeting May")

        _VIDEO_TYPE_OPTIONS = {
            "unknown":    "Auto (let the editor decide)",
            "lecture":    "Lecture / presentation",
            "interview":  "Interview",
            "demo":       "Product demo / walkthrough",
            "podcast":    "Podcast / panel discussion",
            "tutorial":   "Tutorial / how-to",
            "short_film": "Short film",
            "movie":      "Movie / feature film",
            "episode":    "TV episode / series episode",
        }
        video_type_label = st.selectbox(
            "Content type",
            options=list(_VIDEO_TYPE_OPTIONS.keys()),
            format_func=lambda k: _VIDEO_TYPE_OPTIONS[k],
            index=0,
            help="Tells the AI editor how to prioritise segments.",
        )

        if st.button("Analyze video", type="primary", use_container_width=True):
            with st.spinner("Submitting job..."):
                try:
                    payload = {"video_path": tmp_path, "ratio": "auto", "video_type": video_type_label}
                    if job_name.strip():
                        payload["name"] = job_name.strip()
                    resp = api_post("/jobs", json=payload)
                    st.session_state.job_id = resp["job_id"]
                    st.session_state.job_name = resp.get("name", "")
                    st.session_state.phase = "analyzing"
                    st.query_params["job_id"] = resp["job_id"]
                    st.query_params["phase"] = "analyzing"
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to submit: {e}")

# ─ Phase: analyzing ──────────────────────────────────────────────────────────
elif st.session_state.phase == "analyzing":
    st.subheader("Analyzing video...")

    progress_steps = [
        ("Normalizing + extracting audio", 0.15),
        ("Transcribing with Groq Whisper", 0.30),
        ("Computing visual embeddings (MoViNet)", 0.50),
        ("Detecting topic segments (TextTiling)", 0.70),
        ("Computing density score", 0.90),
        ("Done!", 1.0),
    ]

    prog_bar = st.progress(0.0)
    status_text = st.empty()
    step_idx = 0

    # Poll until done
    while True:
        try:
            job = poll_job(st.session_state.job_id)
        except Exception as e:
            st.error(f"Polling error: {e}")
            break

        status = job["status"]

        if status == "awaiting_ratio":
            prog_bar.progress(1.0)
            status_text.success("Analysis complete!")
            st.session_state.density = job.get("density_result")
            st.session_state.phase = "awaiting_ratio"
            time.sleep(0.5)
            st.rerun()
            break

        if status == "failed":
            st.session_state.phase = "failed"
            st.session_state.error = job.get("error", "Unknown error")
            st.rerun()
            break

        # Animate progress while analyzing
        if step_idx < len(progress_steps) - 1:
            step_idx += 1
        label, frac = progress_steps[step_idx]
        prog_bar.progress(frac)
        status_text.info(label)
        time.sleep(3)

# ─ Phase: awaiting_ratio ─────────────────────────────────────────────────────
elif st.session_state.phase == "awaiting_ratio":
    density = st.session_state.density or {}
    score = density.get("score", 0.0)
    rec_ratio = density.get("recommended_ratio")
    ci = density.get("ratio_confidence_interval", 1)
    message = density.get("message")
    signals = density.get("signals", {})

    st.subheader("Analysis results")

    # ── Top metrics ──────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    col1.metric("Density score", f"{score:.2f}", help="0 = very compressible · 1 = very dense")
    if rec_ratio:
        col2.metric("Recommended ratio", f"{rec_ratio}×", delta=f"±{ci}")
        col3.metric("Output length", f"~{round(100/rec_ratio)}% of original")
    else:
        col2.warning("Too dense to compress")
        col3.empty()

    if message:
        st.info(message)

    # ── Density score bar ─────────────────────────────────────────────────────
    st.divider()
    st.markdown("**Overall density score**")
    st.progress(float(score), text=f"{score:.2f}  {'(dense — less compressible)' if score > 0.6 else '(compressible)' if score < 0.4 else '(moderate)'}")

    # ── All 7 signals ─────────────────────────────────────────────────────────
    if signals:
        st.divider()
        st.markdown("**Signal breakdown** — what the density scorer measured")

        SIGNAL_META = {
            "semantic_redundancy": {
                "label": "Semantic redundancy",
                "desc": "How much content repeats itself. High = more compressible.",
                "high_bad": False,
            },
            "silence_ratio": {
                "label": "Silence ratio",
                "desc": "Fraction of audio that is silence. High = lots of dead air.",
                "high_bad": False,
            },
            "filler_word_rate": {
                "label": "Filler word rate",
                "desc": "um / uh / like / you know per 100 words. High = loose speech.",
                "high_bad": False,
            },
            "pacing_variance": {
                "label": "Pacing variance",
                "desc": "Std dev of words-per-minute across scenes. High = uneven pacing.",
                "high_bad": False,
            },
            "lexical_density": {
                "label": "Lexical density",
                "desc": "Unique words / total words (type-token ratio). High = rich vocabulary.",
                "high_bad": True,
            },
            "topic_count": {
                "label": "Topic segments detected",
                "desc": "Number of distinct topic segments found by TextTiling.",
                "high_bad": True,
            },
            "visual_change_rate": {
                "label": "Visual change rate",
                "desc": "Mean frame-to-frame MoViNet distance. High = visually dynamic.",
                "high_bad": True,
            },
        }

        col_a, col_b = st.columns(2)
        for i, (key, meta) in enumerate(SIGNAL_META.items()):
            val = signals.get(key)
            if val is None:
                continue
            col = col_a if i % 2 == 0 else col_b
            with col:
                # Normalize topic_count to 0-1 for display only
                if key == "topic_count":
                    display_val = int(val)
                    bar_val = min(float(val) / 15.0, 1.0)
                    val_str = str(display_val)
                else:
                    bar_val = float(val)
                    val_str = f"{float(val):.3f}"

                st.markdown(f"**{meta['label']}** — `{val_str}`")
                st.caption(meta["desc"])
                st.progress(bar_val)

    st.divider()
    st.subheader("Choose compression ratio")

    if message:  # too dense
        st.warning("Content is too dense for meaningful video compression. You can still try a low ratio.")

    ratio_val = st.slider(
        "Compression ratio",
        min_value=2, max_value=10,
        value=rec_ratio or 4,
        step=1,
        format="%d×",
        help="2× = keep 50% · 5× = keep 20% · 10× = keep 10%"
    )

    # Band label
    if ratio_val <= 3:
        band_label = "Light trim — removes silences and filler"
        band_color = "green"
    elif ratio_val <= 6:
        band_label = "Moderate — submodular greedy scene selection"
        band_color = "orange"
    else:
        band_label = "Highlight reel — best scene per topic cluster"
        band_color = "red"

    st.caption(f":{band_color}[{band_label}]")
    st.caption(f"Output will be ~{round(100/ratio_val)}% of original duration")

    if st.button(f"Summarize at {ratio_val}×", type="primary", use_container_width=True):
        with st.spinner("Confirming ratio..."):
            try:
                api_post(f"/jobs/{st.session_state.job_id}/confirm-ratio", json={"ratio": ratio_val})
                st.session_state.phase = "summarizing"
                st.query_params["phase"] = "summarizing"
                st.rerun()
            except Exception as e:
                st.error(f"Failed: {e}")

# ─ Phase: summarizing ────────────────────────────────────────────────────────
elif st.session_state.phase == "summarizing":
    st.subheader("Creating summary video...")

    steps = [
        ("Scoring segments with DeepSeek V4", 0.30),
        ("Selecting best segments", 0.60),
        ("Rendering output video (ffmpeg)", 0.85),
        ("Done!", 1.0),
    ]

    prog_bar = st.progress(0.0)
    status_text = st.empty()
    step_idx = 0

    while True:
        try:
            job = poll_job(st.session_state.job_id)
        except Exception as e:
            st.error(f"Polling error: {e}")
            break

        status = job["status"]

        if status == "done":
            prog_bar.progress(1.0)
            status_text.success("Summary ready!")
            st.session_state.output_path = job.get("output_path")
            st.session_state.phase = "done"
            time.sleep(0.5)
            st.rerun()
            break

        if status == "failed":
            st.session_state.phase = "failed"
            st.session_state.error = job.get("error", "Unknown error")
            st.rerun()
            break

        if step_idx < len(steps) - 1:
            step_idx += 1
        label, frac = steps[step_idx]
        prog_bar.progress(frac)
        status_text.info(label)
        time.sleep(3)

# ─ Phase: done ───────────────────────────────────────────────────────────────
elif st.session_state.phase == "done":
    st.subheader("Summary ready!")
    st.success("Your video has been summarized.")

    output_path = st.session_state.output_path
    if output_path and Path(output_path).exists():
        st.video(output_path)

        with open(output_path, "rb") as f:
            st.download_button(
                label="Download summary video",
                data=f,
                file_name=Path(output_path).name,
                mime="video/mp4",
                use_container_width=True,
                type="primary",
            )

    st.divider()
    st.subheader("Try a different ratio?")
    col_r, col_b = st.columns([2, 1])
    with col_r:
        new_ratio = st.slider("Compression ratio", min_value=2, max_value=10, value=4, step=1,
                              format="%d×", help="Phase 1 is cached — only scoring + render will re-run")
    with col_b:
        st.caption(f"Output ~{round(100/new_ratio)}% of original")
        if st.button(f"Re-summarize at {new_ratio}×", use_container_width=True):
            with st.spinner("Starting..."):
                try:
                    api_post(f"/jobs/{st.session_state.job_id}/re-summarize", json={"ratio": new_ratio})
                    st.session_state.phase = "summarizing"
                    st.query_params["phase"] = "summarizing"
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed: {e}")

    # ── Report card ───────────────────────────────────────────────────────────
    st.divider()
    with st.expander("Job report card", expanded=False):
        try:
            report = api_get(f"/jobs/{st.session_state.job_id}/report")
            metrics = report.get("metrics", [])
            exports = report.get("exports", [])
            total_cost = report.get("total_cost_usd", 0)

            # Summary row
            rc1, rc2, rc3 = st.columns(3)
            total_elapsed = sum(m.get("elapsed_s") or 0 for m in metrics)
            rc1.metric("Total wall-clock", f"{round(total_elapsed)}s")
            rc2.metric("Total API cost", f"${total_cost:.4f}")
            rc3.metric("Exports made", len(exports))

            # Stage timings table
            if metrics:
                st.markdown("**Stage timings**")
                import pandas as pd
                rows = []
                for m in metrics:
                    cost_str = f"${m['cost_usd']:.4f}" if m.get("cost_usd") else "—"
                    model_str = m.get("model") or "—"
                    tok_str = ""
                    if m.get("input_tokens"):
                        tok_str = f"{m['input_tokens']}in / {m.get('output_tokens', 0)}out"
                    elif m.get("audio_s"):
                        tok_str = f"{round(m['audio_s'])}s audio"
                    rows.append({
                        "Stage": m["stage"].split("(")[0].strip(),
                        "Model": model_str,
                        "Time (s)": m.get("elapsed_s") or "—",
                        "Tokens / Audio": tok_str or "—",
                        "Cost": cost_str,
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            # Exports / compression history + eval metrics
            if exports:
                st.markdown("**Compression history & evaluation**")
                for e in exports:
                    orig = e.get("original_duration_s")
                    out = e.get("output_duration_s")
                    actual_ratio = round(orig / out, 1) if orig and out else None
                    ev = e.get("eval_metrics") or {}

                    st.markdown(f"**Ratio {e['ratio']}×** — {round(orig)}s → {round(out)}s · {e.get('file_size_mb', '?')}MB · render {e.get('render_elapsed_s', '?')}s")

                    m1, m2, m3, m4 = st.columns(4)
                    # Ratio accuracy
                    err = ev.get("ratio_error_pct")
                    m1.metric("Actual ratio", f"{actual_ratio}×" if actual_ratio else "—",
                              delta=f"{err}% off target" if err is not None else None,
                              delta_color="inverse")
                    # Topic coverage
                    cov = ev.get("topic_coverage_pct")
                    m2.metric("Topic coverage",
                              f"{cov}%" if cov is not None else "—",
                              delta=f"{ev.get('topics_covered','?')}/{ev.get('topics_total','?')} segments")
                    # Score gap
                    gap = ev.get("score_gap")
                    m3.metric("Score gap",
                              f"{gap}" if gap is not None else "—",
                              delta=f"sel {ev.get('mean_selected_score','?')} vs rej {ev.get('mean_rejected_score','?')}")
                    # Score distribution
                    m4.metric("Score dist",
                              f"μ={ev.get('score_distribution_mean','?')}",
                              delta=f"σ={ev.get('score_distribution_std','?')}")

                    # Per-segment scores with reasons
                    seg_scores = report.get("scores_by_ratio", {}).get(e["ratio"], [])
                    if seg_scores:
                        with st.expander(f"Segment scores — ratio {e['ratio']}×", expanded=False):
                            srows = []
                            for s in seg_scores:
                                srows.append({
                                    "Seg": s.get("id", "?"),
                                    "Score": s.get("score") or s.get("composite_score") or "—",
                                    "LLM": s.get("llm_score", "—"),
                                    "Audio": round(float(s["audio_emphasis"]), 2) if s.get("audio_emphasis") is not None else "—",
                                    "Visual": round(float(s["visual_salience"]), 2) if s.get("visual_salience") is not None else "—",
                                    "Reason": s.get("reason") or "—",
                                })
                            st.dataframe(pd.DataFrame(srows), use_container_width=True, hide_index=True)

        except Exception as e:
            st.caption(f"Report unavailable: {e}")

    st.divider()
    st.subheader("How was it?")
    rating = st.slider("Rate the summary", 1, 5, 3, format="%d ⭐")
    comment = st.text_input("Comment (optional)")

    if st.button("Submit feedback"):
        try:
            requests.post(
                f"{API_BASE}/jobs/{st.session_state.job_id}/feedback",
                json={"rating": rating, "comment": comment or None}
            )
            st.success("Thanks for the feedback!")
        except Exception as e:
            st.warning(f"Could not submit feedback: {e}")

# ─ Phase: failed ─────────────────────────────────────────────────────────────
elif st.session_state.phase == "failed":
    st.error("Something went wrong.")
    if st.session_state.error:
        st.code(st.session_state.error)
    st.info("Check the API logs for details. Common causes: missing API keys, ffmpeg not installed, audio too large for Groq.")

    if st.session_state.job_id:
        if st.button("Retry from last checkpoint", type="primary", use_container_width=True):
            with st.spinner("Resuming..."):
                try:
                    resp = api_post(f"/jobs/{st.session_state.job_id}/retry")
                    new_status = resp.get("status")
                    st.session_state.error = None
                    if new_status == "awaiting_ratio":
                        st.session_state.density = resp.get("density_result")
                        st.session_state.phase = "awaiting_ratio"
                        st.query_params["phase"] = "awaiting_ratio"
                    elif new_status == "summarizing":
                        st.session_state.phase = "summarizing"
                        st.query_params["phase"] = "summarizing"
                    else:
                        st.session_state.phase = "analyzing"
                        st.query_params["phase"] = "analyzing"
                    st.rerun()
                except Exception as e:
                    st.error(f"Retry failed: {e}")
