"""
Queue Dashboard — shows all jobs, their status, signals, and actions.
Streamlit auto-discovers this as a second page.
"""
from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import requests
import streamlit as st

API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="Queue Dashboard", page_icon="📋", layout="wide")
st.title("📋 Queue Dashboard")

# ── Auto-refresh toggle ───────────────────────────────────────────────────────
col_r, col_a = st.columns([3, 1])
with col_a:
    auto_refresh = st.toggle("Auto-refresh (5s)", value=False)

# ── Fetch jobs from DB directly ───────────────────────────────────────────────
import json
from pathlib import Path

def load_jobs() -> list[dict]:
    backend = os.environ.get("DATABASE_BACKEND", "sqlite")
    if backend == "postgres":
        import psycopg2
        from psycopg2.extras import RealDictCursor
        conn = psycopg2.connect(os.environ["DATABASE_URL"])
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT 50")
        rows = cur.fetchall()
        cur.close(); conn.close()
        jobs = [dict(r) for r in rows]
    else:
        import sqlite3
        db = Path(os.environ.get("SQLITE_PATH", "./data/videosum.db"))
        if not db.exists():
            return []
        conn = sqlite3.connect(str(db)); conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT 50").fetchall()
        conn.close()
        jobs = [dict(r) for r in rows]
    for d in jobs:
        if d.get("density_result") and isinstance(d["density_result"], str):
            try: d["density_result"] = json.loads(d["density_result"])
            except Exception: pass
    return jobs

def fmt_time(iso: str) -> str:
    try:
        dt = datetime.fromisoformat(iso)
        return dt.strftime("%H:%M:%S")
    except Exception:
        return iso or "—"

def elapsed(created: str, updated: str) -> str:
    try:
        c = datetime.fromisoformat(created)
        u = datetime.fromisoformat(updated)
        secs = int((u - c).total_seconds())
        if secs < 60:
            return f"{secs}s"
        return f"{secs//60}m {secs%60}s"
    except Exception:
        return "—"

STATUS_ICON = {
    "analyzing":     "🔄",
    "awaiting_ratio":"⏳",
    "summarizing":   "⚙️",
    "done":          "✅",
    "failed":        "❌",
    "pending":       "🕐",
}

# ── Summary metrics ───────────────────────────────────────────────────────────
jobs = load_jobs()

total    = len(jobs)
done     = sum(1 for j in jobs if j["status"] == "done")
failed   = sum(1 for j in jobs if j["status"] == "failed")
running  = sum(1 for j in jobs if j["status"] in ("analyzing", "summarizing"))
waiting  = sum(1 for j in jobs if j["status"] == "awaiting_ratio")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total", total)
c2.metric("Running", running, delta="active" if running else None)
c3.metric("Waiting for ratio", waiting)
c4.metric("Done", done)
c5.metric("Failed", failed)

st.divider()

# ── Actions ───────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    if st.button("🔄 Refresh", use_container_width=True):
        st.rerun()
with col2:
    if st.button("🗑️ Clear all", type="secondary", use_container_width=True):
        backend = os.environ.get("DATABASE_BACKEND", "sqlite")
        if backend == "postgres":
            import psycopg2
            conn = psycopg2.connect(os.environ["DATABASE_URL"])
            cur = conn.cursor()
            for t in ("exports", "job_metrics", "feedback", "jobs"):
                cur.execute(f"DELETE FROM {t}")
            conn.commit(); cur.close(); conn.close()
        else:
            import sqlite3
            db = Path(os.environ.get("SQLITE_PATH", "./data/videosum.db"))
            if db.exists():
                conn = sqlite3.connect(str(db))
                conn.execute("DELETE FROM jobs")
                conn.commit(); conn.close()
        st.success("Cleared"); st.rerun()

st.divider()

# ── Job table ─────────────────────────────────────────────────────────────────
if not jobs:
    st.info("No jobs yet.")
else:
    for job in jobs:
        status = job["status"]
        icon = STATUS_ICON.get(status, "•")
        score = None
        rec = None
        if isinstance(job.get("density_result"), dict):
            score = job["density_result"].get("score")
            rec   = job["density_result"].get("recommended_ratio")

        with st.expander(
            f"{icon} `{job['job_id'][:8]}…`  —  **{status}**  "
            f"| started {fmt_time(job.get('created_at',''))}  "
            f"| elapsed {elapsed(job.get('created_at',''), job.get('updated_at',''))}",
            expanded=(status in ("failed", "analyzing", "summarizing"))
        ):
            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.markdown(f"**Status**  \n{icon} {status}")
            col_b.markdown(f"**Ratio**  \n{job.get('ratio') or '—'}")
            col_c.markdown(f"**Density**  \n{f'{score:.2f}' if score is not None else '—'}")
            col_d.markdown(f"**Rec. ratio**  \n{rec or '—'}")

            if job.get("error"):
                st.error(job["error"])

            if job.get("output_path"):
                st.success(f"Output: `{job['output_path']}`")

            # ── Report card (inline) ──────────────────────────────────────
            if True:
                try:
                    r = requests.get(f"{API_BASE}/jobs/{job['job_id']}/report", timeout=3)
                    if r.ok:
                        import pandas as pd
                        rpt = r.json()
                        metrics_data = rpt.get("metrics", [])
                        exports_data = rpt.get("exports", [])
                        total_s = round(sum(m.get("elapsed_s") or 0 for m in metrics_data))
                        total_cost = rpt.get("total_cost_usd", 0) or 0
                        st.caption(
                            f"Wall-clock: **{total_s}s**  |  "
                            f"API cost: **${total_cost:.4f}**  |  "
                            f"Stages: **{len(metrics_data)}**"
                        )
                        if metrics_data:
                            mrows = []
                            for m in metrics_data:
                                mrows.append({
                                    "Stage": m["stage"].split("(")[0].strip(),
                                    "Model": m.get("model") or "—",
                                    "Time (s)": float(m["elapsed_s"]) if m.get("elapsed_s") is not None else 0.0,
                                    "Cost ($)": float(m["cost_usd"]) if m.get("cost_usd") else 0.0,
                                })
                            mrows.append({
                                "Stage": "── TOTAL ──",
                                "Model": "",
                                "Time (s)": float(total_s),
                                "Cost ($)": float(total_cost),
                            })
                            df = pd.DataFrame(mrows)
                            st.dataframe(
                                df.style.format({"Time (s)": "{:.1f}", "Cost ($)": "${:.4f}"}),
                                use_container_width=True, hide_index=True
                            )
                        if exports_data:
                            erows = [{"Ratio": f"{e['ratio']}×",
                                      "Output dur": f"{round(e['output_duration_s'])}s" if e.get("output_duration_s") else "—",
                                      "File": f"{e['file_size_mb']}MB" if e.get("file_size_mb") else "—"} for e in exports_data]
                            st.dataframe(pd.DataFrame(erows), use_container_width=True, hide_index=True)
                except Exception as _e:
                    st.warning(f"Report unavailable: {_e}")

            # ── Per-job actions ───────────────────────────────────────────
            btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 4])

            if status == "failed":
                with btn_col1:
                    if st.button("↩️ Retry", key=f"retry_{job['job_id']}"):
                        try:
                            r = requests.post(f"{API_BASE}/jobs/{job['job_id']}/retry", timeout=5)
                            r.raise_for_status()
                            new_status = r.json().get("status")
                            st.success(f"Resumed → {new_status}")
                            time.sleep(1)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Retry failed: {e}")

            if status == "awaiting_ratio":
                with btn_col1:
                    url = f"http://localhost:8501/?job_id={job['job_id']}&phase=awaiting_ratio"
                    st.link_button("▶️ Continue in UI", url)

            if status == "done":
                with btn_col1:
                    url = f"http://localhost:8501/?job_id={job['job_id']}&phase=done"
                    st.link_button("⬇️ Download", url)

            with btn_col2:
                if st.button("🗑️ Delete", key=f"del_{job['job_id']}"):
                    backend = os.environ.get("DATABASE_BACKEND", "sqlite")
                    if backend == "postgres":
                        import psycopg2
                        conn = psycopg2.connect(os.environ["DATABASE_URL"])
                        cur = conn.cursor()
                        cur.execute("DELETE FROM jobs WHERE job_id=%s", (job["job_id"],))
                        conn.commit(); cur.close(); conn.close()
                    else:
                        import sqlite3
                        db = Path(os.environ.get("SQLITE_PATH", "./data/videosum.db"))
                        conn = sqlite3.connect(str(db))
                        conn.execute("DELETE FROM jobs WHERE job_id=?", (job["job_id"],))
                        conn.commit(); conn.close()
                    st.rerun()

# ── Auto-refresh ──────────────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(5)
    st.rerun()
