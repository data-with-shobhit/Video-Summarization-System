from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from typing import Any


# ── Backend helpers ───────────────────────────────────────────────────────────

def _backend() -> str:
    return os.environ.get("DATABASE_BACKEND", "sqlite")


def _get_conn():
    if _backend() == "postgres":
        import psycopg2
        return psycopg2.connect(os.environ["DATABASE_URL"])
    path = os.environ.get("SQLITE_PATH", "./data/videosum.db")
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _ph() -> str:
    """SQL placeholder — ? for SQLite, %s for Postgres."""
    return "%s" if _backend() == "postgres" else "?"


def _rows_to_dicts(cursor) -> list[dict]:
    rows = cursor.fetchall()
    if _backend() == "postgres":
        cols = [d[0] for d in cursor.description]
        return [dict(zip(cols, r)) for r in rows]
    return [dict(r) for r in rows]


def _row_to_dict(cursor) -> dict | None:
    row = cursor.fetchone()
    if row is None:
        return None
    if _backend() == "postgres":
        cols = [d[0] for d in cursor.description]
        return dict(zip(cols, row))
    return dict(row)


# ── Schema ────────────────────────────────────────────────────────────────────

_SQLITE_DDL = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    video_hash TEXT,
    name TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    ratio INTEGER,
    video_type TEXT DEFAULT 'unknown',
    video_path TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    density_result TEXT,
    output_path TEXT,
    error TEXT
);
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    rating INTEGER NOT NULL,
    comment TEXT,
    actual_ratio_used INTEGER,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS job_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    elapsed_s REAL,
    model TEXT,
    audio_s REAL,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cost_usd REAL,
    extra TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS exports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    video_hash TEXT,
    ratio INTEGER,
    output_path TEXT,
    original_duration_s REAL,
    output_duration_s REAL,
    file_size_mb REAL,
    render_elapsed_s REAL,
    eval_metrics TEXT,
    created_at TEXT NOT NULL
);
"""

_PG_DDL = [
    """
    CREATE TABLE IF NOT EXISTS jobs (
        job_id TEXT PRIMARY KEY,
        video_hash TEXT,
        name TEXT,
        status TEXT NOT NULL DEFAULT 'pending',
        ratio INTEGER,
        video_type TEXT DEFAULT 'unknown',
        video_path TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        density_result TEXT,
        output_path TEXT,
        error TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS feedback (
        id SERIAL PRIMARY KEY,
        job_id TEXT NOT NULL,
        rating INTEGER NOT NULL,
        comment TEXT,
        actual_ratio_used INTEGER,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS job_metrics (
        id SERIAL PRIMARY KEY,
        job_id TEXT NOT NULL,
        stage TEXT NOT NULL,
        elapsed_s REAL,
        model TEXT,
        audio_s REAL,
        input_tokens INTEGER,
        output_tokens INTEGER,
        cost_usd REAL,
        extra TEXT,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS exports (
        id SERIAL PRIMARY KEY,
        job_id TEXT NOT NULL,
        video_hash TEXT,
        ratio INTEGER,
        output_path TEXT,
        original_duration_s REAL,
        output_duration_s REAL,
        file_size_mb REAL,
        render_elapsed_s REAL,
        eval_metrics TEXT,
        created_at TEXT NOT NULL
    )
    """,
]


def init_db() -> None:
    conn = _get_conn()
    if _backend() == "postgres":
        cur = conn.cursor()
        for stmt in _PG_DDL:
            cur.execute(stmt)
        # Add name column if missing (migration for older schemas)
        cur.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'jobs' AND column_name = 'name'
        """)
        if not cur.fetchone():
            cur.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS name TEXT")
        cur.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'jobs' AND column_name = 'video_type'
        """)
        if not cur.fetchone():
            cur.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS video_type TEXT DEFAULT 'unknown'")
        cur.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'jobs' AND column_name = 'video_path'
        """)
        if not cur.fetchone():
            cur.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS video_path TEXT")
        # Only mark stuck jobs as failed when NOT using Celery
        # (Celery workers are independent of the API process — jobs survive restarts)
        if not os.environ.get("CELERY_BROKER_URL"):
            cur.execute(
                "UPDATE jobs SET status = 'failed', error = 'Server restarted while job was in progress' "
                "WHERE status IN ('analyzing', 'summarizing')"
            )
        conn.commit()
        cur.close()
        conn.close()
    else:
        conn.executescript(_SQLITE_DDL)
        cols = [r[1] for r in conn.execute("PRAGMA table_info(jobs)").fetchall()]
        if "name" not in cols:
            conn.execute("ALTER TABLE jobs ADD COLUMN name TEXT")
        if "video_type" not in cols:
            conn.execute("ALTER TABLE jobs ADD COLUMN video_type TEXT DEFAULT 'unknown'")
        if "video_path" not in cols:
            conn.execute("ALTER TABLE jobs ADD COLUMN video_path TEXT")
        exp_cols = [r[1] for r in conn.execute("PRAGMA table_info(exports)").fetchall()]
        if "eval_metrics" not in exp_cols:
            conn.execute("ALTER TABLE exports ADD COLUMN eval_metrics TEXT")
        if not os.environ.get("CELERY_BROKER_URL"):
            conn.execute(
                "UPDATE jobs SET status = 'failed', error = 'Server restarted while job was in progress' "
                "WHERE status IN ('analyzing', 'summarizing')"
            )
        conn.commit()
        conn.close()


# ── Jobs ──────────────────────────────────────────────────────────────────────

def upsert_job(job_id: str, **fields: Any) -> None:
    conn = _get_conn()
    now = datetime.utcnow().isoformat()
    fields["updated_at"] = now

    for k in ("density_result",):
        if k in fields and not isinstance(fields[k], str):
            fields[k] = json.dumps(fields[k])

    ph = _ph()

    if _backend() == "postgres":
        fields["job_id"] = job_id
        fields.setdefault("created_at", now)
        cols = list(fields.keys())
        vals = list(fields.values())
        col_str = ", ".join(cols)
        val_ph = ", ".join(ph for _ in vals)
        # All non-primary-key columns for ON CONFLICT update
        update_cols = [c for c in cols if c != "job_id"]
        update_str = ", ".join(f"{c} = EXCLUDED.{c}" for c in update_cols)
        cur = conn.cursor()
        cur.execute(
            f"INSERT INTO jobs ({col_str}) VALUES ({val_ph}) ON CONFLICT (job_id) DO UPDATE SET {update_str}",
            vals,
        )
        conn.commit()
        cur.close()
    else:
        placeholders = ", ".join(f"{k} = {ph}" for k in fields)
        values = list(fields.values()) + [job_id]
        existing = conn.execute(f"SELECT job_id FROM jobs WHERE job_id = {ph}", (job_id,)).fetchone()
        if existing:
            conn.execute(f"UPDATE jobs SET {placeholders} WHERE job_id = {ph}", values)
        else:
            fields["job_id"] = job_id
            fields.setdefault("created_at", now)
            cols = ", ".join(fields.keys())
            qs = ", ".join(ph for _ in fields)
            conn.execute(f"INSERT INTO jobs ({cols}) VALUES ({qs})", list(fields.values()))
        conn.commit()

    conn.close()


def get_job(job_id: str) -> dict | None:
    conn = _get_conn()
    ph = _ph()
    cur = conn.cursor()
    cur.execute(f"SELECT * FROM jobs WHERE job_id = {ph}", (job_id,))
    d = _row_to_dict(cur)
    cur.close()
    conn.close()
    if not d:
        return None
    if d.get("density_result"):
        try:
            d["density_result"] = json.loads(d["density_result"])
        except Exception:
            pass
    return d


# ── Feedback ──────────────────────────────────────────────────────────────────

def insert_feedback(job_id: str, rating: int, comment: str | None, actual_ratio: int | None) -> None:
    conn = _get_conn()
    ph = _ph()
    cur = conn.cursor()
    cur.execute(
        f"INSERT INTO feedback (job_id, rating, comment, actual_ratio_used, created_at) VALUES ({ph},{ph},{ph},{ph},{ph})",
        (job_id, rating, comment, actual_ratio, datetime.utcnow().isoformat()),
    )
    conn.commit()
    cur.close()
    conn.close()


# ── Metrics & exports ─────────────────────────────────────────────────────────

def insert_metric(
    job_id: str,
    stage: str,
    elapsed_s: float | None = None,
    model: str | None = None,
    audio_s: float | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    cost_usd: float | None = None,
    extra: dict | None = None,
) -> None:
    conn = _get_conn()
    ph = _ph()
    cur = conn.cursor()
    cur.execute(
        f"INSERT INTO job_metrics (job_id, stage, elapsed_s, model, audio_s, input_tokens, output_tokens, cost_usd, extra, created_at) "
        f"VALUES ({ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph})",
        (job_id, stage, elapsed_s, model, audio_s, input_tokens, output_tokens, cost_usd,
         json.dumps(extra) if extra else None, datetime.utcnow().isoformat()),
    )
    conn.commit()
    cur.close()
    conn.close()


def insert_export(
    job_id: str,
    video_hash: str,
    ratio: int,
    output_path: str,
    original_duration_s: float | None,
    output_duration_s: float | None,
    file_size_mb: float | None,
    render_elapsed_s: float | None,
    eval_metrics: dict | None = None,
) -> None:
    conn = _get_conn()
    ph = _ph()
    cur = conn.cursor()
    cur.execute(
        f"INSERT INTO exports (job_id, video_hash, ratio, output_path, original_duration_s, output_duration_s, file_size_mb, render_elapsed_s, eval_metrics, created_at) "
        f"VALUES ({ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph})",
        (job_id, video_hash, ratio, output_path, original_duration_s, output_duration_s,
         file_size_mb, render_elapsed_s,
         json.dumps(eval_metrics) if eval_metrics else None,
         datetime.utcnow().isoformat()),
    )
    conn.commit()
    cur.close()
    conn.close()


def get_job_report(job_id: str) -> dict:
    conn = _get_conn()
    ph = _ph()
    cur = conn.cursor()
    cur.execute(
        f"SELECT stage, elapsed_s, model, audio_s, input_tokens, output_tokens, cost_usd, extra, created_at "
        f"FROM job_metrics WHERE job_id = {ph} ORDER BY created_at",
        (job_id,),
    )
    metrics = _rows_to_dicts(cur)
    cur.execute(
        f"SELECT ratio, output_path, original_duration_s, output_duration_s, file_size_mb, render_elapsed_s, eval_metrics, created_at "
        f"FROM exports WHERE job_id = {ph} ORDER BY created_at",
        (job_id,),
    )
    exports = _rows_to_dicts(cur)
    cur.close()
    conn.close()
    for m in metrics:
        if m.get("extra"):
            try:
                m["extra"] = json.loads(m["extra"])
            except Exception:
                pass
    for e in exports:
        if e.get("eval_metrics"):
            try:
                e["eval_metrics"] = json.loads(e["eval_metrics"])
            except Exception:
                pass
    total_cost = sum(m["cost_usd"] or 0 for m in metrics)
    return {"metrics": metrics, "exports": exports, "total_cost_usd": round(total_cost, 5)}
