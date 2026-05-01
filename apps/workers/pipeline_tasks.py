from __future__ import annotations

from apps.celery_app import celery_app
from apps.logger import setup_root_logging

# Worker process has no main.py — set up logging once at import time
setup_root_logging()


@celery_app.task(name="videosum.phase1", bind=True, max_retries=0)
def phase1_task(self, job_id: str, video_hash: str, video_path: str, ratio: int | None = None) -> None:
    from apps.api.dependencies import get_storage
    from apps.api.routers.jobs import _run_phase1
    _run_phase1(job_id, video_hash, video_path, get_storage(), ratio)


@celery_app.task(name="videosum.phase2", bind=True, max_retries=0)
def phase2_task(self, job_id: str, video_hash: str, ratio: int) -> None:
    from apps.api.dependencies import get_storage
    from apps.api.routers.jobs import _run_phase2
    _run_phase2(job_id, video_hash, ratio, get_storage())
