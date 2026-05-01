from __future__ import annotations

from fastapi import APIRouter, HTTPException

from packages.core.models import FeedbackRequest
from packages.storage.db import get_job, insert_feedback

router = APIRouter(prefix="/jobs", tags=["feedback"])


@router.post("/{job_id}/feedback", status_code=204)
async def submit_feedback(job_id: str, body: FeedbackRequest):
    row = get_job(job_id)
    if not row:
        raise HTTPException(status_code=404, detail="Job not found")

    insert_feedback(
        job_id=job_id,
        rating=body.rating,
        comment=body.comment,
        actual_ratio=body.actual_ratio_used,
    )
