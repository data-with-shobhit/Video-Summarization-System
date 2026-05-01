from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from celery import Celery

celery_app = Celery("videosum")

celery_app.conf.update(
    broker_url=os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0"),
    result_backend=os.environ.get("CELERY_RESULT_BACKEND", os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")),
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    broker_transport_options={
        "visibility_timeout": 3600,
        "max_connections": 5,       # limit broker connections — free tier has low limit
        "fanout_prefix": True,      # required for Flower to receive events on Redis broker
        "fanout_patterns": True,    # required for Flower to receive events on Redis broker
    },
    broker_pool_limit=3,            # max connections in Celery's broker pool
    redis_max_connections=5,        # max Redis connections total
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    task_track_started=True,
    task_send_sent_event=True,
    task_routes={
        "videosum.phase1": {"queue": "phase1"},
        "videosum.phase2": {"queue": "phase2"},
    },
)

# Explicitly import tasks so they register — autodiscover unreliable on Windows
import apps.workers.pipeline_tasks  # noqa: F401, E402
