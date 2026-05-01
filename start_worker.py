"""Start Celery worker — Windows/Python 3.13 safe."""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Load .env and set cwd BEFORE any app imports
_root = Path(__file__).resolve().parent
os.chdir(_root)
from dotenv import load_dotenv
load_dotenv(_root / ".env")

# Verify broker URL loaded
broker = os.environ.get("CELERY_BROKER_URL", "")
if not broker or "localhost" in broker:
    print(f"[ERROR] CELERY_BROKER_URL not set or pointing to localhost: {broker!r}")
    print(f"        Check that .env exists at {_root / '.env'} and has CELERY_BROKER_URL set")
    sys.exit(1)

print(f"[worker] broker = {broker[:40]}...")
print(f"[worker] starting — pool=threads, queues=phase1,phase2, concurrency=1")

from apps.celery_app import celery_app

celery_app.worker_main([
    "worker",
    "--loglevel=info",
    "--queues=phase1,phase2",
    "--concurrency=1",
    "--pool=threads",       # threads keep heartbeat loop alive during long tasks
    "--without-gossip",     # reduces Redis connections
    "--without-mingle",     # reduces Redis connections
    "-E",                   # emit task events — required for Flower dashboard
])
