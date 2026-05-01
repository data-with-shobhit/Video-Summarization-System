from __future__ import annotations

import os
from pathlib import Path

# Always run relative to project root so ./data, ./models etc. resolve correctly
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(_PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(_PROJECT_ROOT / ".env")  # explicit path — works regardless of cwd
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("PYTHONWARNINGS", "ignore::DeprecationWarning")
os.environ.setdefault("HF_HUB_OFFLINE", "1")        # load from cache, skip network checks
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from apps.logger import setup_root_logging, get_logger
from apps.api.routers import jobs, feedback
from packages.storage.db import init_db

setup_root_logging()
log = get_logger(__name__)

app = FastAPI(
    title="VideoSum API",
    description="Video summarization — density-aware, video-to-video output",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(jobs.router)
app.include_router(feedback.router)


@app.on_event("startup")
async def startup():
    init_db()
    log.info("Database initialised")
    if os.environ.get("SKIP_VISUAL_EMBEDDINGS_BELOW_RATIO", "3") != "999":
        try:
            from packages.ml.movinet import load_model
            load_model()
            log.info("MoViNet pre-warmed")
        except Exception as e:
            log.warning(f"MoViNet pre-warm skipped: {e}")
    log.info("VideoSum API ready")


@app.get("/health")
async def health():
    from packages.ml.whisper import get_usage_status
    return {"status": "ok", "groq_usage": get_usage_status()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "apps.api.main:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
        reload=os.environ.get("RELOAD", "false").lower() == "true",
    )
