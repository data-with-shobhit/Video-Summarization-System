from __future__ import annotations

"""Integration tests for the FastAPI app (no real ML calls)."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path):
    os.environ["STORAGE_BACKEND"] = "local"
    os.environ["LOCAL_STORAGE_PATH"] = str(tmp_path / "data")
    os.environ["SQLITE_PATH"] = str(tmp_path / "videosum.db")
    os.environ["CACHE_BACKEND"] = "memory"

    from apps.api.main import app
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_create_job_file_not_found(client):
    r = client.post("/jobs", json={"video_path": "/nonexistent/video.mp4"})
    assert r.status_code == 400


def test_get_job_not_found(client):
    r = client.get("/jobs/does-not-exist")
    assert r.status_code == 404


def test_create_job_and_get_status(client, tmp_path):
    # Create a fake video file
    fake_video = tmp_path / "test.mp4"
    fake_video.write_bytes(b"fake video content")

    with patch("apps.api.routers.jobs._run_phase1"):
        r = client.post("/jobs", json={"video_path": str(fake_video)})

    assert r.status_code == 202
    job_id = r.json()["job_id"]
    assert job_id

    r2 = client.get(f"/jobs/{job_id}")
    assert r2.status_code == 200
    assert r2.json()["job_id"] == job_id
