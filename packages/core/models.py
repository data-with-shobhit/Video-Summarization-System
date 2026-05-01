from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    PENDING = "pending"
    ANALYZING = "analyzing"
    AWAITING_RATIO = "awaiting_ratio"
    SUMMARIZING = "summarizing"
    DONE = "done"
    FAILED = "failed"


class VideoType(str, Enum):
    LECTURE = "lecture"
    INTERVIEW = "interview"
    DEMO = "demo"
    PODCAST = "podcast"
    TUTORIAL = "tutorial"
    SHORT_FILM = "short_film"
    MOVIE = "movie"
    EPISODE = "episode"
    UNKNOWN = "unknown"


class Ratioband(str, Enum):
    LIGHT = "light"       # 2-3x
    MODERATE = "moderate" # 4-6x
    HIGHLIGHT = "highlight" # 7-10x


# ── Density ──────────────────────────────────────────────────────────────────

@dataclass
class DensitySignals:
    semantic_redundancy: float   # mean pairwise cosine sim of chunk embeddings
    silence_ratio: float         # fraction of audio that is silence
    filler_word_rate: float      # um/uh/like per 100 words
    pacing_variance: float       # std dev of WPM across scenes
    lexical_density: float       # type-token ratio
    topic_count: int             # number of TextTiling segments
    visual_change_rate: float    # mean frame-to-frame MoViNet distance


class DensityResult(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    recommended_ratio: Optional[int] = Field(None, ge=2, le=10)
    ratio_confidence_interval: Optional[int] = None
    signals: dict = Field(default_factory=dict)
    message: Optional[str] = None


# ── Transcript ────────────────────────────────────────────────────────────────

class Word(BaseModel):
    word: str
    start: float
    end: float


class TranscriptSegment(BaseModel):
    id: int
    start: float
    end: float
    text: str
    words: list[Word] = Field(default_factory=list)


class Transcript(BaseModel):
    language: str
    segments: list[TranscriptSegment]


# ── Scene / Topic ─────────────────────────────────────────────────────────────

class Scene(BaseModel):
    scene_index: int
    start_s: float
    end_s: float
    duration_s: float


class TopicSegment(BaseModel):
    segment_index: int
    start_s: float
    end_s: float
    chunk_indices: list[int]
    text: str
    label: Optional[str] = None


# ── Scoring ───────────────────────────────────────────────────────────────────

class SegmentScore(BaseModel):
    id: int
    score: float = Field(ge=0.0, le=10.0)
    reason: Optional[str] = None
    llm_score: Optional[float] = None
    audio_emphasis: Optional[float] = None
    visual_salience: Optional[float] = None
    composite_score: Optional[float] = None


# ── Selection ─────────────────────────────────────────────────────────────────

class SelectionSegment(BaseModel):
    start_s: float
    end_s: float
    scene_index: Optional[int] = None
    topic_segment_index: Optional[int] = None


# ── Audio features ────────────────────────────────────────────────────────────

class AudioFeatures(BaseModel):
    scene_index: int
    start_s: float
    end_s: float
    rms_energy: float
    pitch_variance: float
    speech_rate_wpm: float
    silence_ratio: float


# ── Job ───────────────────────────────────────────────────────────────────────

class JobCreate(BaseModel):
    video_path: str
    ratio: str = "auto"
    name: Optional[str] = None
    video_type: VideoType = VideoType.UNKNOWN


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    video_hash: Optional[str] = None
    name: Optional[str] = None
    density_result: Optional[DensityResult] = None
    output_path: Optional[str] = None
    error: Optional[str] = None


class ConfirmRatioRequest(BaseModel):
    ratio: int = Field(ge=2, le=10)


class ResummarizeRequest(BaseModel):
    ratio: int = Field(ge=2, le=10)


class FeedbackRequest(BaseModel):
    rating: int = Field(ge=1, le=5)
    comment: Optional[str] = None
    actual_ratio_used: Optional[int] = None
