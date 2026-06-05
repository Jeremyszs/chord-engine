"""Pydantic response schemas (the JSON contract) for the chord-engine API.

Every model in this file is part of the public API contract. Field names,
types, and descriptions must be kept in sync with any frontend consumers.
"""

from typing import Literal

from pydantic import BaseModel, Field


class ChordSegment(BaseModel):
    """A single chord segment with timing and confidence."""

    chord: str = Field(
        ...,
        description="Chord label in Root:quality notation.",
        examples=["C:maj", "A:min", "G:7", "F:maj7", "N"],
    )

    roman: str = Field(
        ...,
        description="Roman numeral relative to the detected key.",
        examples=["I", "V", "vi", "IV", "?"],
    )

    start: float = Field(
        ...,
        description="Start time in seconds, 3 decimal places (ms precision).",
        examples=[0.000, 4.032, 12.507],
    )

    end: float = Field(
        ...,
        description="End time in seconds, 3 decimal places.",
        examples=[4.032, 12.507, 180.250],
    )

    duration: float = Field(
        ...,
        description="Segment duration (end - start) in seconds, 3 decimal places.",
        examples=[4.032, 8.475, 3.120],
    )

    confidence: float = Field(
        ...,
        description="Model confidence for this chord label, 0.0–1.0.",
        examples=[0.913, 0.450, 0.998],
    )


class AnalysisResult(BaseModel):
    """Complete chord analysis result returned when a job finishes."""

    job_id: str = Field(
        ...,
        description="UUID4 hex string. Same ID returned at upload time.",
        examples=["a1b2c3d4e5f6789012345678abcdef01"],
    )

    status: Literal["completed"] = Field(
        default="completed",
        description='Always "completed" when a result is returned.',
    )

    audio_filename: str = Field(
        ...,
        description="Original uploaded filename.",
        examples=["hotel_california.mp3", "guitar_riff.wav"],
    )

    duration_seconds: float = Field(
        ...,
        description="Total audio duration in seconds, 2 decimal places.",
        examples=[245.78, 30.12, 5.00],
    )

    tempo_bpm: float = Field(
        ...,
        description="Detected tempo in beats per minute, 1 decimal place.",
        examples=[120.5, 98.0, 140.2],
    )

    key: str = Field(
        ...,
        description="Detected musical key in <note> <mode> notation.",
        examples=["C major", "E minor", "G major", "Bb major"],
    )

    progression: str = Field(
        ...,
        description="Most repeated chord loop in Roman numerals, arrow-separated.",
        examples=["I → V → vi → IV", "ii → V → I", "I → IV → V"],
    )

    chord_count: int = Field(
        ...,
        description="Number of unique chords detected in the song.",
        examples=[4, 7, 12],
    )

    segments: list[ChordSegment] = Field(
        ...,
        description="Time-ordered list of chord segments. This is the primary "
                    "data a frontend playback widget iterates over.",
    )

    raw_chords: list[str] | None = Field(
        default=None,
        description="Per-frame chord labels (one per BTC-model frame). "
                    'Only present if include_raw_chords=True on upload.',
        examples=[None, ["C:maj", "C:maj", "G:maj", "G:maj", "A:min"]],
    )

    processing_time_seconds: float = Field(
        ...,
        description="Server-side wall-clock time for the analysis pipeline.",
        examples=[2.34, 12.10, 0.85],
    )

    created_at: str = Field(
        ...,
        description="ISO 8601 UTC timestamp of job completion.",
        examples=["2026-06-04T10:23:01.482Z"],
    )

    source: Literal["upload", "youtube"] = Field(
        default="upload",
        description="Where the audio came from.",
    )

    youtube_url: str | None = Field(
        default=None,
        description="Original YouTube URL. Only present if source is 'youtube'.",
    )


class JobStatus(BaseModel):
    """Current status of an analysis job."""

    job_id: str = Field(
        ...,
        description="UUID4 hex string. Same ID returned at upload time.",
        examples=["a1b2c3d4e5f6789012345678abcdef01"],
    )

    status: Literal["queued", "processing", "completed", "failed"] = Field(
        ...,
        description="Current lifecycle state of the job.",
        examples=["queued", "processing", "completed", "failed"],
    )

    progress: int = Field(
        ...,
        description="Pipeline progress as an integer 0–100.",
        examples=[0, 45, 72, 100],
    )

    message: str = Field(
        ...,
        description="Human-readable status message describing the current stage.",
        examples=[
            "Job queued",
            "Extracting CQT chromagram...",
            "Running chord inference...",
            "Done",
        ],
    )

    created_at: str = Field(
        ...,
        description="ISO 8601 UTC timestamp of job creation.",
        examples=["2026-06-04T10:23:01.482Z"],
    )

    error: str | None = Field(
        default=None,
        description="Error description. Only present when status is 'failed'.",
        examples=[None, "File not found", "Audio shorter than minimum duration"],
    )


class UploadResponse(BaseModel):
    """Immediate response returned after a successful upload."""

    job_id: str = Field(
        ...,
        description="UUID4 hex string. Use to poll for status and fetch result.",
        examples=["a1b2c3d4e5f6789012345678abcdef01"],
    )

    status: Literal["queued"] = Field(
        default="queued",
        description='Always "queued" on the initial upload response.',
    )

    message: str = Field(
        ...,
        description="Human-readable confirmation message.",
        examples=["Audio received. Analysis queued."],
    )

    poll_url: str = Field(
        ...,
        description="Relative URL to poll for job progress.",
        examples=["/api/v1/jobs/a1b2c3d4/status"],
    )

    result_url: str = Field(
        ...,
        description="Relative URL to fetch the completed analysis result.",
        examples=["/api/v1/jobs/a1b2c3d4/result"],
    )


class YoutubeUploadResponse(UploadResponse):
    """Immediate response returned after submitting a YouTube URL."""

    video_title: str = Field(
        ...,
        description="Title of the YouTube video.",
        examples=["Never Gonna Give You Up"],
    )

    video_duration_seconds: int = Field(
        ...,
        description="Duration of the video in seconds.",
        examples=[212],
    )

    video_uploader: str = Field(
        ...,
        description="Name of the YouTube channel / uploader.",
        examples=["Rick Astley"],
    )

    thumbnail_url: str = Field(
        ...,
        description="URL of the video thumbnail image.",
        examples=["https://i.ytimg.com/vi/dQw4w9WgXcQ/maxresdefault.jpg"],
    )


class ErrorResponse(BaseModel):
    """Standard error body returned on all API failures."""

    error: str = Field(
        ...,
        description="Short machine-readable error code.",
        examples=[
            "file_too_large",
            "unsupported_format",
            "job_not_found",
            "job_not_complete",
            "processing_failed",
            "validation_error",
        ],
    )

    detail: str = Field(
        ...,
        description="Human-readable explanation of the error.",
        examples=[
            "File exceeds 50MB limit.",
            "Unsupported file extension '.txt'. Supported: .flac, .m4a, .mp3, .ogg, .wav",
            "Job a1b2c3d4 not found",
        ],
    )

    job_id: str | None = Field(
        default=None,
        description="Job ID associated with the error, if applicable.",
        examples=[None, "a1b2c3d4e5f6789012345678abcdef01"],
    )
