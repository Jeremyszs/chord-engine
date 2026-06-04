"""Tests for Pydantic request/response schemas."""

import pytest
from pydantic import ValidationError

from api.models.request import AnalyzeRequest
from api.models.response import (
    ChordSegment,
    AnalysisResult,
    JobStatus,
    UploadResponse,
    ErrorResponse,
)


class TestAnalyzeRequest:
    def test_defaults(self):
        r = AnalyzeRequest()
        assert r.smooth_method == "hmm"
        assert r.device == "cpu"
        assert r.include_raw_chords is False

    def test_custom_values(self):
        r = AnalyzeRequest(smooth_method="median", device="cuda", include_raw_chords=True)
        assert r.smooth_method == "median"
        assert r.device == "cuda"
        assert r.include_raw_chords is True

    def test_invalid_smooth_method(self):
        with pytest.raises(ValidationError):
            AnalyzeRequest(smooth_method="invalid")

    def test_invalid_device(self):
        with pytest.raises(ValidationError):
            AnalyzeRequest(device="gpu")


class TestChordSegment:
    def test_valid_segment(self):
        s = ChordSegment(
            chord="C:maj",
            roman="I",
            start=0.0,
            end=2.0,
            duration=2.0,
            confidence=0.95,
        )
        assert s.chord == "C:maj"
        assert s.roman == "I"
        assert s.confidence == 0.95

    def test_no_chord_label(self):
        s = ChordSegment(
            chord="N",
            roman="?",
            start=1.5,
            end=3.0,
            duration=1.5,
            confidence=0.5,
        )
        assert s.chord == "N"

    def test_missing_fields_raises(self):
        with pytest.raises(ValidationError):
            ChordSegment()


class TestAnalysisResult:
    def test_valid_without_raw_chords(self):
        r = AnalysisResult(
            job_id="abc123",
            audio_filename="test.wav",
            duration_seconds=30.0,
            tempo_bpm=120.5,
            key="C major",
            progression="I → IV → V",
            chord_count=5,
            segments=[],
            processing_time_seconds=2.34,
            created_at="2025-06-04T10:23:01.482Z",
        )
        assert r.status == "completed"
        assert r.raw_chords is None

    def test_valid_with_raw_chords(self):
        segments = [
            ChordSegment(chord="C:maj", roman="I", start=0.0, end=2.0, duration=2.0, confidence=0.9),
        ]
        r = AnalysisResult(
            job_id="abc123",
            audio_filename="test.wav",
            duration_seconds=10.0,
            tempo_bpm=120.0,
            key="Am",
            progression="vi → IV",
            chord_count=1,
            segments=segments,
            raw_chords=["C:maj", "C:maj", "C:maj"],
            processing_time_seconds=1.5,
            created_at="2025-06-04T10:23:01.482Z",
        )
        assert r.raw_chords == ["C:maj", "C:maj", "C:maj"]

    def test_missing_fields_raises(self):
        with pytest.raises(ValidationError):
            AnalysisResult()


class TestJobStatus:
    def test_valid_queued(self):
        s = JobStatus(
            job_id="abc123",
            status="queued",
            progress=0,
            message="Job queued",
            created_at="2025-06-04T10:23:01.482Z",
        )
        assert s.status == "queued"
        assert s.error is None

    def test_valid_failed_with_error(self):
        s = JobStatus(
            job_id="abc123",
            status="failed",
            progress=50,
            message="Out of memory",
            created_at="2025-06-04T10:23:01.482Z",
            error="Out of memory",
        )
        assert s.status == "failed"
        assert s.error == "Out of memory"

    def test_invalid_status(self):
        with pytest.raises(ValidationError):
            JobStatus(
                job_id="abc123",
                status="unknown",
                progress=0,
                message="x",
                created_at="x",
            )


class TestUploadResponse:
    def test_valid_upload_response(self):
        r = UploadResponse(
            job_id="abc123",
            message="Audio received. Analysis queued.",
            poll_url="/jobs/abc123",
            result_url="/jobs/abc123/result",
        )
        assert r.status == "queued"
        assert r.job_id == "abc123"

    def test_missing_fields_raises(self):
        with pytest.raises(ValidationError):
            UploadResponse()


class TestErrorResponse:
    def test_minimal(self):
        e = ErrorResponse(error="file_too_large", detail="File exceeds limit")
        assert e.job_id is None

    def test_with_job_id(self):
        e = ErrorResponse(
            error="job_not_found",
            detail="No such job",
            job_id="abc123",
        )
        assert e.job_id == "abc123"
