"""Tests for job management — JobStore unit tests + endpoint integration tests.

Structure:
  1. Helpers (make_wav_bytes, fake results, mocks)
  2. Fixtures
  3. JobStore unit tests          (18 tests)
  4. Pipeline mock tests          ( 3 tests)
  5. Health endpoint tests        ( 2 tests)
  6. Upload endpoint tests        ( 7 tests)
  7. Status endpoint tests        ( 2 tests)
  8. Result endpoint tests        ( 4 tests)
  9. Schema validation tests      ( 3 tests)
"""

import io
import uuid
import datetime
import unittest.mock
from pathlib import Path

import datetime as _dt

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from pydantic import ValidationError

from api.main import app
from api.models.response import AnalysisResult, ChordSegment
from api.services.job_store import JobStore, job_store as _global_store


# ============================================================================
# Helpers
# ============================================================================


def make_wav_bytes(duration_seconds: float = 3, sr: int = 22050) -> bytes:
    """Generate a minimal valid WAV file as bytes in memory (no disk writes).

    Args:
        duration_seconds: Length of the generated audio in seconds.
        sr: Sample rate.

    Returns:
        WAV file content as ``bytes``.
    """
    num_samples = int(sr * duration_seconds)
    data = np.zeros(num_samples, dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, data, sr, format="WAV")
    return buf.getvalue()


def _fake_analysis_result(job_id: str, audio_filename: str = "test.wav") -> dict:
    """Return a valid AnalysisResult-shaped dict for testing.

    This is what the mocked ``run_pipeline`` will store when it
    ``set_completed`` for a job.
    """
    return {
        "job_id": job_id,
        "status": "completed",
        "audio_filename": audio_filename,
        "duration_seconds": 3.0,
        "tempo_bpm": 120.0,
        "key": "C major",
        "progression": "I → IV → V",
        "chord_count": 3,
        "segments": [
            {
                "chord": "C:maj",
                "roman": "I",
                "start": 0.0,
                "end": 1.0,
                "duration": 1.0,
                "confidence": 0.95,
            },
            {
                "chord": "F:maj",
                "roman": "IV",
                "start": 1.0,
                "end": 2.0,
                "duration": 1.0,
                "confidence": 0.88,
            },
            {
                "chord": "G:maj",
                "roman": "V",
                "start": 2.0,
                "end": 3.0,
                "duration": 1.0,
                "confidence": 0.92,
            },
        ],
        "raw_chords": None,
        "processing_time_seconds": 0.15,
        "created_at": "2026-06-04T12:00:00.000Z",
    }


def _complete_job_side_effect(job_id: str) -> None:
    """Side-effect for ``run_pipeline`` mock: immediately completes the job."""
    record = _global_store.get(job_id)
    if record is None:
        return
    result = _fake_analysis_result(job_id, record.audio_filename)
    _global_store.set_completed(job_id, result)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def store():
    """Provide a fresh isolated JobStore for each unit test."""
    return JobStore()


@pytest.fixture
def client():
    """Provide a sync test client wired to the in-memory app."""
    with TestClient(app) as c:
        yield c


# ============================================================================
# JobStore unit tests
# ============================================================================


class TestJobStoreCreate:
    def test_create_returns_valid_uuid(self, store):
        record = store.create("/tmp/test.wav", "test.wav", {"device": "cpu"})
        # Validate it's a proper UUID4 string
        parsed = uuid.UUID(record.job_id)
        assert parsed.version == 4

    def test_create_status_is_queued(self, store):
        record = store.create("/tmp/test.wav", "test.wav", {})
        assert record.status == "queued"

    def test_create_progress_zero(self, store):
        record = store.create("/tmp/test.wav", "test.wav", {})
        assert record.progress == 0

    def test_create_stores_params(self, store):
        params = {"device": "cuda", "smooth_method": "median"}
        record = store.create("/tmp/test.wav", "test.wav", params)
        assert record.params == params

    def test_create_stores_audio_path_and_filename(self, store):
        record = store.create("/tmp/test.wav", "test.wav", {})
        assert record.audio_path == "/tmp/test.wav"
        assert record.audio_filename == "test.wav"


class TestJobStoreGet:
    def test_get_existing(self, store):
        record = store.create("/tmp/test.wav", "test.wav", {})
        assert store.get(record.job_id) is record  # same object

    def test_get_nonexistent_returns_none(self, store):
        assert store.get("nonexistent") is None


class TestJobStoreUpdateProgress:
    def test_update_progress_changes_values(self, store):
        record = store.create("/tmp/test.wav", "test.wav", {})
        store.update_progress(record.job_id, 50, "Extracting features...")

        r = store.get(record.job_id)
        assert r.progress == 50
        assert r.message == "Extracting features..."

    def test_update_progress_transitions_to_processing(self, store):
        record = store.create("/tmp/test.wav", "test.wav", {})
        assert record.status == "queued"
        store.update_progress(record.job_id, 10, "Working...")
        assert record.status == "processing"

    def test_update_progress_nonexistent_raises(self, store):
        with pytest.raises(KeyError, match="Job not found"):
            store.update_progress("no-such-job", 50, "x")


class TestJobStoreSetCompleted:
    def test_set_completed_status(self, store):
        record = store.create("/tmp/test.wav", "test.wav", {})
        result = {"key": "C major", "segments": []}
        store.set_completed(record.job_id, result)

        r = store.get(record.job_id)
        assert r.status == "completed"
        assert r.progress == 100
        assert r.message == "Done"

    def test_set_completed_stores_result(self, store):
        record = store.create("/tmp/test.wav", "test.wav", {})
        result = {"key": "C major", "segments": []}
        store.set_completed(record.job_id, result)
        assert record.result == result

    def test_set_completed_nonexistent_raises(self, store):
        with pytest.raises(KeyError, match="Job not found"):
            store.set_completed("no-such-job", {})


class TestJobStoreSetFailed:
    def test_set_failed_status_and_error(self, store):
        record = store.create("/tmp/test.wav", "test.wav", {})
        store.set_failed(record.job_id, "Out of memory")

        r = store.get(record.job_id)
        assert r.status == "failed"
        assert r.error == "Out of memory"
        assert r.message == "Out of memory"

    def test_set_failed_nonexistent_raises(self, store):
        with pytest.raises(KeyError, match="Job not found"):
            store.set_failed("no-such-job", "error")


class TestJobStoreCleanup:
    def test_cleanup_old_jobs_removes_expired(self, store, tmp_path):
        # Create a temp file so the cleanup can try to remove it
        audio_file = tmp_path / "old.wav"
        audio_file.write_text("fake content")

        record = store.create(str(audio_file), "old.wav", {})
        # Manually set created_at to 2 hours ago
        record.created_at = datetime.datetime.now(datetime.timezone.utc) - _dt.timedelta(hours=2)

        count = store.cleanup_old_jobs(max_age_minutes=60)
        assert count == 1
        assert store.get(record.job_id) is None

    def test_cleanup_old_jobs_keeps_recent(self, store):
        record = store.create("/tmp/recent.wav", "recent.wav", {})
        count = store.cleanup_old_jobs(max_age_minutes=60)
        assert count == 0
        assert store.get(record.job_id) is not None

    def test_cleanup_old_jobs_deletes_temp_file(self, store, tmp_path):
        audio_file = tmp_path / "cleanup_test.wav"
        audio_file.write_text("audio data")

        record = store.create(str(audio_file), "cleanup_test.wav", {})
        record.created_at = datetime.datetime.now(datetime.timezone.utc) - _dt.timedelta(hours=2)

        store.cleanup_old_jobs(max_age_minutes=60)
        assert not audio_file.exists()

    def test_cleanup_old_jobs_missing_file_does_not_crash(self, store):
        record = store.create("/tmp/ghost.wav", "ghost.wav", {})
        record.created_at = datetime.datetime.now(datetime.timezone.utc) - _dt.timedelta(hours=2)

        # Should not raise even though file doesn't exist
        count = store.cleanup_old_jobs(max_age_minutes=60)
        assert count == 1


# ============================================================================
# Pipeline mock tests  (run_pipeline with all engine modules mocked)
# ============================================================================


class TestPipeline:
    """Test that run_pipeline progresses through all stages correctly.

    All engine modules are mocked so the test is fast and deterministic.
    Jobs are created on the global ``job_store`` singleton so the
    thread-pool worker in ``_run_stages`` can find them.
    """

    @pytest.mark.asyncio
    async def test_run_pipeline_completes(self, tmp_path):
        """run_pipeline should end with status=completed and progress=100."""
        audio_file = tmp_path / "test.wav"
        audio_file.write_text("fake audio content")

        record = _global_store.create(
            audio_path=str(audio_file),
            audio_filename="test.wav",
            params={"device": "cpu", "smooth_method": "hmm", "include_raw_chords": False},
        )

        with unittest.mock.patch.multiple(
            "api.services.pipeline",
            load_audio=mock_load_audio,
            extract_chroma=mock_extract_chroma,
            beat_sync_chroma=mock_beat_sync_chroma,
            get_detector=mock_get_detector,
            smooth_chords=mock_smooth_chords,
            merge_segments=mock_merge_segments,
            sync_chords_to_beats=mock_sync_chords_to_beats,
            infer_key=mock_infer_key,
            to_roman_numerals=mock_to_roman_numerals,
            extract_progression=mock_extract_progression,
            build_output=mock_build_output,
        ):
            from api.services.pipeline import run_pipeline
            await run_pipeline(record.job_id)

        final = _global_store.get(record.job_id)
        assert final is not None
        assert final.status == "completed"
        assert final.progress == 100
        assert final.result is not None
        assert final.result["job_id"] == record.job_id
        assert final.result["key"] == "C major"

    @pytest.mark.asyncio
    async def test_run_pipeline_failure_sets_failed(self):
        """When the engine raises, the job should end in 'failed'."""
        record = _global_store.create(
            audio_path="/tmp/does-not-exist.wav",
            audio_filename="missing.wav",
            params={"device": "cpu", "smooth_method": "hmm", "include_raw_chords": False},
        )

        # Don't mock — the real load_audio will fail on a missing file
        from api.services.pipeline import run_pipeline
        await run_pipeline(record.job_id)

        final = _global_store.get(record.job_id)
        assert final is not None
        assert final.status == "failed"
        assert final.error is not None

    @pytest.mark.asyncio
    async def test_run_pipeline_cleans_up_temp_file(self, tmp_path):
        """The temp audio file should be deleted after pipeline runs."""
        audio_file = tmp_path / "cleanme.wav"
        audio_file.write_text("content")

        record = _global_store.create(
            audio_path=str(audio_file),
            audio_filename="cleanme.wav",
            params={"device": "cpu", "smooth_method": "hmm", "include_raw_chords": False},
        )

        assert audio_file.exists()

        with unittest.mock.patch.multiple(
            "api.services.pipeline",
            load_audio=mock_load_audio,
            extract_chroma=mock_extract_chroma,
            beat_sync_chroma=mock_beat_sync_chroma,
            get_detector=mock_get_detector,
            smooth_chords=mock_smooth_chords,
            merge_segments=mock_merge_segments,
            sync_chords_to_beats=mock_sync_chords_to_beats,
            infer_key=mock_infer_key,
            to_roman_numerals=mock_to_roman_numerals,
            extract_progression=mock_extract_progression,
            build_output=mock_build_output,
        ):
            from api.services.pipeline import run_pipeline
            await run_pipeline(record.job_id)

        assert not audio_file.exists()


# Helper mocks — produce realistic return values for each engine function


def mock_load_audio(audio_path):
    """Return a minimal audio dict."""
    sr = 22050
    return {"y": np.zeros(sr * 2, dtype=np.float32), "sr": sr}


def mock_extract_chroma(audio_dict, config=None):
    """Return a plausible chromagram (12 bins × ~86 frames for 2s @ 22050)."""
    return np.random.rand(12, 86).astype(np.float32)


def mock_beat_sync_chroma(chroma, y, sr, hop_length=512):
    """Return beat-synced chroma and beat times."""
    n_beats = 8
    beat_chroma = np.random.rand(12, n_beats).astype(np.float32)
    beat_times = np.linspace(0.5, 2.0, n_beats)
    return beat_chroma, beat_times


def mock_get_detector(device):
    """Return a duck-typed detector stub."""

    class _StubDetector:
        def predict(self, y, sr):
            # Return a list of chord labels, one per ~86 frames
            chords = ["C:maj", "C:maj", "G:maj", "G:maj", "A:min", "A:min", "F:maj", "F:maj"]
            # Repeat to get roughly the right frame count
            n_frames = y.shape[0] // 256  # rough frame estimate
            return (chords * (n_frames // len(chords) + 1))[:n_frames]

    return _StubDetector()


def mock_smooth_chords(chord_labels, method=None, confidences=None):
    """Pass through unchanged for simplicity."""
    return list(chord_labels)


def mock_merge_segments(chord_labels, times):
    """Return simplified segments."""
    # Group consecutive identical labels
    segments = []
    if len(chord_labels) == 0:
        return segments
    current = chord_labels[0]
    start = float(times[0])
    for i in range(1, len(chord_labels)):
        if chord_labels[i] != current:
            end = float((times[i - 1] + (times[i] if i < len(times) else (times[-1] * 1.01))) / 2)
            segments.append({
                "chord": current,
                "start": start,
                "end": end,
                "duration": round(end - start, 3),
                "confidence": 0.9,
            })
            current = chord_labels[i]
            start = end
    end = float(times[-1]) + 0.5
    segments.append({
        "chord": current,
        "start": start,
        "end": end,
        "duration": round(end - start, 3),
        "confidence": 0.9,
    })
    return segments


def mock_sync_chords_to_beats(segments, beat_times, sr, hop_length):
    """Pass through segments unchanged for simplicity."""
    return segments


def mock_infer_key(segments):
    return "C major"


def mock_to_roman_numerals(segments, key):
    roman_map = {"C:maj": "I", "G:maj": "V", "A:min": "vi", "F:maj": "IV", "N": "?"}
    for seg in segments:
        seg["roman"] = roman_map.get(seg["chord"], "?")
    return segments


def mock_extract_progression(segments):
    return "I → V → vi → IV"


def mock_build_output(segments, key, progression, audio_dict, raw_chords, confidences=None, beats=None):
    """Return a dict matching what build_output normally returns."""
    y = audio_dict["y"]
    sr = audio_dict["sr"]
    duration = float(len(y) / sr)
    result = {
        "key": key,
        "progression": progression,
        "tempo_bpm": 120.0,
        "duration_seconds": duration,
        "chord_count": len(set(s["chord"] for s in segments if s["chord"] not in ("N", "X", "-"))),
        "segments": segments,
        "raw_chords": raw_chords,
    }
    if beats is not None:
        result["beats"] = [float(t) for t in beats]
    return result


# ============================================================================
# Health endpoint tests
# ============================================================================


class TestHealth:
    """Health-check endpoints (GET /api/v1/health and /health/ping)."""

    def test_health_ok(self, client):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"

    def test_health_ping(self, client):
        response = client.get("/api/v1/health/ping")
        assert response.status_code == 200
        assert response.json() == {"ping": "pong"}


# ============================================================================
# Upload endpoint tests
# ============================================================================


class TestUpload:
    """POST /api/v1/jobs — happy path."""

    def test_upload_valid_wav_returns_202(self, client):
        files = {"audio": ("song.wav", make_wav_bytes(), "audio/wav")}
        response = client.post("/api/v1/jobs", files=files)
        assert response.status_code == 202

    def test_upload_valid_wav_has_job_id(self, client):
        files = {"audio": ("song.wav", make_wav_bytes(), "audio/wav")}
        response = client.post("/api/v1/jobs", files=files)
        body = response.json()
        assert "job_id" in body
        # Validate UUID4 format
        parsed = uuid.UUID(body["job_id"])
        assert parsed.version == 4

    def test_upload_valid_wav_has_urls(self, client):
        files = {"audio": ("song.wav", make_wav_bytes(), "audio/wav")}
        response = client.post("/api/v1/jobs", files=files)
        body = response.json()
        job_id = body["job_id"]
        assert body["poll_url"] == f"/api/v1/jobs/{job_id}/status"
        assert body["result_url"] == f"/api/v1/jobs/{job_id}/result"

    def test_upload_valid_wav_status_queued(self, client):
        files = {"audio": ("song.wav", make_wav_bytes(), "audio/wav")}
        response = client.post("/api/v1/jobs", files=files)
        body = response.json()
        assert body["status"] == "queued"
        assert body["message"] == "Audio received. Analysis queued."


class TestUploadValidation:
    """POST /api/v1/jobs — validation errors."""

    def test_upload_no_file_returns_422(self, client):
        response = client.post("/api/v1/jobs")
        assert response.status_code == 422
        body = response.json()
        assert body["error"] == "validation_error"

    def test_upload_txt_file_returns_400(self, client):
        files = {"audio": ("notes.txt", b"hello world", "text/plain")}
        response = client.post("/api/v1/jobs", files=files)
        assert response.status_code == 400
        body = response.json()
        assert body["error"] == "unsupported_format"

    def test_upload_large_file_returns_413(self, client):
        oversized = b"x" * (51 * 1024 * 1024)
        files = {"audio": ("big.wav", oversized, "audio/wav")}
        response = client.post("/api/v1/jobs", files=files)
        assert response.status_code == 413
        body = response.json()
        assert body["error"] == "file_too_large"


# ============================================================================
# Status polling tests
# ============================================================================


class TestJobStatus:
    """GET /api/v1/jobs/{job_id}/status."""

    def test_get_status_valid_job(self, client):
        files = {"audio": ("song.wav", make_wav_bytes(), "audio/wav")}
        upload_resp = client.post("/api/v1/jobs", files=files)
        job_id = upload_resp.json()["job_id"]

        response = client.get(f"/api/v1/jobs/{job_id}/status")
        assert response.status_code == 200
        body = response.json()
        assert body["job_id"] == job_id
        assert body["status"] in ("queued", "processing", "completed", "failed")
        assert isinstance(body["progress"], int)
        assert 0 <= body["progress"] <= 100
        assert isinstance(body["message"], str)
        assert isinstance(body["created_at"], str)

    def test_get_status_nonexistent_job(self, client):
        response = client.get("/api/v1/jobs/nonexistent-id/status")
        assert response.status_code == 404
        body = response.json()
        assert body["error"] == "job_not_found"


# ============================================================================
# Result fetch tests
# ============================================================================


class TestJobResult:
    """GET /api/v1/jobs/{job_id}/result."""

    def test_get_result_nonexistent(self, client):
        response = client.get("/api/v1/jobs/nonexistent-id/result")
        assert response.status_code == 404
        body = response.json()
        assert body["error"] == "job_not_found"

    def test_get_result_queued(self, client):
        """Requesting the result while the job is still queued returns 409."""
        wav_bytes = make_wav_bytes()
        files = {"audio": ("song.wav", wav_bytes, "audio/wav")}

        with unittest.mock.patch(
            "api.routes.jobs.run_pipeline",
            side_effect=lambda **_: None,  # no-op so job stays queued
        ):
            response = client.post("/api/v1/jobs", files=files)
            job_id = response.json()["job_id"]

        result_resp = client.get(f"/api/v1/jobs/{job_id}/result")
        assert result_resp.status_code == 409
        body = result_resp.json()
        assert body["error"] == "job_not_complete"

    def test_get_result_failed(self, client):
        """Requesting the result for a failed job returns 422."""
        record = _global_store.create(
            audio_path="/tmp/fake.wav",
            audio_filename="fake.wav",
            params={"device": "cpu", "smooth_method": "hmm", "include_raw_chords": False},
        )
        _global_store.set_failed(record.job_id, "Something went wrong")

        response = client.get(f"/api/v1/jobs/{record.job_id}/result")
        assert response.status_code == 422
        body = response.json()
        assert body["error"] == "processing_failed"

    def test_get_result_completed_with_patch(self, client):
        """Full round-trip: upload → completed (mocked) → result validated.

        Uses a mock ``run_pipeline`` that immediately calls
        ``job_store.set_completed`` with a fake AnalysisResult dict, so the
        test never waits for real audio processing.
        """
        wav_bytes = make_wav_bytes()
        files = {"audio": ("song.wav", wav_bytes, "audio/wav")}

        with unittest.mock.patch(
            "api.routes.jobs.run_pipeline",
            side_effect=_complete_job_side_effect,
        ):
            upload_resp = client.post("/api/v1/jobs", files=files)
            assert upload_resp.status_code == 202
            job_id = upload_resp.json()["job_id"]

        # The mocked run_pipeline has already been called (TestClient
        # processes BackgroundTasks synchronously), so the job is completed.
        result_resp = client.get(f"/api/v1/jobs/{job_id}/result")
        assert result_resp.status_code == 200
        body = result_resp.json()
        assert body["job_id"] == job_id
        assert body["status"] == "completed"
        assert body["key"] == "C major"
        assert body["progression"] == "I → IV → V"
        assert body["chord_count"] == 3
        assert body["audio_filename"] == "song.wav"


# ============================================================================
# Schema validation tests
# ============================================================================


class TestResultSchema:
    """Validate that completed results conform to the Pydantic models."""

    def test_analysis_result_validates(self, client):
        """The raw response JSON should pass AnalysisResult.model_validate."""
        wav_bytes = make_wav_bytes()
        files = {"audio": ("song.wav", wav_bytes, "audio/wav")}

        with unittest.mock.patch(
            "api.routes.jobs.run_pipeline",
            side_effect=_complete_job_side_effect,
        ):
            upload_resp = client.post("/api/v1/jobs", files=files)
            job_id = upload_resp.json()["job_id"]

        result_resp = client.get(f"/api/v1/jobs/{job_id}/result")
        body = result_resp.json()

        # This raises if the shape doesn't match the model
        parsed = AnalysisResult.model_validate(body)
        assert parsed.job_id == job_id

    def test_every_segment_has_start_before_end(self, client):
        """Each ChordSegment should satisfy start < end."""
        wav_bytes = make_wav_bytes()
        files = {"audio": ("song.wav", wav_bytes, "audio/wav")}

        with unittest.mock.patch(
            "api.routes.jobs.run_pipeline",
            side_effect=_complete_job_side_effect,
        ):
            upload_resp = client.post("/api/v1/jobs", files=files)
            job_id = upload_resp.json()["job_id"]

        result_resp = client.get(f"/api/v1/jobs/{job_id}/result")
        body = result_resp.json()
        parsed = AnalysisResult.model_validate(body)

        for seg in parsed.segments:
            assert seg.start < seg.end, (
                f"Segment '{seg.chord}' has start={seg.start} >= end={seg.end}"
            )

    def test_every_segment_confidence_in_range(self, client):
        """Each ChordSegment should have 0.0 <= confidence <= 1.0."""
        wav_bytes = make_wav_bytes()
        files = {"audio": ("song.wav", wav_bytes, "audio/wav")}

        with unittest.mock.patch(
            "api.routes.jobs.run_pipeline",
            side_effect=_complete_job_side_effect,
        ):
            upload_resp = client.post("/api/v1/jobs", files=files)
            job_id = upload_resp.json()["job_id"]

        result_resp = client.get(f"/api/v1/jobs/{job_id}/result")
        body = result_resp.json()
        parsed = AnalysisResult.model_validate(body)

        for seg in parsed.segments:
            assert 0.0 <= seg.confidence <= 1.0, (
                f"Segment '{seg.chord}' has confidence={seg.confidence} "
                f"outside [0.0, 1.0]"
            )
