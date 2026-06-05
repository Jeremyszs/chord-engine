"""In-memory job state management with thread-safe operations.

Stores job metadata and results in a dictionary guarded by a
threading.Lock. Jobs are lost on server restart. Not safe for
multi-process deployments.
"""

import os
import uuid
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone


@dataclass
class JobRecord:
    """A single job's complete state, stored in memory."""

    job_id: str
    status: str           # "queued" | "processing" | "completed" | "failed"
    progress: int         # 0–100
    message: str
    audio_path: str       # Absolute path to the uploaded temp file
    audio_filename: str   # Original filename from the upload
    params: dict          # smooth_method, device, include_raw_chords
    result: dict | None   # The full AnalysisResult dict, once complete
    error: str | None     # Error message if failed
    created_at: datetime
    source: str = "upload"
    youtube_url: str | None = None


class JobStore:
    """Thread-safe in-memory store for audio analysis jobs."""

    def __init__(self):
        self._jobs: dict[str, JobRecord] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create(self, audio_path: str, audio_filename: str, params: dict,
               source: str = "upload", youtube_url: str | None = None) -> JobRecord:
        """Create a new queued job and return its record.

        Args:
            audio_path: Absolute path to the uploaded temp file.
            audio_filename: Original filename from the upload.
            params: Analysis parameters (smooth_method, device, ...).
            source: ``"upload"`` for file uploads, ``"youtube"`` for YouTube jobs.
            youtube_url: The original YouTube URL, if applicable.

        Returns:
            The newly created JobRecord with status="queued".
        """
        record = JobRecord(
            job_id=uuid.uuid4().hex,
            status="queued",
            progress=0,
            message="Job queued",
            audio_path=audio_path,
            audio_filename=audio_filename,
            params=params,
            result=None,
            error=None,
            created_at=datetime.now(timezone.utc),
            source=source,
            youtube_url=youtube_url,
        )
        with self._lock:
            self._jobs[record.job_id] = record
        return record

    def get(self, job_id: str) -> JobRecord | None:
        """Retrieve a job record by ID.

        Args:
            job_id: The job identifier.

        Returns:
            The JobRecord if found, or None.
        """
        with self._lock:
            return self._jobs.get(job_id)

    def update_progress(self, job_id: str, progress: int, message: str) -> None:
        """Atomically update progress percentage and status message.

        Args:
            job_id: The job identifier.
            progress: Integer 0–100.
            message: Human-readable status message.

        Raises:
            KeyError: If the job ID does not exist.
        """
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                raise KeyError(f"Job not found: {job_id}")
            record.progress = progress
            record.message = message
            if record.status == "queued" and progress > 0:
                record.status = "processing"

    def set_completed(self, job_id: str, result: dict) -> None:
        """Mark a job as completed and attach the result.

        Args:
            job_id: The job identifier.
            result: The AnalysisResult dict.

        Raises:
            KeyError: If the job ID does not exist.
        """
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                raise KeyError(f"Job not found: {job_id}")
            record.status = "completed"
            record.progress = 100
            record.message = "Done"
            record.result = result

    def set_failed(self, job_id: str, error: str) -> None:
        """Mark a job as failed and store the error message.

        Args:
            job_id: The job identifier.
            error: Human-readable error description.

        Raises:
            KeyError: If the job ID does not exist.
        """
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                raise KeyError(f"Job not found: {job_id}")
            record.status = "failed"
            record.error = error
            record.message = error

    def delete(self, job_id: str) -> bool:
        """Remove a job from the store and its temp audio file.

        Args:
            job_id: The job identifier.

        Returns:
            True if the job was found and removed, False otherwise.
        """
        record = None
        with self._lock:
            record = self._jobs.pop(job_id, None)
        if record is not None:
            self._try_remove_file(record.audio_path)
            return True
        return False

    def cleanup_old_jobs(self, max_age_minutes: int = 60) -> int:
        """Remove jobs older than *max_age_minutes* and their temp files.

        Args:
            max_age_minutes: Age threshold in minutes (default 60).

        Returns:
            Number of jobs cleaned up.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=max_age_minutes)

        to_delete: list[JobRecord] = []
        with self._lock:
            for jid, record in list(self._jobs.items()):
                if record.created_at < cutoff:
                    to_delete.append(record)
                    del self._jobs[jid]

        # Delete temp audio files outside the lock to minimise contention.
        for record in to_delete:
            self._try_remove_file(record.audio_path)

        return len(to_delete)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _try_remove_file(path: str) -> None:
        """Remove a file if it exists; swallow errors."""
        try:
            os.remove(path)
        except (FileNotFoundError, PermissionError, OSError):
            pass


# Module-level singleton
job_store = JobStore()
