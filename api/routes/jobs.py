"""Job management endpoints: upload, status, and result retrieval.

All responses use the Pydantic schemas from ``api/models/response.py``.
"""

import secrets
import tempfile
from pathlib import Path
from typing import Literal

import logging

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

from config import API, STORAGE
from api.models.request import YoutubeRequest
from api.models.response import (
    AnalysisResult,
    ErrorResponse,
    JobStatus,
    UploadResponse,
    YoutubeUploadResponse,
)
from api.services.job_store import JobRecord, job_store
from api.services.pipeline import run_pipeline, run_pipeline_from_youtube
from engine.youtube import validate_youtube_url, get_video_metadata

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])

# ---------------------------------------------------------------------------
# Constants (from config)
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = set(API["supported_extensions"])
SUPPORTED_EXTENSIONS_STR = ", ".join(sorted(SUPPORTED_EXTENSIONS))
MAX_FILE_SIZE = API["max_file_size_bytes"]

# Temporary directory created once at import time; all uploaded files are
# saved here with their original filenames.
# On Hugging Face Spaces only /tmp is writable, so we create the temp dir
# under STORAGE["tmp_dir"].
_UPLOAD_DIR = Path(tempfile.mkdtemp(dir=STORAGE["tmp_dir"], prefix="chord-engine-"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _error(status_code: int, error: str, detail: str, job_id: str | None = None) -> JSONResponse:
    """Shortcut to return a JSON response carrying an ``ErrorResponse`` body."""
    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse(error=error, detail=detail, job_id=job_id).model_dump(),
    )


def _record_to_status(record: JobRecord) -> JobStatus:
    """Convert a JobRecord into a JobStatus response model."""
    return JobStatus(
        job_id=record.job_id,
        status=record.status,
        progress=record.progress,
        message=record.message,
        created_at=record.created_at.isoformat().replace("+00:00", "Z"),
        error=record.error,
    )


# ---------------------------------------------------------------------------
# Endpoint 1 — POST /api/v1/jobs
# ---------------------------------------------------------------------------


@router.post("", response_model=UploadResponse, status_code=202)
async def create_job(
    background_tasks: BackgroundTasks,
    audio: UploadFile = File(..., description="Audio file to analyze (.mp3, .wav, .flac, .ogg, .m4a)"),
    smooth_method: Literal["hmm", "median"] = Query("hmm", description="Chord smoothing method"),
    device: Literal["cpu", "cuda"] = Query("cpu", description="Inference device"),
    include_raw_chords: bool = Query(False, description="Include per-frame chord predictions in the result"),
):
    """Upload an audio file for chord analysis.

    Accepts MP3, WAV, FLAC, OGG, or M4A files up to 50 MB.
    Returns immediately (HTTP 202) with a ``job_id``.  The client
    should poll ``GET /api/v1/jobs/{job_id}/status`` to track progress
    and ``GET /api/v1/jobs/{job_id}/result`` to retrieve the completed
    analysis.  Typical analysis time is 5–30 seconds depending on
    song length and server load.

    Errors:
    - ``400`` ``unsupported_format`` — file extension not in the
      allowed list (``.mp3``, ``.wav``, ``.flac``, ``.ogg``, ``.m4a``).
    - ``413`` ``file_too_large`` — uploaded file exceeds the 50 MB limit.
      The server stops reading as soon as the limit is crossed.
    """
    # --- Validation -------------------------------------------------------

    if audio.filename is None:
        return _error(400, "invalid_file", "No filename provided")

    ext = Path(audio.filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return _error(
            400,
            "unsupported_format",
            f"Unsupported file extension '{ext}'. Supported: {SUPPORTED_EXTENSIONS_STR}",
        )

    # --- Streaming size check — stops early if the limit is exceeded -------

    contents = b""
    CHUNK_SIZE = 1024 * 1024  # 1 MiB per read
    while True:
        chunk = await audio.read(CHUNK_SIZE)
        if not chunk:
            break
        contents += chunk
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=ErrorResponse(
                    error="file_too_large",
                    detail="File exceeds 50MB limit.",
                ).model_dump(),
            )

    if len(contents) == 0:
        return _error(400, "invalid_file", "Uploaded file is empty")

    # --- Save file --------------------------------------------------------

    file_path = _UPLOAD_DIR / audio.filename
    # Avoid collisions by appending a short hex suffix
    if file_path.exists():
        stem = file_path.stem
        suffix = file_path.suffix
        file_path = _UPLOAD_DIR / f"{stem}_{secrets.token_hex(4)}{suffix}"

    file_path.write_bytes(contents)

    # --- Create job & launch pipeline ------------------------------------

    params = {
        "smooth_method": smooth_method,
        "device": device,
        "include_raw_chords": include_raw_chords,
    }
    record = job_store.create(
        audio_path=str(file_path),
        audio_filename=audio.filename,
        params=params,
    )

    background_tasks.add_task(run_pipeline, job_id=record.job_id)

    return UploadResponse(
        job_id=record.job_id,
        message="Audio received. Analysis queued.",
        poll_url=f"/api/v1/jobs/{record.job_id}/status",
        result_url=f"/api/v1/jobs/{record.job_id}/result",
    )


# ---------------------------------------------------------------------------
# Endpoint 2 — POST /api/v1/jobs/youtube
# ---------------------------------------------------------------------------


@router.post("/youtube", response_model=YoutubeUploadResponse, status_code=202)
async def create_youtube_job(
    background_tasks: BackgroundTasks,
    body: YoutubeRequest,
):
    """Analyse audio from a YouTube video.

    Accepts a YouTube URL in a JSON body.  The server fetches the video
    metadata, downloads the audio, and runs the full chord analysis
    pipeline.  Returns immediately (HTTP 202) with a ``job_id``.  The
    client should poll ``GET /api/v1/jobs/{job_id}/status`` to track
    progress and ``GET /api/v1/jobs/{job_id}/result`` to retrieve the
    completed analysis.

    Supported URL formats:

    * ``https://www.youtube.com/watch?v=XXXXXXXXXXX``
    * ``https://youtu.be/XXXXXXXXXXX``
    * ``https://music.youtube.com/watch?v=XXXXXXXXXXX``
    * ``https://www.youtube.com/shorts/XXXXXXXXXXX``

    Video length is limited to **10 minutes** to prevent abuse and
    excessive processing time.

    Errors:
    - ``400`` ``invalid_youtube_url`` — URL does not match a supported
      YouTube format.
    - ``400`` ``youtube_unavailable`` — video is private, deleted,
      unavailable, or exceeds the 10-minute limit.
    """
    # --- Validate URL format ------------------------------------------------
    if not validate_youtube_url(body.url):
        return _error(
            400,
            "invalid_youtube_url",
            "URL must be a valid YouTube link.",
        )

    # --- Fetch metadata (validates availability & duration) ------------------
    try:
        metadata = get_video_metadata(body.url)
    except ValueError as exc:
        return _error(
            400,
            "youtube_unavailable",
            str(exc),
        )

    # --- Create job ---------------------------------------------------------
    params = {
        "smooth_method": body.smooth_method,
        "device": body.device,
        "include_raw_chords": body.include_raw_chords,
    }
    record = job_store.create(
        audio_path=body.url,  # pipeline uses this to know where to download from
        audio_filename=f"{metadata['title']}.mp3",
        params=params,
        source="youtube",
        youtube_url=body.url,
    )

    # --- Launch pipeline ----------------------------------------------------
    background_tasks.add_task(run_pipeline_from_youtube, job_id=record.job_id)

    return YoutubeUploadResponse(
        job_id=record.job_id,
        message="YouTube audio received. Analysis queued.",
        poll_url=f"/api/v1/jobs/{record.job_id}/status",
        result_url=f"/api/v1/jobs/{record.job_id}/result",
        video_title=metadata["title"],
        video_duration_seconds=metadata["duration"],
        video_uploader=metadata["uploader"],
        thumbnail_url=metadata["thumbnail"],
    )


# ---------------------------------------------------------------------------
# Endpoint 3 — GET /api/v1/jobs/{job_id}/status
# ---------------------------------------------------------------------------


@router.get("/{job_id}/status", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Poll the progress of a chord analysis job.

    Jobs transition through ``queued`` → ``processing`` → ``completed``,
    or to ``failed`` if something goes wrong.  The ``progress`` field
    (0–100) advances through 10 checkpoints during processing.

    Recommended poll interval: every 2–3 seconds for responsive UIs,
    or every 5 seconds for low-traffic clients.

    Errors:
    - ``404`` ``job_not_found`` — the given job ID does not exist.
    """
    record = job_store.get(job_id)
    if record is None:
        return _error(404, "job_not_found", f"Job {job_id} not found")
    return _record_to_status(record)


# ---------------------------------------------------------------------------
# Endpoint 3 — GET /api/v1/jobs/{job_id}/result
# ---------------------------------------------------------------------------


@router.get("/{job_id}/result", response_model=AnalysisResult)
async def get_job_result(job_id: str):
    """Fetch the completed analysis result for a job.

    Returns the full chord analysis (key, tempo, chord segments,
    progression) as an ``AnalysisResult`` JSON body.  This endpoint
    only succeeds when the job has reached ``completed`` status;
    you must poll ``GET /api/v1/jobs/{job_id}/status`` first.

    Errors:
    - ``404`` ``job_not_found`` — the given job ID does not exist.
    - ``409`` ``job_not_complete`` — the job is still queued or
      processing; try again after a short delay.
    - ``422`` ``processing_failed`` — the analysis pipeline raised
      an error.  The ``detail`` field contains the error message.
    """
    record = job_store.get(job_id)
    if record is None:
        return _error(404, "job_not_found", f"Job {job_id} not found")

    if record.status in ("queued", "processing"):
        return _error(
            409,
            "job_not_complete",
            f"Job {job_id} is still {record.status}. "
            f"Poll /api/v1/jobs/{job_id}/status for progress.",
        )

    if record.status == "failed":
        return _error(
            422,
            "processing_failed",
            record.error or "An unknown error occurred during analysis.",
        )

    # record.status == "completed"
    result = record.result
    if result is None:
        return _error(422, "processing_failed", "Job completed but no result was stored.")

    return AnalysisResult(**result)
