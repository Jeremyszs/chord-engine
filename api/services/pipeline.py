"""Async pipeline that runs the full chord-engine for a given job.

Each blocking engine call is offloaded to a thread pool executor so the
FastAPI event loop stays responsive. Progress is reported to the job store
at every stage for live client feedback.
"""

import asyncio
import os
import time
from datetime import datetime, timezone

from engine.loader import load_audio
from engine.features import extract_chroma, beat_sync_chroma
from engine.model import load_detector
from engine.postprocess import (
    smooth_chords,
    merge_segments,
    infer_key,
    to_roman_numerals,
    extract_progression,
)
from engine.output import build_output
from api.services.job_store import job_store
from config import MODEL, FEATURES

# ---------------------------------------------------------------------------
# Model cache — the BTC model is expensive to load, so we keep a single
# instance at module level and reuse it across jobs.
# ---------------------------------------------------------------------------
_detector = None


def get_detector(device: str):
    """Return the shared BTC detector, loading it on first call."""
    global _detector
    if _detector is None:
        _detector = load_detector(device)
    return _detector


# ---------------------------------------------------------------------------
# Pipeline stages (run inside the executor)
# ---------------------------------------------------------------------------

def _run_stages(job_id: str) -> dict:
    """Synchronous pipeline body — runs inside a thread-pool worker.

    Args:
        job_id: The job to process.

    Returns:
        Analysis result dictionary ready to store on the job record.
    """
    record = job_store.get(job_id)
    if record is None:
        raise ValueError(f"Job not found: {job_id}")

    audio_path = record.audio_path
    device = record.params.get("device", "cpu")
    smooth_method = record.params.get("smooth_method", "hmm")
    include_raw = record.params.get("include_raw_chords", False)

    # ---- 5% ---------------------------------------------------------------
    job_store.update_progress(job_id, 5, "Loading audio...")
    audio_dict = load_audio(audio_path)
    y = audio_dict["y"]
    sr = audio_dict["sr"]

    # ---- 15% --------------------------------------------------------------
    job_store.update_progress(job_id, 15, "Extracting CQT chromagram...")
    chroma = extract_chroma(audio_dict)

    # ---- 30% --------------------------------------------------------------
    job_store.update_progress(job_id, 30, "Syncing to beat grid...")
    beat_chroma, beat_times = beat_sync_chroma(
        chroma, y, sr, hop_length=FEATURES["hop_length"]
    )

    # ---- 45% --------------------------------------------------------------
    job_store.update_progress(job_id, 45, "Loading BTC model...")
    detector = get_detector(device)

    # ---- 60% --------------------------------------------------------------
    job_store.update_progress(job_id, 60, "Running chord inference...")
    raw_chords = detector.predict(y, sr)

    # ---- 72% --------------------------------------------------------------
    job_store.update_progress(job_id, 72, "Smoothing chord sequence...")
    smoothed = smooth_chords(raw_chords, method=smooth_method)

    # ---- 82% --------------------------------------------------------------
    job_store.update_progress(job_id, 82, "Building chord segments...")
    hop_length_btc = MODEL["hop_length"]
    frame_times = [i * hop_length_btc / sr for i in range(len(smoothed))]
    import numpy as np
    segments = merge_segments(smoothed, np.array(frame_times))

    # ---- 88% --------------------------------------------------------------
    job_store.update_progress(job_id, 88, "Inferring key and progression...")
    key = infer_key(segments)
    segments = to_roman_numerals(segments, key)
    progression = extract_progression(segments)

    # ---- 95% --------------------------------------------------------------
    job_store.update_progress(job_id, 95, "Formatting output...")
    output = build_output(
        segments=segments,
        key=key,
        progression=progression,
        audio_dict=audio_dict,
        raw_chords=list(raw_chords),
    )

    # Build the dict consumed by the async wrapper
    chord_segments = []
    for seg in output["segments"]:
        chord_segments.append({
            "chord": seg["chord"],
            "roman": seg.get("roman", "?"),
            "start": round(seg["start"], 3),
            "end": round(seg["end"], 3),
            "duration": round(seg["duration"], 3),
            "confidence": round(seg.get("confidence", 0.0), 3),
        })

    return {
        "job_id": job_id,
        "status": "completed",
        "audio_filename": record.audio_filename,
        "duration_seconds": round(output["duration_seconds"], 2),
        "tempo_bpm": round(output["tempo_bpm"], 1),
        "key": output["key"],
        "progression": output["progression"],
        "chord_count": output["chord_count"],
        "segments": chord_segments,
        "raw_chords": list(output["raw_chords"]) if include_raw else None,
        "processing_time_seconds": 0.0,  # filled in by async wrapper
        "created_at": "",
    }


# ---------------------------------------------------------------------------
# Public async entry point
# ---------------------------------------------------------------------------

async def run_pipeline(job_id: str) -> None:
    """Run the full chord-engine pipeline for *job_id*.

    Updates the job store with progress at each stage. On success the job
    is marked ``completed`` and the result is attached. On any failure the
    job is marked ``failed`` with the error message. The temp audio file is
    always cleaned up in a ``finally`` block.

    Args:
        job_id: Identifies the job to process.
    """
    loop = asyncio.get_event_loop()
    start_time = time.monotonic()
    audio_path = None

    # Snapshot the audio path before we start so we can clean up even
    # if the job record disappears.
    record = job_store.get(job_id)
    if record is not None:
        audio_path = record.audio_path

    try:
        result = await loop.run_in_executor(
            None,  # default thread-pool executor
            _run_stages,
            job_id,
        )

        # Fill in server-side timing metadata
        result["processing_time_seconds"] = round(time.monotonic() - start_time, 2)
        now = datetime.now(timezone.utc)
        result["created_at"] = now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"

        job_store.set_completed(job_id, result)

    except Exception as exc:
        job_store.set_failed(job_id, str(exc))

    finally:
        # Always remove the uploaded temp file, success or failure.
        if audio_path is not None:
            try:
                os.remove(audio_path)
            except (FileNotFoundError, PermissionError, OSError):
                pass
