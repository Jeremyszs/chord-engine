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
    sync_chords_to_beats,
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
    """Return the shared BTC detector, loading it on first call.
    
    Uses ChordMiniApp's superior BTC-PL model by default for best accuracy.
    """
    global _detector
    if _detector is None:
        _detector = load_detector(device, use_chordmini=True)
    return _detector


# ---------------------------------------------------------------------------
# Shared analysis stages (used by both upload and YouTube pipelines)
# ---------------------------------------------------------------------------


def _run_analysis_stages(
    job_id: str,
    audio_path: str,
    audio_filename: str,
    device: str,
    smooth_method: str,
    include_raw: bool,
    progress_offset: int = 0,
) -> dict:
    """Run the core chord-analysis pipeline on an audio file.

    This function is called by both the upload pipeline and the YouTube
    pipeline.  The *progress_offset* parameter adjusts the progress
    reporting so that each caller can report a different base (e.g. 40%
    for YouTube where the first 40% was spent on download).

    Args:
        job_id: The job to process.
        audio_path: Path to the audio file on disk.
        audio_filename: Display name for the audio file.
        device: Inference device (``"cpu"`` or ``"cuda"``).
        smooth_method: Chord smoothing method (``"hmm"`` or ``"median"``).
        include_raw: Whether to include per-frame raw chords in output.
        progress_offset: Percentage offset to add to all progress reports.

    Returns:
        Analysis result dictionary ready to store on the job record.
    """
    # ---- Load audio -------------------------------------------------------
    job_store.update_progress(job_id, 5 + progress_offset, "Loading audio...")
    audio_dict = load_audio(audio_path)
    y = audio_dict["y"]
    sr = audio_dict["sr"]

    # ---- CQT chromagram ---------------------------------------------------
    job_store.update_progress(job_id, 15 + progress_offset, "Extracting CQT chromagram...")
    chroma = extract_chroma(audio_dict)

    # ---- Beat sync --------------------------------------------------------
    job_store.update_progress(job_id, 30 + progress_offset, "Syncing to beat grid...")
    beat_chroma, beat_times = beat_sync_chroma(
        chroma, y, sr, hop_length=FEATURES["hop_length"]
    )

    # ---- Model loading ----------------------------------------------------
    job_store.update_progress(job_id, 45 + progress_offset, "Loading BTC model...")
    detector = get_detector(device)

    # ---- Chord inference --------------------------------------------------
    job_store.update_progress(job_id, 60 + progress_offset, "Running chord inference...")
    raw_chords = detector.predict(y, sr)

    # ---- Smoothing --------------------------------------------------------
    job_store.update_progress(job_id, 72 + progress_offset, "Smoothing chord sequence...")
    smoothed = smooth_chords(raw_chords, method=smooth_method)

    # ---- Build segments ---------------------------------------------------
    job_store.update_progress(job_id, 82 + progress_offset, "Building chord segments...")
    hop_length_btc = MODEL["hop_length"]
    frame_times = [i * hop_length_btc / sr for i in range(len(smoothed))]
    import numpy as np
    segments = merge_segments(smoothed, np.array(frame_times))

    # ---- Beat synchronization ---------------------------------------------
    job_store.update_progress(job_id, 85 + progress_offset, "Syncing chords to beats...")
    segments = sync_chords_to_beats(segments, beat_times, sr, FEATURES["hop_length"])

    # ---- Key & progression ------------------------------------------------
    job_store.update_progress(job_id, 88 + progress_offset, "Inferring key and progression...")
    key = infer_key(segments)
    segments = to_roman_numerals(segments, key)
    progression = extract_progression(segments)

    # ---- Format output ----------------------------------------------------
    job_store.update_progress(job_id, 95 + progress_offset, "Formatting output...")
    output = build_output(
        segments=segments,
        key=key,
        progression=progression,
        audio_dict=audio_dict,
        raw_chords=list(raw_chords),
        beats=beat_times,
    )

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
        "audio_filename": audio_filename,
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
# Upload pipeline (synchronous body)
# ---------------------------------------------------------------------------


def _run_stages(job_id: str) -> dict:
    """Synchronous pipeline body for uploaded files."""
    record = job_store.get(job_id)
    if record is None:
        raise ValueError(f"Job not found: {job_id}")

    return _run_analysis_stages(
        job_id=job_id,
        audio_path=record.audio_path,
        audio_filename=record.audio_filename,
        device=record.params.get("device", "cpu"),
        smooth_method=record.params.get("smooth_method", "hmm"),
        include_raw=record.params.get("include_raw_chords", False),
        progress_offset=0,
    )


# ---------------------------------------------------------------------------
# Async wrapper
# ---------------------------------------------------------------------------


async def run_pipeline(job_id: str) -> None:
    """Run the full chord-engine pipeline for an uploaded audio file.

    Args:
        job_id: Identifies the job to process.
    """
    await _run_async_wrapper(job_id, _run_stages)


async def _run_async_wrapper(job_id: str, stages_fn) -> None:
    """Generic async wrapper around a synchronous stages function.

    Handles executor offloading, timing, error handling, and cleanup.

    Args:
        job_id: The job to process.
        stages_fn: Synchronous callable ``(job_id) -> dict`` that
            performs the actual work.
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
            stages_fn,
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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
