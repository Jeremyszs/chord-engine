"""Health check endpoints for the chord-engine API.

Two endpoints with different response-weight:
  - GET /api/v1/health       — full status (model loaded, uptime)
  - GET /api/v1/health/ping  — absolute minimal ping (no dependencies)
"""

import time

from fastapi import APIRouter, Request

from api.services.pipeline import _detector

router = APIRouter(prefix="/api/v1", tags=["health"])

VERSION = "1.0.0"
_FALLBACK_START = time.time()  # used when app.state is absent (e.g. tests)


@router.get("/health")
async def health_check(request: Request):
    """Return service health status.

    Includes model load status and server uptime in seconds.
    Relies on the globally cached ``_detector`` from the pipeline module.
    """
    start = getattr(request.app.state, "start_time", _FALLBACK_START)
    return {
        "status": "ok",
        "version": VERSION,
        "model_loaded": _detector is not None,
        "uptime_seconds": round(time.time() - start, 1),
    }


@router.get("/health/ping")
async def ping():
    """Lightest possible health check — instant, no logic, no dependencies."""
    return {"ping": "pong"}
