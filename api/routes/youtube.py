"""YouTube metadata endpoint (read-only).

Provides a single read-only endpoint that returns video metadata
using the public YouTube oEmbed API.  No audio is downloaded on the
server — the response includes a ``download_command`` the user can
run locally to obtain the audio file, which they then upload via
POST /api/v1/jobs.
"""

import logging

from fastapi import APIRouter, Query

from api.models.response import (
    ErrorResponse,
    YoutubeInfoResponse,
)
from engine.youtube import get_video_info, validate_youtube_url

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/youtube", tags=["youtube"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _json_error(status_code: int, error: str, detail: str):
    """Shortcut to return a JSON response with an ErrorResponse body."""
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse(error=error, detail=detail).model_dump(),
    )


# ---------------------------------------------------------------------------
# Endpoint — GET /api/v1/youtube/info
# ---------------------------------------------------------------------------


@router.get("/info", response_model=YoutubeInfoResponse)
async def youtube_info(
    url: str = Query(..., description="YouTube video URL to fetch metadata for."),
):
    """Fetch metadata for a YouTube video using the public oEmbed API.

    Returns the video title, author, thumbnail, and an exact ``yt-dlp``
    command the user can run locally to download the audio as MP3.

    This is a **read-only** endpoint — no audio is downloaded or
    processed server-side.  After downloading the audio locally, the
    user uploads the MP3 file via ``POST /api/v1/jobs``.

    Supported URL formats:

    * ``https://www.youtube.com/watch?v=XXXXXXXXXXX``
    * ``https://youtu.be/XXXXXXXXXXX``
    * ``https://music.youtube.com/watch?v=XXXXXXXXXXX``
    * ``https://www.youtube.com/shorts/XXXXXXXXXXX``

    Errors:
    - ``400`` ``invalid_youtube_url`` — URL does not match a supported
      YouTube format.
    - ``400`` ``youtube_unavailable`` — video is private, deleted, or
      otherwise unavailable.
    """
    # --- Validate URL format ------------------------------------------------
    if not validate_youtube_url(url):
        return _json_error(
            400,
            "invalid_youtube_url",
            "URL must be a valid YouTube link.",
        )

    # --- Fetch metadata via oEmbed ------------------------------------------
    try:
        info = get_video_info(url)
    except ValueError as exc:
        return _json_error(
            400,
            "youtube_unavailable",
            str(exc),
        )

    return YoutubeInfoResponse(
        title=info["title"],
        author=info["author"],
        thumbnail=info["thumbnail"],
        video_id=info["video_id"],
        url=info["url"],
        download_command=info["download_command"],
    )
