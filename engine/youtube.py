"""YouTube audio download and validation for the chord-engine pipeline.

Provides URL validation, metadata fetching, and audio download services
using yt-dlp. Integrates with the existing engine by producing an MP3
file that ``engine/loader.py`` can load normally.
"""

import importlib.metadata
import logging
import os
import re
import time
from typing import Final

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_DURATION_SECONDS: Final[int] = 600  # 10 minutes
_MAX_RETRIES: Final[int] = 3
_RETRY_DELAY_SECONDS: Final[float] = 2.0

# Regex that matches all supported YouTube URL formats.
_YOUTUBE_RE: Final[re.Pattern] = re.compile(
    r"^https?://"
    r"(?:www\.|music\.)?"
    r"(?:youtube\.com/(?:watch\?v=|shorts/)|youtu\.be/)"
    r"([a-zA-Z0-9_-]{11})"
    r"(?:[?&#]\S*)?$"
)

# Shared yt-dlp options used for both metadata and download.
# The 'android' client avoids the "Sign in to confirm you're not a bot" wall
# that plagues cloud/containerised environments (HF Spaces, etc.).
_BASE_OPTS: Final[dict] = {
    "quiet": True,
    "no_warnings": True,
    "extractor_args": {"youtube": {"client": ["android"]}},
    "extractor_retries": 3,
    "file_access_retries": 3,
    "retries": 5,
}

# ---------------------------------------------------------------------------
# URL validation
# ---------------------------------------------------------------------------


def validate_youtube_url(url: str) -> bool:
    """Check whether *url* is a valid, supported YouTube link.

    Supported formats:

    * ``https://www.youtube.com/watch?v=XXXXXXXXXXX``
    * ``https://youtu.be/XXXXXXXXXXX``
    * ``https://music.youtube.com/watch?v=XXXXXXXXXXX``
    * ``https://www.youtube.com/shorts/XXXXXXXXXXX``

    This function uses only regular expressions — it makes **no** network
    request.  It does **not** verify whether the video actually exists or
    is accessible; for that, use :func:`get_video_metadata`.

    Args:
        url: A candidate URL string.

    Returns:
        ``True`` if the URL matches a supported YouTube format,
        ``False`` otherwise.
    """
    if not url or not isinstance(url, str):
        return False
    return _YOUTUBE_RE.match(url.strip()) is not None


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def get_video_metadata(url: str) -> dict:
    """Fetch metadata for a YouTube video **without** downloading it.

    Uses yt-dlp in quiet extract-info mode.  Retries up to 3 times on
    transient SSL / network errors.  Raises if the video is unavailable,
    private, deleted, or if its duration exceeds the 10-minute limit.

    Args:
        url: A valid YouTube URL (call :func:`validate_youtube_url` first).

    Returns:
        A dictionary with keys:

        * **title** — video title.
        * **duration** — video duration in seconds.
        * **uploader** — channel / uploader name.
        * **thumbnail** — URL of the video thumbnail.

    Raises:
        ValueError:
            * Video is unavailable, private, or deleted.
            * Video duration exceeds 10 minutes (600 seconds).
    """
    import yt_dlp

    ydl_opts: dict = {
        **_BASE_OPTS,
        # Do **not** download anything — only extract info.
        "extract_flat": False,
    }

    last_exc: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
            last_exc = None
            break  # success
        except yt_dlp.utils.DownloadError as exc:
            last_exc = exc
            err_str = str(exc).lower()
            # Don't retry on clearly permanent errors.
            if "private" in err_str or "deleted" in err_str or "unavailable" in err_str:
                break
            logger.warning(
                "yt-dlp metadata attempt %d/%d failed: %s", attempt, _MAX_RETRIES, exc
            )
            if attempt < _MAX_RETRIES:
                time.sleep(_RETRY_DELAY_SECONDS * attempt)
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "yt-dlp metadata attempt %d/%d failed: %s", attempt, _MAX_RETRIES, exc
            )
            if attempt < _MAX_RETRIES:
                time.sleep(_RETRY_DELAY_SECONDS * attempt)

    if last_exc is not None:
        raise ValueError(
            "Video is unavailable, private, or deleted."
        ) from last_exc

    if info is None:
        raise ValueError("Could not retrieve video metadata.")

    duration: int | None = info.get("duration")
    if duration is not None and duration > _MAX_DURATION_SECONDS:
        raise ValueError(
            f"Video too long. Maximum duration is 10 minutes "
            f"(got {duration} seconds)."
        )

    return {
        "title": info.get("title", "Unknown"),
        "duration": duration or 0,
        "uploader": info.get("uploader", info.get("channel", "Unknown")),
        "thumbnail": info.get("thumbnail", ""),
    }


# ---------------------------------------------------------------------------
# Audio download
# ---------------------------------------------------------------------------


def download_audio(url: str, output_dir: str) -> str:
    """Download audio from a YouTube video and convert it to MP3.

    Uses yt-dlp with the FFmpeg audio extract post-processor to produce
    a 192 kbps MP3 file.  The file is saved into *output_dir* with the
    video ID as its filename for predictability.  Retries up to 3 times
    on transient network / SSL errors.

    Args:
        url: A valid YouTube URL.
        output_dir: Absolute path to a writable directory (e.g. a
            ``/tmp`` subdirectory).  Will be created if it does not
            exist.

    Returns:
        Absolute path to the downloaded ``.mp3`` file.  The file is
        guaranteed to exist on disk when this function returns.

    Raises:
        RuntimeError: If the download or conversion fails for any
            reason (network error, FFmpeg not available, disk full,
            etc.).
    """
    import yt_dlp

    os.makedirs(output_dir, exist_ok=True)

    ydl_opts: dict = {
        **_BASE_OPTS,
        "format": "bestaudio/best",
        # Use the stable video ID in the filename to avoid issues with
        # special characters in titles and cross-platform sanitisation.
        "outtmpl": os.path.join(output_dir, "%(id)s.%(ext)s"),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
    }

    last_exc: Exception | None = None
    info: dict | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
            last_exc = None
            break  # success
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "yt-dlp download attempt %d/%d failed: %s", attempt, _MAX_RETRIES, exc
            )
            if attempt < _MAX_RETRIES:
                time.sleep(_RETRY_DELAY_SECONDS * attempt)

    if last_exc is not None:
        raise RuntimeError(
            f"Failed to download audio from YouTube: {last_exc}"
        ) from last_exc

    if info is None:
        raise RuntimeError("YouTube download returned no info.")

    video_id: str = info.get("id", "audio")
    expected_path: str = os.path.join(output_dir, f"{video_id}.mp3")

    if not os.path.isfile(expected_path):
        # Fallback: scan output_dir for any .mp3 file
        for fname in os.listdir(output_dir):
            if fname.endswith(".mp3"):
                expected_path = os.path.join(output_dir, fname)
                break
        else:
            raise RuntimeError(
                f"Download appeared to succeed but MP3 file not found "
                f"in {output_dir}."
            )

    return expected_path
