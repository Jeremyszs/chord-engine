"""YouTube audio download and validation for the chord-engine pipeline.

Provides URL validation, metadata fetching, and audio download services
using yt-dlp.  On Hugging Face Spaces (where YouTube blocks the server
IPs), authentication is provided via a **browser cookies file** exported
from the user's real browser and stored as an HF Secret.

See ``COOKIES_SETUP.md`` for instructions on setting up cookies.

Integrates with the existing engine by producing an MP3 file that
``engine/loader.py`` can load normally.
"""

import logging
import os
import re
from typing import Final

import yt_dlp

from config import YOUTUBE as YT_CFG

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Regex that matches all supported YouTube URL formats.
_YOUTUBE_RE: Final[re.Pattern] = re.compile(
    r"^https?://"
    r"(?:www\.|music\.)?"
    r"(?:youtube\.com/(?:watch\?v=|shorts/)|youtu\.be/)"
    r"([a-zA-Z0-9_-]{11})"
    r"(?:[?&#]\S*)?$"
)

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


def get_video_id(url: str) -> str | None:
    """Extract the 11-character YouTube video ID from a URL.

    Args:
        url: A YouTube URL.

    Returns:
        The video ID string, or ``None`` if the URL is not a valid
        YouTube link.
    """
    m = _YOUTUBE_RE.match(url.strip())
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Shared yt-dlp options builder
# ---------------------------------------------------------------------------


def _build_ydl_opts(**overrides: object) -> dict:
    """Return a base set of yt-dlp options, merged with *overrides*.

    Reads ``cookies_path`` from ``config.YOUTUBE``.  If the path is
    set and the file exists on disk, the cookiefile option is included;
    if not, yt-dlp runs anonymously (which works on personal machines
    but will usually fail on HF Spaces).
    """
    opts: dict = {
        "quiet": True,
        "no_warnings": True,
    }

    cookies_path = YT_CFG.get("cookies_path")
    if cookies_path and os.path.isfile(cookies_path):
        opts["cookiefile"] = cookies_path

    opts.update(overrides)
    return opts


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def get_video_metadata(url: str) -> dict:
    """Fetch metadata for a YouTube video **without** downloading it.

    Uses yt-dlp in quiet extract-info mode.  If a cookies file is
    configured (via the ``YOUTUBE_COOKIES`` HF Secret), it is passed
    to yt-dlp to bypass YouTube's bot detection.

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
            * Video duration exceeds the configured maximum
              (default: 10 minutes / 600 seconds).
            * Network or extraction error from yt-dlp.
    """
    ydl_opts = _build_ydl_opts(skip_download=True)

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except yt_dlp.utils.DownloadError as exc:
        raise ValueError(
            f"Could not fetch video info: {exc}"
        ) from exc

    if info is None:
        raise ValueError("Could not fetch video info — yt-dlp returned nothing.")

    max_duration = YT_CFG.get("max_duration_seconds", 600)
    duration: int = info.get("duration", 0)
    if duration > max_duration:
        raise ValueError(
            f"Video too long. Maximum duration is {max_duration // 60} minutes "
            f"(got {duration} seconds)."
        )

    return {
        "title": info.get("title", "Unknown"),
        "duration": duration,
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
    video ID as its filename.

    If a cookies file is configured (via the ``YOUTUBE_COOKIES`` HF
    Secret), it is passed to yt-dlp to bypass YouTube's bot detection
    on HF Spaces.  Without cookies, yt-dlp will attempt an anonymous
    download — this works on personal machines but usually fails on
    cloud IP ranges.

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
            YouTube blocking, etc.).
    """
    os.makedirs(output_dir, exist_ok=True)

    ydl_opts = _build_ydl_opts(
        format="bestaudio/best",
        outtmpl=os.path.join(output_dir, "%(id)s.%(ext)s"),
        postprocessors=[{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": str(YT_CFG.get("audio_quality", "192")),
        }],
    )

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
    except yt_dlp.utils.DownloadError as exc:
        raise RuntimeError(
            "Could not download audio from YouTube. The server may be "
            "blocked by YouTube. Please download the audio file yourself "
            f"and use the file upload option instead. Error: {exc}"
        ) from exc

    if info is None:
        raise RuntimeError("yt-dlp returned no info after a successful download.")

    video_id: str = info.get("id", "")
    mp3_path: str = os.path.join(output_dir, f"{video_id}.mp3")

    if not os.path.isfile(mp3_path):
        # Fallback: scan the directory for any .mp3 file
        mp3_files = [f for f in os.listdir(output_dir) if f.endswith(".mp3")]
        if not mp3_files:
            raise RuntimeError(
                "Download appeared to succeed but no MP3 file was found "
                f"in {output_dir}."
            )
        mp3_path = os.path.join(output_dir, mp3_files[0])

    return mp3_path
