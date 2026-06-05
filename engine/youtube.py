"""YouTube audio download and validation for the chord-engine pipeline.

Provides URL validation, metadata fetching, and audio download services
using yt-dlp. Integrates with the existing engine by producing an MP3
file that ``engine/loader.py`` can load normally.
"""

import os
import re
from typing import Final

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_DURATION_SECONDS: Final[int] = 600  # 10 minutes

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


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def get_video_metadata(url: str) -> dict:
    """Fetch metadata for a YouTube video **without** downloading it.

    Uses yt-dlp in quiet extract-info mode.  Raises if the video is
    unavailable, private, deleted, or if its duration exceeds the
    10-minute limit.

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
        "quiet": True,
        "no_warnings": True,
        # Do **not** download anything — only extract info.
        "extract_flat": False,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except yt_dlp.utils.DownloadError as exc:
        raise ValueError(
            "Video is unavailable, private, or deleted."
        ) from exc
    except Exception as exc:
        raise ValueError(str(exc)) from exc

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
    video title as its filename.

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
        "quiet": True,
        "no_warnings": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download audio from YouTube: {exc}"
        ) from exc

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
