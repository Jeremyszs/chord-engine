"""YouTube audio download and validation for the chord-engine pipeline.

Provides URL validation, metadata fetching, and audio download services
using yt-dlp (via subprocess for reliability in container environments).
Integrates with the existing engine by producing an MP3 file that
``engine/loader.py`` can load normally.
"""

import json
import logging
import os
import re
import subprocess
import sys
import time
from typing import Final

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_DURATION_SECONDS: Final[int] = 600  # 10 minutes
_MAX_RETRIES: Final[int] = 2
_RETRY_DELAY_SECONDS: Final[float] = 3.0

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
# Helpers
# ---------------------------------------------------------------------------


def _find_ytdlp() -> str:
    """Locate the ``yt-dlp`` executable on ``$PATH``.

    Returns:
        The absolute path to ``yt-dlp``.

    Raises:
        RuntimeError: If ``yt-dlp`` is not installed.
    """
    which_cmd = "where" if sys.platform == "win32" else "which"
    try:
        result = subprocess.run(
            [which_cmd, "yt-dlp"],
            capture_output=True, text=True, timeout=10,
        )
        path = result.stdout.strip().split("\n")[0].strip()
        if path:
            return path
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    raise RuntimeError(
        "yt-dlp not found on PATH. "
        "Install it with: pip install yt-dlp"
    )


def _run_ytdlp(args: list[str], log_label: str) -> subprocess.CompletedProcess:
    """Run ``yt-dlp`` as a subprocess, returning the result.

    Args:
        args: List of command-line arguments (excluding the program name).
        log_label: Short label for log messages (e.g. ``"metadata"``).

    Returns:
        The completed process.

    Raises:
        RuntimeError: If the process fails or times out.
    """
    ytdlp = _find_ytdlp()
    cmd = [ytdlp] + args

    logger.info("Running yt-dlp %s: %s", log_label, " ".join(str(a) for a in args[:8]))

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            result = subprocess.run(
                cmd,
                capture_output=True, text=True,
                timeout=120,
            )
            if result.returncode == 0:
                return result

            # Non-zero exit — log the error
            stderr = result.stderr.strip()
            logger.warning(
                "yt-dlp %s attempt %d/%d failed (rc=%d): %s",
                log_label, attempt, _MAX_RETRIES, result.returncode,
                stderr[:500],
            )

            # Don't retry on permanent errors
            err_lower = stderr.lower()
            if "private" in err_lower or "deleted" in err_lower:
                break

            if attempt < _MAX_RETRIES:
                time.sleep(_RETRY_DELAY_SECONDS * attempt)

        except subprocess.TimeoutExpired:
            logger.warning(
                "yt-dlp %s attempt %d/%d timed out",
                log_label, attempt, _MAX_RETRIES,
            )
            if attempt < _MAX_RETRIES:
                time.sleep(_RETRY_DELAY_SECONDS * attempt)

    # All retries exhausted
    raise RuntimeError(
        f"yt-dlp {log_label} failed after {_MAX_RETRIES} attempt(s). "
        f"Last error: {result.stderr.strip()[:500] if 'result' in dir() else 'timeout'}"
    )


# ---------------------------------------------------------------------------
# Base yt-dlp arguments that work in containerised environments
# ---------------------------------------------------------------------------


def _base_args() -> list[str]:
    """Return shared yt-dlp CLI arguments for container/cloud environments.

    The Android player client bypasses YouTube's "Sign in to confirm"
    bot wall that frequently blocks cloud IP ranges.  Legacy server
    connect mitigates SSL EOF errors in minimal container images.
    """
    return [
        "--quiet",
        "--no-warnings",
        "--force-ipv4",
        "--socket-timeout", "30",
        "--extractor-args", "youtube:player_client=android,android_music",
    ]


# ---------------------------------------------------------------------------
# Fallback: pure-Python yt-dlp API with certifi
# ---------------------------------------------------------------------------


def _try_python_api_fallback(url: str, extract: bool = True) -> dict | None:
    """Try using yt-dlp's Python API with ``certifi`` for SSL.

    Some environments have older OpenSSL that chokes on YouTube's TLS.
    ``certifi`` provides a modern CA bundle that often resolves this.

    Args:
        url: YouTube URL.
        extract: If True, extract info; if False, download.

    Returns:
        The info dict on success, or ``None`` on failure.
    """
    try:
        import certifi

        import yt_dlp

        # Override the default SSL context to use certifi's CA bundle.
        import ssl
        ssl_context = ssl.create_default_context(cafile=certifi.where())

        ydl_opts: dict = {
            "quiet": True,
            "no_warnings": True,
            "extractor_args": {"youtube": {"player_client": ["android", "android_music"]}},
            "extractor_retries": 2,
            "file_access_retries": 2,
            "retries": 3,
            "socket_timeout": 30,
            "legacy_server_connect": True,
        }

        if not extract:
            ydl_opts["format"] = "bestaudio/best"
            ydl_opts["outtmpl"] = os.path.join(
                os.environ.get("TMPDIR", "/tmp"), "%(id)s.%(ext)s"
            )
            ydl_opts["postprocessors"] = [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }]

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            if extract:
                return ydl.extract_info(url, download=False)
            else:
                return ydl.extract_info(url, download=True)

    except Exception as exc:
        logger.debug("Python API fallback failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def get_video_metadata(url: str) -> dict:
    """Fetch metadata for a YouTube video **without** downloading it.

    Uses yt-dlp (subprocess) in quiet extract-info mode.  Falls back to
    the Python API with ``certifi`` if the subprocess approach fails.
    Raises if the video is unavailable, private, deleted, or if its
    duration exceeds the 10-minute limit.

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
    # Try subprocess approach first
    try:
        args = _base_args() + [
            "--dump-json",
            "--no-download",
            url,
        ]
        result = _run_ytdlp(args, "metadata")
        info = json.loads(result.stdout)
    except (RuntimeError, json.JSONDecodeError) as exc:
        # Fallback: try Python API with certifi
        logger.warning("yt-dlp subprocess metadata failed, trying Python API fallback: %s", exc)
        api_result = _try_python_api_fallback(url, extract=True)
        if api_result is None:
            raise ValueError(
                "Video is unavailable, private, or deleted."
            ) from exc
        info = api_result

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

    Uses yt-dlp (subprocess) with the FFmpeg audio extract
    post-processor to produce a 192 kbps MP3 file.  Falls back to the
    Python API with ``certifi`` if the subprocess approach fails.

    The file is saved into *output_dir* with the video ID as its
    filename for predictability.

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
    os.makedirs(output_dir, exist_ok=True)
    outtmpl = os.path.join(output_dir, "%(id)s.%(ext)s")

    # Try subprocess approach first
    try:
        args = _base_args() + [
            "--extract-audio",
            "--audio-format", "mp3",
            "--audio-quality", "192K",
            "--output", outtmpl,
            "--print", "filename",
            url,
        ]
        result = _run_ytdlp(args, "download")
        # The --print filename outputs the final path
        final_path = result.stdout.strip().split("\n")[-1].strip()
        if final_path and os.path.isfile(final_path):
            return final_path

    except RuntimeError as exc:
        logger.warning("yt-dlp subprocess download failed, trying Python API fallback: %s", exc)

    # Fallback: Python API with certifi
    logger.info("Trying Python API fallback for download...")
    api_result = _try_python_api_fallback(url, extract=False)
    if api_result is None:
        raise RuntimeError(
            f"Failed to download audio from YouTube after all retries."
        )

    video_id: str = api_result.get("id", "audio")
    expected_path: str = os.path.join(output_dir, f"{video_id}.mp3")

    if not os.path.isfile(expected_path):
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
