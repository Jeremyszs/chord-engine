"""YouTube audio download and validation for the chord-engine pipeline.

Provides URL validation, metadata fetching, and audio download services
using yt-dlp with browser impersonation to bypass bot detection on
cloud IPs (HF Spaces, etc.).

Integrates with the existing engine by producing an MP3 file that
``engine/loader.py`` can load normally.
"""

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

# Minimal CONSENT cookie to bypass YouTube's consent/bot wall.
# The "YES+cb" value is what YouTube sets when a user accepts.
_COOKIES_PATH: Final[str] = "/tmp/yt-cookies.txt"
_MINIMAL_COOKIES: Final[str] = (
    "# Netscape HTTP Cookie File\n"
    ".youtube.com\tTRUE\t/\tTRUE\t2147483647\tCONSENT\tYES+cb\n"
    ".youtube.com\tTRUE\t/\tTRUE\t2147483647\t__Secure-3PSID\t\n"
    ".youtube.com\tTRUE\t/\tTRUE\t2147483647\t__Secure-3PAPISID\t\n"
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
# Cookie file
# ---------------------------------------------------------------------------


def _ensure_cookies() -> str:
    """Write a minimal CONSENT cookie file if it doesn't exist yet."""
    if not os.path.isfile(_COOKIES_PATH):
        try:
            with open(_COOKIES_PATH, "w") as f:
                f.write(_MINIMAL_COOKIES)
        except OSError:
            pass
    return _COOKIES_PATH


# ---------------------------------------------------------------------------
# Python API extraction with curl_cffi impersonation
# ---------------------------------------------------------------------------


def _extract_with_curl_cffi(url: str, download: bool = False,
                            output_dir: str | None = None) -> dict | None:
    """Extract video info using yt-dlp's Python API with ``curl_cffi``.

    ``curl_cffi`` provides **browser-level TLS fingerprint impersonation**.
    yt-dlp will use it when available, making requests look like they
    come from a real Chrome browser rather than a Python script.

    Returns:
        Info dict on success, or ``None``.
    """
    try:
        # Importing curl_cffi registers it with yt-dlp's request system.
        import curl_cffi  # noqa: F401
        import yt_dlp

        _ensure_cookies()

        # curl_cffi-based options — yt-dlp auto-detects curl_cffi and uses
        # it for TLS fingerprint impersonation when ``--impersonate`` is set.
        ydl_opts: dict = {
            "quiet": True,
            "no_warnings": True,
            "cookiefile": _COOKIES_PATH,
            "extractor_args": {
                "youtube": {
                    "player_client": ["android", "android_music"],
                    "player_skip": ["webpage", "configs", "js"],
                }
            },
            "impersonate": "chrome",
            "legacy_server_connect": True,
            "socket_timeout": 60,
            "extractor_retries": 2,
            "geo_bypass": True,
        }

        if download:
            outdir = output_dir or os.environ.get("TMPDIR", "/tmp")
            os.makedirs(outdir, exist_ok=True)
            ydl_opts["format"] = "bestaudio/best"
            ydl_opts["outtmpl"] = os.path.join(outdir, "%(id)s.%(ext)s")
            ydl_opts["postprocessors"] = [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }]

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            return ydl.extract_info(url, download=download)

    except Exception as exc:
        logger.warning("curl_cffi extraction failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Subprocess extraction (fallback)
# ---------------------------------------------------------------------------


def _find_ytdlp() -> str:
    """Locate the ``yt-dlp`` executable on ``$PATH``."""
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
    raise RuntimeError("yt-dlp not found on PATH. Install with: pip install yt-dlp")


def _extract_subprocess(url: str, download: bool = False,
                        output_dir: str | None = None,
                        timeout: int = 180) -> dict | None:
    """Extract video info using yt-dlp CLI subprocess.

    Rotates through multiple player clients on each attempt.
    """
    ytdlp = _find_ytdlp()
    _ensure_cookies()

    player_clients = ["ios", "android", "android_music"]

    last_stderr = ""
    for attempt in range(1, _MAX_RETRIES + 1):
        client = player_clients[(attempt - 1) % len(player_clients)]
        cmd = [
            ytdlp,
            "--quiet", "--no-warnings",
            "--cookies", _COOKIES_PATH,
            "--force-ipv4",
            "--socket-timeout", "30",
            "--extractor-args", f"youtube:player_client={client}",
            "--extractor-args", "youtube:player_skip=webpage,configs,js",
            "--geo-bypass",
            "--impersonate", "chrome",
        ]

        if download and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            outtmpl = os.path.join(output_dir, "%(id)s.%(ext)s")
            cmd += [
                "--extract-audio", "--audio-format", "mp3",
                "--audio-quality", "192K",
                "--output", outtmpl,
                url,
            ]
        else:
            cmd += ["--dump-json", "--no-download", url]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout,
            )
            if result.returncode == 0:
                if download:
                    if output_dir and os.path.isdir(output_dir):
                        for fname in os.listdir(output_dir):
                            if fname.endswith(".mp3"):
                                return {"_path": os.path.join(output_dir, fname)}
                    return {"_path": ""}
                if result.stdout.strip():
                    import json
                    return json.loads(result.stdout)

            last_stderr = result.stderr.strip()[:500]
            logger.warning(
                "yt-dlp attempt %d/%d (client=%s) rc=%d: %s",
                attempt, _MAX_RETRIES, client,
                result.returncode, last_stderr,
            )

            if "private" in last_stderr.lower() or "deleted" in last_stderr.lower():
                break
            if attempt < _MAX_RETRIES:
                time.sleep(_RETRY_DELAY_SECONDS * attempt)

        except subprocess.TimeoutExpired:
            logger.warning("yt-dlp attempt %d/%d (client=%s) timed out", attempt, _MAX_RETRIES, client)
            if attempt < _MAX_RETRIES:
                time.sleep(_RETRY_DELAY_SECONDS * attempt)

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_video_metadata(url: str) -> dict:
    """Fetch metadata for a YouTube video **without** downloading it.

    Uses yt-dlp with ``curl_cffi`` for browser TLS impersonation,
    with a CONSENT cookie to bypass bot detection.  Falls back to
    subprocess if the Python API fails.

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
    # Strategy 1: Python API with curl_cffi TLS impersonation
    info = _extract_with_curl_cffi(url, download=False)

    # Strategy 2: subprocess as fallback
    if info is None:
        logger.warning("curl_cffi approach failed, trying subprocess...")
        info = _extract_subprocess(url, download=False)

    if info is None:
        raise ValueError("Video is unavailable, private, or deleted.")

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


def download_audio(url: str, output_dir: str) -> str:
    """Download audio from a YouTube video and convert it to MP3.

    Uses yt-dlp with ``curl_cffi`` for browser TLS impersonation.
    Falls back to subprocess if the Python API fails.

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

    # Strategy 1: Python API with curl_cffi impersonation
    info = _extract_with_curl_cffi(url, download=True, output_dir=output_dir)
    if info is not None:
        video_id = info.get("id", "audio")
        expected = os.path.join(output_dir, f"{video_id}.mp3")
        if os.path.isfile(expected):
            return expected
        # Scan for any mp3
        for fname in os.listdir(output_dir):
            if fname.endswith(".mp3"):
                return os.path.join(output_dir, fname)

    # Strategy 2: subprocess download (if Python API failed)
    logger.warning("curl_cffi download failed, trying subprocess...")
    result = _extract_subprocess(url, download=True, output_dir=output_dir, timeout=300)
    if result and os.path.isdir(output_dir):
        for fname in os.listdir(output_dir):
            if fname.endswith(".mp3"):
                return os.path.join(output_dir, fname)

    raise RuntimeError(
        "Failed to download audio from YouTube after all retries. "
        "YouTube may be blocking this server's IP range."
    )
