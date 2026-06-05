"""YouTube audio download and validation for the chord-engine pipeline.

Provides URL validation, metadata fetching, and audio download services.
Integrates with the existing engine by producing an MP3 file that
``engine/loader.py`` can load normally.

The primary download mechanism is **pytubefix** (a maintained fork of
pytube that fixes YouTube API changes).  **yt-dlp** serves as a fallback
if pytubefix fails.
"""

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
_RETRY_DELAY: Final[float] = 2.0

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
# pytubefix helper
# ---------------------------------------------------------------------------


def _extract_pytubefix(url: str, output_dir: str | None = None) -> dict | None:
    """Extract metadata (and optionally download audio) using pytubefix.

    pytubefix uses a completely different implementation from yt-dlp —
    it directly mimics the YouTube web client's cipher and API calls,
    which often works when yt-dlp gets blocked.

    Args:
        url: YouTube video URL.
        output_dir: If set, also download audio as MP4 and convert.

    Returns:
        Info dict (title, duration, uploader, thumbnail, and ``_path``
        if downloaded), or ``None`` on failure.
    """
    try:
        from pytubefix import YouTube
        from pytubefix.cli import on_progress

        result: dict = {}

        def _build_result(yt: "YouTube") -> dict:
            return {
                "title": yt.title or "Unknown",
                "duration": int(getattr(yt, "length", 0) or 0),
                "uploader": yt.author or "Unknown",
                "thumbnail": yt.thumbnail_url or "",
                "id": yt.video_id,
            }

        if output_dir:
            # Download audio stream
            yt = YouTube(url, on_progress_callback=on_progress, use_oauth=False, allow_oauth_cache=False)
            result = _build_result(yt)

            # Get the audio-only stream with the highest bitrate
            audio_stream = yt.streams.get_audio_only()
            if not audio_stream:
                # Fallback: get any progressive stream and we'll convert
                audio_stream = yt.streams.filter(only_audio=True).first()
            if not audio_stream:
                return None

            dl_path = audio_stream.download(output_path=output_dir)
            result["_path"] = dl_path
        else:
            # Metadata only
            yt = YouTube(url, use_oauth=False, allow_oauth_cache=False)
            result = _build_result(yt)

        return result

    except Exception as exc:
        logger.debug("pytubefix extraction failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# yt-dlp fallback
# ---------------------------------------------------------------------------


def _extract_ytdlp(url: str, download: bool = False,
                   output_dir: str | None = None) -> dict | None:
    """Fallback extraction using yt-dlp as a subprocess.

    Only called when pytubefix fails.  Rotates through iOS, Android,
    and web player clients across retry attempts.
    """
    import json
    import subprocess
    import sys

    def _find() -> str:
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
        raise RuntimeError("yt-dlp not found")

    def _client_for(attempt: int) -> str:
        return ["ios", "android,android_music", "web"][(attempt - 1) % 3]

    ytdlp = _find()
    for attempt in range(1, _MAX_RETRIES + 1):
        client = _client_for(attempt)
        cmd = [
            ytdlp,
            "--quiet", "--no-warnings",
            "--force-ipv4",
            "--socket-timeout", "30",
            "--extractor-args", f"youtube:player_client={client}",
            "--geo-bypass",
        ]
        if client != "web":
            cmd += ["--extractor-args", "youtube:player_skip=webpage,configs,js"]

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
                cmd, capture_output=True, text=True,
                timeout=600 if download else 180,
            )
            if result.returncode == 0:
                if download and output_dir:
                    for fname in os.listdir(output_dir):
                        if fname.endswith(".mp3"):
                            return {"_path": os.path.join(output_dir, fname)}
                    return {"_path": ""}
                if result.stdout.strip():
                    info = json.loads(result.stdout)
                    return info

            stderr_snippet = result.stderr.strip()[:500]
            logger.warning(
                "yt-dlp attempt %d/%d (client=%s) rc=%d: %s",
                attempt, _MAX_RETRIES, client,
                result.returncode, stderr_snippet,
            )
            if "private" in stderr_snippet.lower() or "deleted" in stderr_snippet.lower():
                break
            if attempt < _MAX_RETRIES:
                time.sleep(_RETRY_DELAY * attempt)

        except subprocess.TimeoutExpired:
            logger.warning("yt-dlp attempt %d/%d timed out", attempt, _MAX_RETRIES)
            if attempt < _MAX_RETRIES:
                time.sleep(_RETRY_DELAY * attempt)

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_video_metadata(url: str) -> dict:
    """Fetch metadata for a YouTube video **without** downloading it.

    Primary method: pytubefix (YouTube web client cipher).
    Fallback: yt-dlp subprocess with multiple player clients.

    Args:
        url: A valid YouTube URL.

    Returns:
        A dictionary with title, duration, uploader, and thumbnail.

    Raises:
        ValueError: Video is unavailable/private/deleted or too long.
    """
    # Try pytubefix first
    info = _extract_pytubefix(url, output_dir=None)

    # Fallback to yt-dlp
    if info is None:
        logger.warning("pytubefix failed, trying yt-dlp fallback...")
        info = _extract_ytdlp(url, download=False)

    if info is None:
        raise ValueError("Video is unavailable, private, or deleted.")

    duration: int = info.get("duration", 0)
    if duration > _MAX_DURATION_SECONDS:
        raise ValueError(
            f"Video too long. Maximum duration is 10 minutes "
            f"(got {duration} seconds)."
        )

    return {
        "title": info.get("title", "Unknown"),
        "duration": duration or 0,
        "uploader": info.get("uploader", info.get("author", "Unknown")),
        "thumbnail": info.get("thumbnail", info.get("thumbnail_url", "")),
    }


def download_audio(url: str, output_dir: str) -> str:
    """Download audio from a YouTube video and convert it to MP3.

    Primary method: pytubefix (downloads audio stream).
    Fallback: yt-dlp subprocess with FFmpeg conversion.

    Args:
        url: A valid YouTube URL.
        output_dir: Writable directory for the output file.

    Returns:
        Absolute path to the downloaded MP3 file.

    Raises:
        RuntimeError: If download fails after all retries.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Strategy 1: pytubefix
    info = _extract_pytubefix(url, output_dir=output_dir)

    if info is not None and info.get("_path"):
        dl_path = info["_path"]
        # pytubefix downloads in the original format (usually .mp4 or .webm).
        # We need to convert to MP3 using FFmpeg.
        if not dl_path.endswith(".mp3"):
            import subprocess
            mp3_path = os.path.join(output_dir, f"{info.get('id', 'audio')}.mp3")
            try:
                subprocess.run(
                    ["ffmpeg", "-i", dl_path, "-vn",
                     "-acodec", "libmp3lame", "-ab", "192k",
                     "-y", mp3_path],
                    capture_output=True, text=True, timeout=120,
                )
                if os.path.isfile(mp3_path):
                    # Remove the original non-mp3 file
                    try:
                        os.remove(dl_path)
                    except OSError:
                        pass
                    return mp3_path
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
                logger.warning("FFmpeg conversion failed: %s", exc)

        # If it's already mp3 or conversion failed, return what we have
        if os.path.isfile(dl_path):
            return dl_path

    # Strategy 2: yt-dlp fallback
    logger.warning("pytubefix download failed, trying yt-dlp fallback...")
    result = _extract_ytdlp(url, download=True, output_dir=output_dir)
    if result and result.get("_path"):
        path = result["_path"]
        if path and os.path.isfile(path):
            return path
        # Maybe it's still being processed — scan the dir
        for fname in os.listdir(output_dir):
            if fname.endswith(".mp3"):
                return os.path.join(output_dir, fname)

    raise RuntimeError(
        "Failed to download audio from YouTube after all retries."
    )
