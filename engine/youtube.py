"""YouTube audio download and validation for the chord-engine pipeline.

Provides URL validation, metadata fetching, and audio download services
using yt-dlp via an Invidious proxy to bypass IP-based blocking on cloud
environments (HF Spaces, etc.).

How it works:
  1. Uses an Invidious instance as a proxy/relay for both metadata
     extraction and audio download.
  2. Falls back to direct yt-dlp (mobile clients) if the proxy fails.
  3. Falls back to pytubefix as a final resort.
"""

import json
import logging
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Final

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_DURATION_SECONDS: Final[int] = 600  # 10 minutes
_MAX_RETRIES: Final[int] = 2
_RETRY_DELAY: Final[float] = 2.0

# Regex that matches all supported YouTube URL formats.
_YOUTUBE_RE: Final[re.Pattern] = re.compile(
    r"^https?://"
    r"(?:www\.|music\.)?"
    r"(?:youtube\.com/(?:watch\?v=|shorts/)|youtu\.be/)"
    r"([a-zA-Z0-9_-]{11})"
    r"(?:[?&#]\S*)?$"
)

# Public Invidious instances to try as proxies.
# Invidious is a privacy-focused YouTube frontend with a REST API.
# See https://api.invidious.io/ for the full list.
_INVIDIOUS_INSTANCES: Final[list[str]] = [
    "https://inv.nadeko.net",
    "https://yewtu.be",
    "https://inv.vern.cc",
    "https://inv.riverside.rocks",
    "https://invidious.snopyta.org",
    "https://invidious.jing.rocks",
    "https://invidious.private.coffee",
]

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
# URL helpers
# ---------------------------------------------------------------------------


def _extract_video_id(url: str) -> str | None:
    """Extract the 11-character YouTube video ID from a URL."""
    m = _YOUTUBE_RE.match(url.strip())
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Invidious proxy — bypasses IP-based YouTube blocking
# ---------------------------------------------------------------------------


def _inv_request(instance: str, path: str, timeout: int = 15) -> dict | None:
    """Make a GET request to an Invidious instance and return JSON.

    Args:
        instance: Base URL of the invidious instance (e.g. ``https://inv.nadeko.net``).
        path: API path (e.g. ``/api/v1/videos/dQw4w9WgXcQ``).
        timeout: Request timeout in seconds.

    Returns:
        Parsed JSON dict, or ``None`` on failure.
    """
    url = f"{instance.rstrip('/')}/{path.lstrip('/')}"
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except (urllib.error.URLError, urllib.error.HTTPError,
            json.JSONDecodeError, OSError, TimeoutError) as exc:
        logger.debug("Invidious request failed (%s): %s", url, exc)
        return None


def _pick_working_instance() -> str | None:
    """Find an Invidious instance that responds to the health endpoint.

    Tries each known instance and returns the first one that works.
    """
    import random
    instances = list(_INVIDIOUS_INSTANCES)
    random.shuffle(instances)

    for instance in instances:
        result = _inv_request(instance, "/api/v1/stats", timeout=8)
        if result is not None:
            logger.debug("Using Invidious instance: %s", instance)
            return instance
    return None


def _extract_via_invidious(video_id: str, instance: str) -> dict | None:
    """Fetch video info from an Invidious instance.

    Args:
        video_id: 11-character YouTube video ID.
        instance: Base URL of a working Invidious instance.

    Returns:
        Info dict with title, duration, author, thumbnailUrl, or None.
    """
    return _inv_request(instance, f"/api/v1/videos/{video_id}", timeout=20)


def _download_via_invidious(video_id: str, instance: str,
                            output_dir: str) -> str | None:
    """Download audio from an Invidious instance.

    Invidious provides direct audio stream URLs in its video info.
    We pick the best audio-only format and download it, then convert
    to MP3 via FFmpeg.

    Args:
        video_id: 11-character YouTube video ID.
        instance: Base URL of a working Invidious instance.
        output_dir: Directory to save the MP3 file.

    Returns:
        Path to the downloaded MP3 file, or None.
    """
    info = _extract_via_invidious(video_id, instance)
    if info is None:
        return None

    # Find the best audio-only stream URL
    audio_streams = info.get("adaptiveFormats", [])
    if not audio_streams:
        logger.warning("No adaptive formats found in Invidious response")
        return None

    # Sort by bitrate (descending), pick audio-only with highest bitrate
    audio_only = [
        s for s in audio_streams
        if s.get("type", "").startswith("audio/")
    ]
    if not audio_only:
        audio_only = audio_streams

    audio_only.sort(key=lambda s: s.get("bitrate", 0), reverse=True)
    best = audio_only[0]
    stream_url: str = best.get("url", "")
    if not stream_url:
        return None

    # Download the audio stream
    import subprocess
    raw_path = os.path.join(output_dir, f"{video_id}.webm")
    mp3_path = os.path.join(output_dir, f"{video_id}.mp3")

    try:
        urllib.request.urlretrieve(stream_url, raw_path)
    except Exception as exc:
        logger.warning("Invidious stream download failed: %s", exc)
        return None

    # Convert to MP3 via FFmpeg
    if not os.path.isfile(raw_path):
        return None

    try:
        subprocess.run(
            ["ffmpeg", "-i", raw_path, "-vn",
             "-acodec", "libmp3lame", "-ab", "192k",
             "-y", mp3_path],
            capture_output=True, text=True, timeout=120,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        logger.warning("FFmpeg conversion failed: %s", exc)
        # Return raw file if conversion fails (might still work)
        if os.path.isfile(raw_path):
            return raw_path
        return None

    # Clean up the raw file
    try:
        os.remove(raw_path)
    except OSError:
        pass

    if os.path.isfile(mp3_path):
        return mp3_path
    return None


# ---------------------------------------------------------------------------
# yt-dlp fallback (subprocess)
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


def _extract_ytdlp(url: str, download: bool = False,
                   output_dir: str | None = None) -> dict | None:
    """Fallback extraction using yt-dlp as a subprocess."""
    ytdlp = _find_ytdlp()
    for attempt in range(1, _MAX_RETRIES + 1):
        client = ["ios", "android,android_music", "web"][(attempt - 1) % 3]
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
                    return json.loads(result.stdout)

            stderr_snippet = result.stderr.strip()[:300]
            logger.debug("yt-dlp attempt %d/%d (client=%s) failed: %s",
                         attempt, _MAX_RETRIES, client, stderr_snippet)
            if attempt < _MAX_RETRIES:
                time.sleep(_RETRY_DELAY * attempt)
        except subprocess.TimeoutExpired:
            if attempt < _MAX_RETRIES:
                time.sleep(_RETRY_DELAY * attempt)
    return None


# ---------------------------------------------------------------------------
# pytubefix fallback
# ---------------------------------------------------------------------------


def _extract_pytubefix(url: str, output_dir: str | None = None) -> dict | None:
    """Fallback using pytubefix."""
    try:
        from pytubefix import YouTube
        from pytubefix.cli import on_progress

        yt = YouTube(url, on_progress_callback=on_progress,
                     use_oauth=False, allow_oauth_cache=False)
        result = {
            "title": yt.title or "Unknown",
            "duration": int(getattr(yt, "length", 0) or 0),
            "uploader": yt.author or "Unknown",
            "thumbnail": yt.thumbnail_url or "",
            "id": yt.video_id,
        }

        if output_dir:
            audio = yt.streams.get_audio_only()
            if not audio:
                audio = yt.streams.filter(only_audio=True).first()
            if audio:
                dl = audio.download(output_path=output_dir)
                result["_path"] = dl

        return result
    except Exception as exc:
        logger.debug("pytubefix failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_video_metadata(url: str) -> dict:
    """Fetch metadata for a YouTube video **without** downloading it.

    Strategy:
      1. Invidious proxy (bypasses IP block)
      2. yt-dlp subprocess (direct)
      3. pytubefix (last resort)

    Args:
        url: A valid YouTube URL.

    Returns:
        title, duration, uploader, thumbnail.

    Raises:
        ValueError: Video is unavailable/private/deleted or too long.
    """
    video_id = _extract_video_id(url)
    if not video_id:
        raise ValueError("Could not extract video ID from URL.")

    info: dict | None = None

    # Strategy 1: Invidious proxy
    logger.info("Trying Invidious proxy for metadata...")
    instance = _pick_working_instance()
    if instance:
        inv_info = _extract_via_invidious(video_id, instance)
        if inv_info:
            info = {
                "title": inv_info.get("title", "Unknown"),
                "duration": inv_info.get("lengthSeconds", 0),
                "uploader": inv_info.get("author", inv_info.get("authorId", "Unknown")),
                "thumbnail": inv_info.get("thumbnailUrl", ""),
            }

    # Strategy 2: yt-dlp
    if info is None:
        logger.info("Invidious failed, trying yt-dlp...")
        yt_info = _extract_ytdlp(url, download=False)
        if yt_info:
            info = {
                "title": yt_info.get("title", "Unknown"),
                "duration": yt_info.get("duration", 0),
                "uploader": yt_info.get("uploader", yt_info.get("channel", "Unknown")),
                "thumbnail": yt_info.get("thumbnail", ""),
            }

    # Strategy 3: pytubefix
    if info is None:
        logger.info("yt-dlp failed, trying pytubefix...")
        pt_info = _extract_pytubefix(url, output_dir=None)
        if pt_info:
            info = {
                "title": pt_info.get("title", "Unknown"),
                "duration": pt_info.get("duration", 0),
                "uploader": pt_info.get("uploader", pt_info.get("author", "Unknown")),
                "thumbnail": pt_info.get("thumbnail", pt_info.get("thumbnail_url", "")),
            }

    if info is None:
        raise ValueError("Video is unavailable, private, or deleted.")

    duration: int = info.get("duration", 0)
    if duration > _MAX_DURATION_SECONDS:
        raise ValueError(
            f"Video too long. Maximum duration is 10 minutes "
            f"(got {duration} seconds)."
        )

    return info


def download_audio(url: str, output_dir: str) -> str:
    """Download audio from a YouTube video and convert it to MP3.

    Strategy:
      1. Invidious proxy (bypasses IP block)
      2. yt-dlp subprocess
      3. pytubefix

    Args:
        url: A valid YouTube URL.
        output_dir: Writable directory for the output file.

    Returns:
        Absolute path to the downloaded MP3 file.

    Raises:
        RuntimeError: If download fails after all retries.
    """
    os.makedirs(output_dir, exist_ok=True)
    video_id = _extract_video_id(url)

    # Strategy 1: Invidious proxy
    if video_id:
        logger.info("Trying Invidious proxy for download...")
        instance = _pick_working_instance()
        if instance:
            path = _download_via_invidious(video_id, instance, output_dir)
            if path and os.path.isfile(path):
                return path

    # Strategy 2: yt-dlp
    logger.info("Invidious download failed, trying yt-dlp...")
    yt_result = _extract_ytdlp(url, download=True, output_dir=output_dir)
    if yt_result and yt_result.get("_path"):
        p = yt_result["_path"]
        if p and os.path.isfile(p):
            return p
    # Scan for mp3 if yt-dlp succeeded but returned empty path
    for fname in os.listdir(output_dir):
        if fname.endswith(".mp3"):
            return os.path.join(output_dir, fname)

    # Strategy 3: pytubefix
    logger.info("yt-dlp download failed, trying pytubefix...")
    pt_result = _extract_pytubefix(url, output_dir=output_dir)
    if pt_result and pt_result.get("_path"):
        dl_path = pt_result["_path"]
        if dl_path.endswith(".mp3"):
            return dl_path
        # Convert non-MP3 to MP3
        mp3_path = os.path.join(output_dir, f"{video_id or 'audio'}.mp3")
        try:
            subprocess.run(
                ["ffmpeg", "-i", dl_path, "-vn",
                 "-acodec", "libmp3lame", "-ab", "192k",
                 "-y", mp3_path],
                capture_output=True, text=True, timeout=120,
            )
            if os.path.isfile(mp3_path):
                try:
                    os.remove(dl_path)
                except OSError:
                    pass
                return mp3_path
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
            logger.warning("FFmpeg conversion failed: %s", exc)
            if os.path.isfile(dl_path):
                return dl_path

    raise RuntimeError(
        "Failed to download audio from YouTube after all retries."
    )
