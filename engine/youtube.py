"""YouTube URL validation and metadata lookup for the chord-engine.

Provides URL validation and public video metadata lookup using
YouTube's **oEmbed API** (no authentication, no yt-dlp, no downloads).

Audio download happens **client-side** — the server returns an exact
yt-dlp command for the user to run locally, then upload the resulting
MP3 via POST /api/v1/jobs.
"""

import json
import logging
import re
import urllib.request
from typing import Final

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
    is accessible; for that, use :func:`get_video_info`.

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
# Metadata lookup (YouTube oEmbed API)
# ---------------------------------------------------------------------------


def get_video_info(url: str) -> dict:
    """Return basic video info using the public YouTube oEmbed API.

    This is a **public, unauthenticated** API that works from any IP
    address.  No yt-dlp, no cookies, no downloads.  If the video is
    private or unavailable the call will fail and a ``ValueError`` is
    raised.

    Args:
        url: A valid YouTube URL.

    Returns:
        A dictionary with keys:

        * **title** — video title.
        * **author** — channel / uploader name.
        * **thumbnail** — URL of the video thumbnail.
        * **video_id** — the 11-character YouTube video ID.
        * **url** — canonical ``youtube.com/watch?v=`` link.
        * **download_command** — exact ``yt-dlp`` command a user can
          run locally to download the audio as MP3.

    Raises:
        ValueError: if the URL is invalid or the video is unavailable.
    """
    if not validate_youtube_url(url):
        raise ValueError("Invalid YouTube URL.")

    video_id = get_video_id(url)
    if not video_id:
        raise ValueError("Could not extract video ID from URL.")

    oembed_url = (
        f"https://www.youtube.com/oembed"
        f"?url=https://www.youtube.com/watch?v={video_id}"
        f"&format=json"
    )

    try:
        with urllib.request.urlopen(oembed_url, timeout=10) as resp:
            data = json.loads(resp.read().decode())
    except Exception as exc:
        raise ValueError(
            f"Could not fetch video info. "
            f"The video may be private or unavailable. ({exc})"
        ) from exc

    download_command = (
        f'yt-dlp -x --audio-format mp3 '
        f'"https://www.youtube.com/watch?v={video_id}"'
    )

    return {
        "title": data.get("title", "Unknown"),
        "author": data.get("author_name", "Unknown"),
        "thumbnail": f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg",
        "video_id": video_id,
        "url": f"https://www.youtube.com/watch?v={video_id}",
        "download_command": download_command,
    }
