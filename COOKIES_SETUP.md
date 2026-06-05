# YouTube Integration

The chord engine does not download from YouTube directly.
YouTube blocks server IP ranges which makes server-side downloading
unreliable.

## How it works

1. The frontend calls `GET /api/v1/youtube/info?url=<youtube_url>`
   to fetch the video title and thumbnail.

2. The user downloads the audio locally using yt-dlp:

   ```bash
   yt-dlp -x --audio-format mp3 "https://youtube.com/watch?v=..."
   ```

3. The user uploads the downloaded MP3 via `POST /api/v1/jobs`.

## Why not server-side download?

YouTube actively blocks datacenter IP ranges (including Hugging Face,
Google Cloud, AWS) at the SSL level. No amount of cookies, proxies,
or format flags resolves this reliably.
