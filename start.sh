#!/bin/bash
set -e

# If YOUTUBE_COOKIES secret is set (Hugging Face Spaces), write it to
# /tmp/cookies.txt and set the env var so yt-dlp can use it.
if [ -n "$YOUTUBE_COOKIES" ]; then
    echo "$YOUTUBE_COOKIES" > /tmp/cookies.txt
    export YOUTUBE_COOKIES_PATH=/tmp/cookies.txt
    echo "YouTube cookies loaded from secret."
else
    echo "Warning: No YOUTUBE_COOKIES secret set."
    echo "YouTube downloads will likely fail on HF Spaces."
    echo "See COOKIES_SETUP.md for instructions."
fi

# Start the API
exec uvicorn api.main:app --host 0.0.0.0 --port "${PORT:-7860}"
