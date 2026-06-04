---
title: Chord Engine
emoji: 🎸
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
---

# Chord Engine API

Async REST API for audio chord recognition.
Detects chords, key, tempo, and chord progression from any audio file.

## Endpoints

| Method | URL | Description |
|--------|-----|-------------|
| POST | `/api/v1/jobs` | Upload audio, returns job_id |
| GET | `/api/v1/jobs/{id}/status` | Poll progress |
| GET | `/api/v1/jobs/{id}/result` | Fetch result |
| GET | `/api/v1/health` | Health check |
| GET | `/api/v1/health/ping` | Liveness ping |
| GET | `/api/docs` | Swagger UI |

## Usage

Upload an audio file and poll for the result:

```bash
# Upload
curl -X POST https://YOUR_USERNAME-chord-engine.hf.space/api/v1/jobs \
  -F "audio=@song.mp3"

# Poll status
curl https://YOUR_USERNAME-chord-engine.hf.space/api/v1/jobs/{job_id}/status

# Get result
curl https://YOUR_USERNAME-chord-engine.hf.space/api/v1/jobs/{job_id}/result
```
