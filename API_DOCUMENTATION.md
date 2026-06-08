# Chord Engine API Documentation

**Version:** 2.0 (Beat-Synchronized Timing)  
**Last Updated:** June 8, 2026

## Overview

Chord Engine is a production-ready REST API for automatic chord recognition from audio files. It uses ChordMiniApp's superior BTC-PL model with beat-synchronized timing for musically accurate chord boundaries.

### Key Features

- **State-of-the-art accuracy**: ChordMiniApp BTC-PL model (+2.37% MIREX improvement)
- **Beat-synchronized timing**: Chord changes align with musical beats (no mid-bar transitions)
- **170-chord vocabulary**: Complex chords including 7ths, diminished, augmented, suspended
- **Automatic key detection**: Identifies musical key with roman numeral analysis
- **Beat tracking**: Returns beat times and accurate tempo from audio
- **YouTube support**: Direct chord recognition from YouTube URLs

---

## Base URL

```
https://huggingface.co/spaces/Jeremyszs/chord-engine
```

---

## Endpoints

### 1. POST `/api/analyze`

Upload an audio file for chord recognition.

**Request:**

```http
POST /api/analyze
Content-Type: multipart/form-data

file: <audio_file>          # Required: MP3, WAV, FLAC, OGG, M4A
device: cpu|cuda            # Optional: Default "cpu"
smooth_method: hmm|median   # Optional: Default "hmm"
include_raw: true|false     # Optional: Default false
```

**Response:**

```json
{
  "job_id": "1a2b3c4d",
  "status": "completed",
  "audio_filename": "song.mp3",
  "duration_seconds": 180.5,
  "tempo_bpm": 120.4,
  "key": "C major",
  "progression": "I → IV → V → I",
  "chord_count": 12,
  "beats": [0.5, 1.0, 1.5, 2.0, ...],
  "segments": [
    {
      "chord": "C",
      "roman": "I",
      "start": 0.0,
      "end": 2.0,
      "duration": 2.0,
      "confidence": 0.95
    },
    ...
  ],
  "raw_chords": ["C", "C", "F", "F", ...],
  "processing_time_seconds": 15.3,
  "created_at": "2026-06-08T13:42:00Z"
}
```

---

### 2. POST `/api/analyze-youtube`

Analyze chord progression from a YouTube URL.

**Request:**

```http
POST /api/analyze-youtube
Content-Type: application/json

{
  "url": "https://youtube.com/watch?v=...",
  "device": "cpu",
  "smooth_method": "hmm",
  "include_raw": false
}
```

**Response:** Same format as `/api/analyze`

---

### 3. GET `/api/jobs/{job_id}`

Get the status and result of a chord recognition job.

**Request:**

```http
GET /api/jobs/1a2b3c4d
```

**Response:**

```json
{
  "job_id": "1a2b3c4d",
  "status": "completed",
  ...
}
```

Possible statuses: `pending`, `processing`, `completed`, `failed`

---

## Response Fields

### Core Fields

| Field | Type | Description |
|-------|------|-------------|
| `job_id` | string | Unique job identifier |
| `status` | string | Job status: pending, processing, completed, failed |
| `audio_filename` | string | Original audio filename |
| `duration_seconds` | number | Total audio duration |
| `tempo_bpm` | number | **NEW**: Tempo calculated from detected beats |
| `key` | string | Detected musical key (e.g., "C major", "G minor") |
| `progression` | string | Simplified chord progression pattern |
| `chord_count` | number | Number of unique chords detected |

### NEW: Beat Tracking Fields

| Field | Type | Description |
|-------|------|-------------|
| `beats` | array[number] | **NEW**: Beat times in seconds [0.5, 1.0, 1.5, ...] |

**What are beats?**  
Beat times represent the musical pulse of the song. Chord boundaries are now aligned to these beat positions, ensuring chord changes happen on musically meaningful moments (downbeats, strong beats) rather than arbitrary frame boundaries.

**Why it matters:**  
- Eliminates jittery, mid-bar chord transitions
- Chord changes align with the musical grid
- More accurate tempo calculation from actual beat intervals
- Enables beat-aware music applications (metronomes, click tracks, sync)

### Segment Fields

| Field | Type | Description |
|-------|------|-------------|
| `segments` | array | List of chord segments with timing |
| `segments[].chord` | string | Chord symbol (e.g., "C", "F", "G7", "Am") |
| `segments[].roman` | string | Roman numeral in detected key (e.g., "I", "IV", "V") |
| `segments[].start` | number | **BEAT-ALIGNED**: Segment start time in seconds |
| `segments[].end` | number | **BEAT-ALIGNED**: Segment end time in seconds |
| `segments[].duration` | number | Segment duration (end - start) |
| `segments[].confidence` | number | Model confidence score (0.0 - 1.0) |

**Beat Alignment:**  
`start` and `end` times are now snapped to the nearest detected beat position. This ensures chord boundaries occur at musically meaningful moments.

### Optional Fields

| Field | Type | Description |
|-------|------|-------------|
| `raw_chords` | array[string] | Frame-by-frame predictions (if include_raw=true) |
| `processing_time_seconds` | number | Total processing time |
| `created_at` | string | ISO 8601 timestamp |

---

## Example Request/Response

### cURL Example

```bash
curl -X POST https://huggingface.co/spaces/Jeremyszs/chord-engine/api/analyze \
  -F "file=@song.mp3" \
  -F "device=cpu" \
  -F "smooth_method=hmm"
```

### Python Example

```python
import requests

url = "https://huggingface.co/spaces/Jeremyszs/chord-engine/api/analyze"
files = {"file": open("song.mp3", "rb")}
data = {"device": "cpu", "smooth_method": "hmm"}

response = requests.post(url, files=files, data=data)
result = response.json()

print(f"Key: {result['key']}")
print(f"Tempo: {result['tempo_bpm']} BPM")
print(f"Beats detected: {len(result['beats'])}")

for segment in result['segments']:
    print(f"{segment['start']:.1f}s - {segment['chord']} ({segment['roman']})")
```

### JavaScript Example

```javascript
const formData = new FormData();
formData.append('file', audioFile);
formData.append('device', 'cpu');

const response = await fetch(
  'https://huggingface.co/spaces/Jeremyszs/chord-engine/api/analyze',
  { method: 'POST', body: formData }
);

const result = await response.json();
console.log(`Key: ${result.key}`);
console.log(`Tempo: ${result.tempo_bpm} BPM`);
console.log(`Beats: ${result.beats.length}`);
```

---

## What's New in v2.0

### Beat-Synchronized Timing

**The Problem:**  
Previous versions used frame-based timing (~93ms resolution), causing chord changes to occur at arbitrary points that didn't align with the music's beat structure. This created "jittery" transitions and mid-bar chord changes.

**The Solution:**  
Version 2.0 implements ChordMiniApp's beat synchronization approach:

1. **Beat Detection**: Uses librosa's beat tracker to detect the musical pulse
2. **Chord-to-Beat Sync**: Applies a two-pointer algorithm with 50% midpoint rule
3. **Boundary Snapping**: Chord boundaries snap to the nearest beat position
4. **Musical Alignment**: Ensures chord changes happen on musically meaningful moments

### API Changes Summary

| Change | Impact |
|--------|--------|
| **Added `beats` array** | New field containing beat times in seconds |
| **Beat-aligned timing** | `start`/`end` times now snap to beat positions |
| **Improved tempo** | Calculated from beat intervals (more accurate) |
| **Backward compatible** | Existing fields unchanged, new field is additive |

---

## Rate Limits

- **10 requests per minute** per IP address
- **Max file size**: 50 MB
- **Max duration**: 10 minutes of audio

---

## Error Responses

```json
{
  "error": "Error message",
  "status": "failed"
}
```

Common error codes:
- `400`: Bad request (invalid file format, missing parameters)
- `413`: File too large (>50 MB)
- `429`: Rate limit exceeded
- `500`: Internal server error

---

## Supported Audio Formats

- MP3
- WAV
- FLAC
- OGG
- M4A
- AAC

Sample rates: Any (automatically resampled to 22050 Hz)

---

## Technical Details

### Model Architecture

- **Chord Model**: ChordMiniApp BTC-PL (pseudo-labeling + knowledge distillation)
- **Beat Tracking**: Librosa beat tracker with onset strength
- **Vocabulary**: 170 chords (majors, minors, 7ths, dim, aug, sus, add, etc.)
- **Accuracy**: +2.37% MIREX improvement over baseline BTC

### Processing Pipeline

1. Audio loading and resampling (22050 Hz)
2. CQT feature extraction (144 bins, 2048 hop length)
3. Beat detection (librosa)
4. BTC-PL chord inference
5. HMM smoothing with music-theoretic transitions
6. Chord-to-beat synchronization
7. Key detection and roman numeral analysis

### Performance

- **Typical processing time**: 15-30 seconds for 3-minute song (CPU)
- **GPU acceleration**: 3-5x faster with CUDA
- **Memory usage**: ~2 GB RAM

---

## Support & Contact

- **GitHub**: https://github.com/Jeremyszs/chord-engine
- **Issues**: https://github.com/Jeremyszs/chord-engine/issues
- **Hugging Face Space**: https://huggingface.co/spaces/Jeremyszs/chord-engine
