---
title: Chord Engine
emoji: 🎸
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
---

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue?style=flat-square&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/framework-FastAPI-009688?style=flat-square&logo=fastapi" alt="FastAPI">
  <img src="https://img.shields.io/badge/model-BTC-FF6F00?style=flat-square&logo=pytorch" alt="BTC Model">
  <img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="License">
  <img src="https://img.shields.io/badge/status-production-brightgreen?style=flat-square" alt="Status">
</p>

<h1 align="center">🎸 Chord Engine</h1>

<p align="center">
  <strong>Audio chord recognition powered by a Bidirectional Transformer (BTC).</strong><br>
  Upload a song → get back timestamped chords, detected key, tempo, and Roman numeral progression.
</p>

<p align="center">
  <b>Live demo:</b>
  <a href="https://jeremyszs-chord-engine.hf.space/api/docs">
    🚀 jeremyszs-chord-engine.hf.space
  </a>
</p>

---

## 📋 Table of Contents

- [Features](#-features)
- [Quick Start](#-quick-start)
- [API Reference](#-api-reference)
- [Architecture](#️-architecture)
- [Installation (Local)](#-installation-local)
- [CLI Usage](#-cli-usage)
- [Evaluation](#-evaluation)
- [Deployment](#-deployment)
- [Technology Stack](#-technology-stack)
- [Project Structure](#-project-structure)
- [Roadmap](#-roadmap)

---

## ✨ Features

- **🎵 Chord Recognition** — detects major, minor, seventh chords with millisecond precision
- **🎹 Key Detection** — identifies the musical key using the Krumhansl-Schmuckler algorithm
- **📈 Tempo Estimation** — BPM detection via librosa's beat tracker
- **🔢 Roman Numerals** — chords mapped to functional harmony (I, IV, V, vi, etc.)
- **🔄 Chord Progressions** — extracts the most repeated chord loop from the song
- **⏱️ Async API** — non-blocking upload with progress polling
- **📊 Confidence Scores** — per-chord confidence from the BTC model
- **🎚️ Smoothing Options** — HMM Viterbi or median filter for cleaner results
- **🐳 Docker Ready** — deploy anywhere (Hugging Face Spaces, VPS, cloud)
- **📖 Swagger UI** — interactive API docs at `/api/docs`

---

## 🚀 Quick Start

### Using the Live API

```bash
# 1. Upload an audio file
curl -X POST https://jeremyszs-chord-engine.hf.space/api/v1/jobs \
  -F "audio=@your_song.mp3"

# 2. Poll for completion (replace <job_id>)
curl https://jeremyszs-chord-engine.hf.space/api/v1/jobs/<job_id>/status

# 3. Get the full analysis
curl https://jeremyszs-chord-engine.hf.space/api/v1/jobs/<job_id>/result
```

Or open the **Swagger UI**: [https://jeremyszs-chord-engine.hf.space/api/docs](https://jeremyszs-chord-engine.hf.space/api/docs)

### Run Locally

```bash
git clone https://github.com/Jeremyszs/chord-engine.git
cd chord-engine
python -m venv .venv
source .venv/bin/activate     # Linux/Mac — or .venv\Scripts\activate on Windows
pip install -r requirements.txt

# Start the API server
uvicorn api.main:app --reload --port 8000

# Or use the CLI directly
python main.py --audio path/to/song.mp3
```

---

## 📖 API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/jobs` | Upload audio file for analysis |
| `GET` | `/api/v1/jobs/{job_id}/status` | Poll analysis progress |
| `GET` | `/api/v1/jobs/{job_id}/result` | Fetch completed analysis |
| `GET` | `/api/v1/health` | Server health check |
| `GET` | `/api/v1/health/ping` | Lightweight liveness check |
| `GET` | `/api/docs` | Swagger UI documentation |

### Upload Flow

The API uses an **async job pattern** — you upload a file, get a `job_id`, then poll until it's done.

```
POST /api/v1/jobs  ──────→  { job_id, poll_url, result_url }
         │
         └──→ GET {poll_url} every 2-3s
                    │
                    ├── status: "processing" → keep polling (check progress %)
                    ├── status: "failed"     → show error
                    └── status: "completed"  → GET {result_url}
```

### Supported File Types

| Format | Extension |
|--------|-----------|
| MP3 | `.mp3` |
| WAV | `.wav` |
| FLAC | `.flac` |
| OGG | `.ogg` |
| M4A | `.m4a` |

**Max file size:** 50 MB

### Request Parameters

Upload audio as `multipart/form-data` with these optional query parameters:

| Parameter | Default | Options | Description |
|-----------|---------|---------|-------------|
| `smooth_method` | `hmm` | `hmm`, `median` | Chord smoothing algorithm |
| `device` | `cpu` | `cpu`, `cuda` | Inference device |
| `include_raw_chords` | `false` | `true`, `false` | Include per-frame predictions |

### Example Response (200 — completed)

```json
{
  "job_id": "a1b2c3d4e5f6789012345678abcdef01",
  "status": "completed",
  "audio_filename": "song.mp3",
  "duration_seconds": 245.78,
  "tempo_bpm": 120.5,
  "key": "C major",
  "progression": "I → V → vi → IV",
  "chord_count": 4,
  "segments": [
    {
      "chord": "C:maj",
      "roman": "I",
      "start": 0.0,
      "end": 4.032,
      "duration": 4.032,
      "confidence": 0.913
    },
    {
      "chord": "G:maj",
      "roman": "V",
      "start": 4.032,
      "end": 8.475,
      "duration": 4.443,
      "confidence": 0.884
    },
    {
      "chord": "A:min",
      "roman": "vi",
      "start": 8.475,
      "end": 12.507,
      "duration": 4.032,
      "confidence": 0.952
    },
    {
      "chord": "F:maj",
      "roman": "IV",
      "start": 12.507,
      "end": 17.250,
      "duration": 4.743,
      "confidence": 0.876
    }
  ],
  "raw_chords": null,
  "processing_time_seconds": 2.34,
  "created_at": "2026-06-04T10:23:01.482Z"
}
```

### Chord Label Notation

| Label | Meaning |
|-------|---------|
| `C:maj` | C major |
| `A:min` | A minor |
| `G:7` | G dominant seventh |
| `F:maj7` | F major seventh |
| `N` | No chord (silence) |
| `X` | Percussion / noise |

### Error Codes

| Status | Error Code | Description |
|--------|------------|-------------|
| `400` | `unsupported_format` | File extension not supported |
| `400` | `invalid_file` | Empty file or missing filename |
| `404` | `job_not_found` | Job ID does not exist |
| `409` | `job_not_complete` | Job still processing (poll again) |
| `413` | `file_too_large` | Exceeds 50 MB limit |
| `422` | `processing_failed` | Analysis pipeline error |
| `422` | `validation_error` | Invalid request parameters |

---

## 🏗️ Architecture

The analysis pipeline runs in five stages, each handled by a dedicated module:

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌────────────┐
│   Audio     │    │  Features    │    │  BTC Model   │    │  Post-       │    │  Output    │
│   Loading   │───▶│  Extraction  │───▶│  Inference   │───▶│  Processing  │───▶│  Formatting│
│             │    │              │    │              │    │              │    │            │
│ loader.py   │    │ features.py  │    │ model.py     │    │ postprocess  │    │ output.py  │
└─────────────┘    └──────────────┘    └──────────────┘    └──────────────┘    └────────────┘
       │                  │                   │                   │                   │
       │ Mono            │ CQT Chroma        │ Pretrained        │ HMM/Median       │ JSON +     │
       │ 22050 Hz        │ HPSS              │ Transformer       │ Key Inference    │ Tempo      │
       │ Silence Trim    │ Beat Sync         │ 170/25 vocab      │ Roman Numerals   │ Timestamps │
       │ Normalize       │                   │                   │ Progression      │            │
       ▼                  ▼                   ▼                   ▼                   ▼
```

### Pipeline Stages

| Stage | Module | What It Does |
|-------|--------|-------------|
| **1. Audio Loading** | `engine/loader.py` | Loads audio via librosa, converts to mono, resamples to 22050 Hz, trims silence, normalizes amplitude |
| **2. Feature Extraction** | `engine/features.py` | Computes 12-bin chroma CQT features with optional HPSS (harmonic-percussive separation) |
| **3. Chord Prediction** | `engine/model.py` | Runs the BTC (Bidirectional Transformer) model — internally computes 144-bin log-CQT, outputs frame-level chord labels |
| **4. Post-processing** | `engine/postprocess.py` | Smooths predictions (HMM Viterbi or median filter), merges consecutive identical chords into segments, infers musical key via Krumhansl-Schmuckler, adds Roman numeral analysis |
| **5. Output** | `engine/output.py` | Scales segment durations, detects tempo, formats timestamps, builds the final JSON response |

### API Layer

```
HTTP Request
     │
     ▼
┌─────────────────────┐
│  FastAPI Server      │
│  (api/main.py)      │
│                     │
│  ┌───────────────┐  │
│  │ Routes         │  │
│  │ ┌───────────┐  │  │
│  │ │ Health    │  │  │
│  │ │ Jobs      │──│──│──▶ Upload → Background Task
│  │ └───────────┘  │  │        │
│  └───────────────┘  │        ▼
│                     │  ┌─────────────┐
│  ┌───────────────┐  │  │  Pipeline   │
│  │ Services      │  │  │  (Thread    │
│  │ ┌───────────┐ │  │  │   Pool)     │
│  │ │ JobStore  │◀─│──│──│            │
│  │ │ Pipeline  │ │  │  │ load_audio │
│  │ └───────────┘ │  │  │ extract_* │
│  └───────────────┘  │  │ predict   │
│                     │  │ smooth    │
│  ┌───────────────┐  │  │ output    │
│  │ Middleware     │  │  └─────────────┘
│  │ (Error        │  │
│  │  Handling)    │  │
│  └───────────────┘  │
└─────────────────────┘
```

---

## 💻 Installation (Local)

### Prerequisites

- **Python 3.10+**
- **~2 GB free disk space** (includes PyTorch + BTC model weights)
- **Linux / macOS / Windows**
- CUDA-capable GPU optional (falls back to CPU)

### Setup

```bash
# Clone the repository
git clone https://github.com/Jeremyszs/chord-engine.git
cd chord-engine

# Create virtual environment
python -m venv .venv

# Activate it
# Linux/macOS:
source .venv/bin/activate
# Windows:
source .venv/Scripts/activate

# Install dependencies
pip install -r requirements.txt
```

### Download Model Weights

The BTC model checkpoint will **auto-download** on first use. If you want to download it ahead of time:

```bash
# The weights are automatically fetched from:
# https://raw.githubusercontent.com/jayg996/BTC-ISMIR19/master/test/btc_model.pt
# This happens on first API call — no manual step needed.
```

---

## 🔧 CLI Usage

```bash
# Analyze an audio file
python main.py --audio path/to/song.mp3

# Save results to a custom path
python main.py --audio path/to/song.mp3 --output my_results.json

# Verbose output
python main.py --audio path/to/song.mp3 --verbose

# Use GPU (if available)
python main.py --audio path/to/song.wav --device cuda

# Change smoothing method
python main.py --audio path/to/song.wav --smooth median
```

### CLI Output

```
$ python main.py --audio samples/demo_chords.wav

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Chord Engine Analysis Results
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

 File:          samples/demo_chords.wav
 Duration:      12.51s
 Tempo:         120.0 BPM
 Detected Key:  C major
 Progression:   I → V → vi → IV

 Segments:
  0.000  -  4.032    C:maj  (I)         ████████████████  0.91
  4.032  -  8.475    G:maj  (V)         ████████████████  0.88
  8.475  - 12.507    A:min  (vi)        ████████████████  0.95
 12.507  - 17.250    F:maj  (IV)        ████████████████  0.88
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 📊 Evaluation

The project includes `eval.py` for benchmarking against ground-truth chord annotations using `mir_eval`.

```bash
# Evaluate against a .lab file
python eval.py --audio samples/eval_test.wav --ref samples/eval_test.lab
```

This computes the **MIREX chord recognition metric** alongside component-level scores:

| Metric | What It Measures |
|--------|-----------------|
| **Root** | Root note correct |
| **Thirds** | Root + quality (major/minor) correct |
| **Triads** | Root + quality + triad correct |
| **Sevenths** | Full seventh chord correct |
| **MIREX** | Weighted composite (primary benchmark score) |

### Running Tests

```bash
# Engine tests
pytest tests/ -v

# API tests
pytest api/tests/ -v

# All tests
pytest
```

---

## 🐳 Deployment

### Hugging Face Spaces (Current)

The project is configured for [Hugging Face Spaces](https://huggingface.co/spaces) with Docker SDK:

```bash
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/chord-engine
git push hf master:main
```

HF Spaces will automatically build the Docker image and start the server.

### Docker

```bash
# Build the image
docker build -t chord-engine .

# Run
docker run -p 7860:7860 chord-engine
```

### Custom Server

```bash
# Production-ready with gunicorn
gunicorn -k uvicorn.workers.UvicornWorker api.main:app --bind 0.0.0.0:8000 --workers 4
```

---

## 🛠️ Technology Stack

| Technology | Purpose |
|------------|---------|
| **Python 3.10+** | Core language |
| **FastAPI** | REST API framework |
| **PyTorch** | BTC model inference |
| **Librosa** | Audio processing, feature extraction |
| **BTC (Transformer)** | State-of-the-art chord recognition model |
| **HMM** | Chord smoothing via Viterbi decoding |
| **Uvicorn** | ASGI server |
| **Docker** | Containerization |
| **Hugging Face Spaces** | Hosting platform |
| **Pydantic** | API request/response validation |
| **pytest** | Testing |

---

## 📁 Project Structure

```
chord-engine/
├── api/                    # FastAPI REST API layer
│   ├── main.py             # App entry point, lifespan, CORS
│   ├── models/             # Pydantic schemas (request/response)
│   ├── routes/             # Endpoint routers (health, jobs)
│   ├── services/           # Business logic (job store, pipeline)
│   ├── middleware/          # Error handling middleware
│   └── tests/              # API tests (64 tests)
├── engine/                 # Core audio analysis engine
│   ├── loader.py           # Audio file loading & preprocessing
│   ├── features.py         # Chroma CQT extraction, beat sync
│   ├── model.py            # BTC model wrapper & inference
│   ├── postprocess.py      # Smoothing, key detection, roman numerals
│   └── output.py           # Result formatting & tempo detection
├── btc_model/              # BTC model source code & checkpoint
│   ├── btc_model.py        # BTC architecture (PyTorch)
│   ├── utils/              # Model utilities (vocab, hparams)
│   └── test/               # Model weights (auto-downloaded)
├── tests/                  # Engine unit tests (13 tests)
├── samples/                # Demo audio files for testing
├── config.py               # Centralized configuration
├── main.py                 # CLI entry point
├── Dockerfile              # Production container
├── requirements.txt        # Python dependencies
└── CLAUDE.md               # AI assistant instructions
```

---

## 🗺️ Roadmap

- [ ] **Database backend** — persistent job storage (Redis/PostgreSQL)
- [ ] **User auth** — API key authentication
- [ ] **Batch processing** — analyze multiple files at once
- [ ] **Webhook callbacks** — notified when analysis completes
- [ ] **Real-time streaming** — WebSocket for live progress updates
- [ ] **Fine-tuned models** — specialized models for different genres

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **[BTC-ISMIR19](https://github.com/jayg996/BTC-ISMIR19)** — the Bi-Directional Transformer model and pretrained weights
- **[Librosa](https://librosa.org/)** — audio analysis library
- **[FastAPI](https://fastapi.tiangolo.com/)** — modern Python web framework
- **[Hugging Face Spaces](https://huggingface.co/spaces)** — hosting and deployment

---

<p align="center">
  <sub>Built with ❤️ and 🎶</sub>
  <br>
  <a href="https://jeremyszs-chord-engine.hf.space/api/docs">Try it now →</a>
</p>
