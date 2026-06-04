# chord-engine

## Project Overview
**Purpose:** Audio chord recognition engine (no frontend). Accepts an audio file, returns timestamped chord labels, detected key, and chord progression.

## Environment
- **Python version:** 3.10+
- **Virtual environment:** `.venv`
  - Activate before any command: `source .venv/bin/activate`

## Running the Project
- **Run tests:** `pytest tests/ -v`
- **Run the engine:** `python main.py --audio samples/test.wav`

## Key Libraries
- librosa
- torch
- madmom
- numpy
- hmmlearn

## Model
- **Model used:** BTC (Bidirectional Transformer for Chord Recognition)
- **Pretrained weights from:** jayg996/BTC-ISMIR19

## Configuration
- **Configuration file:** `config.py`
- **Purpose:** Single source of truth for all tunable parameters
- **Main sections:**
  - `AUDIO`: Sample rate, silence thresholds, minimum duration
  - `FEATURES`: Hop length, bins per octave, HPSS settings
  - `MODEL`: Checkpoint directory, device, vocab size
  - `POSTPROCESS`: Smoothing methods, HMM parameters, window sizes
- **How to tune:** Edit values in `config.py` - all modules import from there

## API Layer
- **Entry point:** `api/main.py`
- **Run dev server:** `uvicorn api.main:app --reload --port 8000`
- **HF Spaces port:** 7860 (set in Dockerfile CMD)
- **Run API tests:** `pytest api/tests/ -v`
- All API logic lives in `api/`. Never import from `api/` inside `engine/`.
- The `engine/` modules are pure Python with no HTTP concerns.
- Job state is in-memory for now (no database). Jobs are lost on restart.
- **Hugging Face Spaces notes:**
  - Writable dir on HF: `/tmp` only (all temp files go there)
  - All cache env vars are set in Dockerfile (`HF_HOME`, `TORCH_HOME`, `TRANSFORMERS_CACHE`, `XDG_CACHE_HOME`, `TMPDIR`)

## API Endpoints

| Method | URL                            | Description                         |
|--------|--------------------------------|-------------------------------------|
| POST   | `/api/v1/jobs`                 | Upload audio, get job_id            |
| GET    | `/api/v1/jobs/{id}/status`     | Poll job progress                   |
| GET    | `/api/v1/jobs/{id}/result`     | Fetch completed analysis            |
| GET    | `/api/v1/health`               | Health check with model status      |
| GET    | `/api/v1/health/ping`          | Lightweight ping                    |
| GET    | `/api/docs`                    | Swagger UI                          |
| GET    | `/api/redoc`                   | ReDoc UI                            |

## Notes
**Do NOT modify CLAUDE.md unless asked.**
