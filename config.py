"""Configuration for chord-engine - single source of truth for all parameters."""

import os

# Audio processing parameters
AUDIO = {
    "sample_rate": 22050,
    "mono": True,
    "silence_top_db": 20,
    "min_duration_seconds": 2.0,
}

# Feature extraction parameters
FEATURES = {
    "hop_length": 512,
    "bins_per_octave": 36,
    "use_hpss": True,
}

# BTC model parameters
MODEL = {
    "checkpoint_dir": "btc_model/",
    "device": "cpu",
    "vocab_size": 170,
    "hop_length": 2048,  # BTC-specific hop length for feature extraction
}

# Post-processing parameters
POSTPROCESS = {
    "smooth_method": "hmm",
    "hmm_n_iter": 100,
    "median_window": 5,
}

# Chord vocabulary (25 classes: major/minor + N)
CHORD_LABELS = [
    "C", "C:min", "C#", "C#:min", "D", "D:min", "D#", "D#:min",
    "E", "E:min", "F", "F:min", "F#", "F#:min", "G", "G:min",
    "G#", "G#:min", "A", "A:min", "A#", "A#:min", "B", "B:min", "N"
]

# API layer parameters
API = {
    "max_file_size_bytes": 50 * 1024 * 1024,   # 50 MB upload limit
    "supported_extensions": [".mp3", ".wav", ".flac", ".ogg", ".m4a"],
    "cleanup_interval_seconds": 1800,            # periodic job purge interval
    "max_job_age_minutes": 60,                   # jobs older than this are purged
}

# Storage / temp directory (HF Spaces only allows writes to /tmp)
STORAGE = {
    "tmp_dir": "/tmp",
    "max_upload_mb": 50,
}

# Output format options
OUTPUT_FORMAT = "json"  # Default output format for save operations

# YouTube download configuration
YOUTUBE = {
    "cookies_path": os.environ.get("YOUTUBE_COOKIES_PATH"),
    "max_duration_seconds": 600,
    "audio_quality": "192",
}
