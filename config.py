"""Configuration for chord-engine - single source of truth for all parameters."""

# Audio processing parameters
AUDIO = {
    "sample_rate": 22050,
    "mono": True,
    "silence_top_db": 20,
    "min_duration_seconds": 2.0,
}

# Feature extraction parameters
FEATURES = {
    "hop_length": 2048,        # BTC model uses 2048
    "bins_per_octave": 24,     # BTC model uses 24
    "n_bins": 144,             # BTC model uses 144 (24 * 6 octaves)
    "use_hpss": True,
}

# BTC model parameters
MODEL = {
    "checkpoint_dir": "btc_model/",
    "device": "cpu",
    "vocab_size": 170,  # 170-chord vocabulary (incl. maj/min/dim/aug/7ths/etc.)
    "hop_length": 2048,  # BTC-specific hop length for feature extraction
    "confidence_threshold": 0.5,  # frames below this → 'N' (no chord)
}

# Post-processing parameters
POSTPROCESS = {
    "smooth_method": "hmm",
    "hmm_transition_prior": "musical",  # Use music-theoretic transitions (circle of fifths)
    "median_window": 5,
    "onset_refinement": True,  # Enable boundary snapping to onsets
    "onset_max_shift": 0.15,  # Max boundary shift in seconds
    "confidence_threshold": 0.3,  # Minimum confidence for valid chord prediction
}

# Chord vocabulary — 170 chords (12 roots × 14 qualities + N + X)
# Matches BTC large-vocabulary mapping from idx2voca_chord()
_CHORD_ROOTS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
_CHORD_QUALITIES = ['min', 'maj', 'dim', 'aug', 'min6', 'maj6', 'min7', 'minmaj7',
                    'maj7', '7', 'dim7', 'hdim7', 'sus2', 'sus4']

CHORD_LABELS = []
for root in _CHORD_ROOTS:
    for quality in _CHORD_QUALITIES:
        if quality == 'maj':
            CHORD_LABELS.append(root)  # bare root = major
        else:
            CHORD_LABELS.append(f'{root}:{quality}')
CHORD_LABELS.append('X')   # index 168 — unknown
CHORD_LABELS.append('N')   # index 169 — no chord

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
