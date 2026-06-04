"""Audio file loading and preprocessing."""
from pathlib import Path
import librosa
import numpy as np
from config import AUDIO


# Supported audio formats
SUPPORTED_FORMATS = {'.mp3', '.wav', '.flac', '.ogg', '.m4a'}


def validate_audio(path: str) -> bool:
    """
    Validate an audio file without loading it.

    Checks if the file exists and has a supported format.

    Args:
        path: Path to the audio file

    Returns:
        True if the file exists and format is supported, False otherwise
    """
    file_path = Path(path)

    # Check if file exists
    if not file_path.exists() or not file_path.is_file():
        return False

    # Check if format is supported
    extension = file_path.suffix.lower()
    if extension not in SUPPORTED_FORMATS:
        return False

    return True


def load_audio(path: str) -> dict:
    """
    Load an audio file and preprocess it for chord recognition.

    Accepts MP3, WAV, FLAC, OGG, and M4A formats. The audio is converted
    to mono, resampled to 22050 Hz, normalized to [-1, 1], and silence
    is trimmed from the beginning and end.

    Args:
        path: Path to the audio file

    Returns:
        Dictionary containing:
            - y: Audio waveform as numpy array
            - sr: Sample rate (22050 Hz)
            - duration: Duration in seconds
            - original_path: Original file path

    Raises:
        ValueError: If file doesn't exist, format is unsupported,
                   or audio is shorter than 2 seconds
    """
    file_path = Path(path)

    # Validate file existence
    if not file_path.exists():
        raise ValueError(
            f"Audio file not found: {path}\n"
            f"Please check that the file exists and the path is correct."
        )

    if not file_path.is_file():
        raise ValueError(
            f"Path is not a file: {path}\n"
            f"Please provide a path to an audio file, not a directory."
        )

    # Validate file format
    extension = file_path.suffix.lower()
    if extension not in SUPPORTED_FORMATS:
        supported_list = ', '.join(sorted(SUPPORTED_FORMATS))
        raise ValueError(
            f"Unsupported audio format: {extension}\n"
            f"Supported formats: {supported_list}\n"
            f"Please convert your file to a supported format."
        )

    # Load audio file
    # librosa automatically converts to mono and resamples
    try:
        y, sr = librosa.load(path, sr=AUDIO["sample_rate"], mono=AUDIO["mono"])
    except Exception as e:
        raise ValueError(
            f"Failed to load audio file: {path}\n"
            f"Error: {str(e)}\n"
            f"The file may be corrupted or in an unsupported codec."
        )

    # Trim leading and trailing silence
    y, _ = librosa.effects.trim(y, top_db=AUDIO["silence_top_db"])

    # Normalize amplitude to [-1, 1]
    y = librosa.util.normalize(y)

    # Calculate duration
    duration = float(len(y)) / sr

    # Validate minimum duration
    if duration < AUDIO["min_duration_seconds"]:
        raise ValueError(
            f"Audio is too short: {duration:.2f} seconds\n"
            f"Minimum duration for chord detection: {AUDIO['min_duration_seconds']} seconds\n"
            f"Please provide a longer audio file."
        )

    return {
        "y": y,
        "sr": sr,
        "duration": duration,
        "original_path": str(file_path.absolute())
    }
