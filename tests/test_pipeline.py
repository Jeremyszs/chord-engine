"""Tests for the complete chord recognition pipeline."""
import pytest
import numpy as np
from engine.features import extract_chroma, beat_sync_chroma, frames_to_times
from engine.postprocess import extract_progression


def generate_sine_wave(freq: float, duration: float, sr: int = 22050) -> np.ndarray:
    """
    Generate a synthetic sine wave for testing.

    Args:
        freq: Frequency in Hz
        duration: Duration in seconds
        sr: Sample rate

    Returns:
        Audio waveform as numpy array
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return np.sin(2 * np.pi * freq * t)


def test_extract_chroma_shape():
    """Test that extract_chroma returns correct shape (12, T)."""
    # Generate 5 seconds of 440 Hz sine wave (A4 note)
    sr = 22050
    duration = 5.0
    y = generate_sine_wave(440.0, duration, sr)

    audio_dict = {
        'y': y,
        'sr': sr
    }

    config = {
        'hop_length': 512,
        'bins_per_octave': 36
    }

    # Extract chroma features
    chroma = extract_chroma(audio_dict, config)

    # Check shape: should be (12, T) where T is number of frames
    assert chroma.shape[0] == 12, f"Expected 12 chroma bins, got {chroma.shape[0]}"
    assert chroma.ndim == 2, f"Expected 2D array, got {chroma.ndim}D"
    assert chroma.shape[1] > 0, "Expected at least one time frame"


def test_extract_chroma_value_range():
    """Test that extract_chroma values are in valid range [0, 1]."""
    # Generate 5 seconds of 440 Hz sine wave
    sr = 22050
    duration = 5.0
    y = generate_sine_wave(440.0, duration, sr)

    audio_dict = {
        'y': y,
        'sr': sr
    }

    config = {
        'hop_length': 512,
        'bins_per_octave': 36
    }

    # Extract chroma features
    chroma = extract_chroma(audio_dict, config)

    # Check value range: normalized features should be in [0, 1]
    assert np.all(chroma >= 0), f"Found negative values: min={np.min(chroma)}"
    assert np.all(chroma <= 1), f"Found values > 1: max={np.max(chroma)}"
    assert not np.any(np.isnan(chroma)), "Found NaN values in chroma"
    assert not np.any(np.isinf(chroma)), "Found Inf values in chroma"


def test_extract_chroma_missing_keys():
    """Test that extract_chroma raises ValueError for missing dict keys."""
    # Missing 'sr' key
    with pytest.raises(ValueError, match="must contain"):
        extract_chroma({'y': np.zeros(1000)})

    # Missing 'y' key
    with pytest.raises(ValueError, match="must contain"):
        extract_chroma({'sr': 22050})


def test_beat_sync_chroma_reduction():
    """Test that beat_sync_chroma reduces temporal dimension."""
    # Generate 5 seconds of audio with clear beats using a click track
    # Combined with a 440 Hz sine wave for harmonic content
    sr = 22050
    duration = 5.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # Create a click track at 2 Hz (120 BPM)
    # Generate clicks as short impulses every 0.5 seconds
    y = np.zeros_like(t)
    click_positions = np.arange(0, duration, 0.5)  # Every 0.5 seconds
    for pos in click_positions:
        idx = int(pos * sr)
        if idx < len(y) - 100:
            # Add a short click (exponentially decaying impulse)
            click_len = 100
            click = np.exp(-np.linspace(0, 5, click_len))
            y[idx:idx + click_len] += click

    # Add harmonic content (440 Hz sine wave) so chroma has something to detect
    y += 0.3 * np.sin(2 * np.pi * 440 * t)

    audio_dict = {
        'y': y,
        'sr': sr
    }

    config = {
        'hop_length': 512,
        'bins_per_octave': 36
    }

    # Extract chroma features
    chroma = extract_chroma(audio_dict, config)
    original_frames = chroma.shape[1]

    # Apply beat synchronization
    beat_chroma, beat_times = beat_sync_chroma(chroma, y, sr, hop_length=512)

    # Check that beat-synced chroma has fewer frames than original
    assert beat_chroma.shape[0] == 12, f"Expected 12 chroma bins, got {beat_chroma.shape[0]}"
    assert beat_chroma.shape[1] < original_frames, \
        f"Beat-synced chroma should have fewer frames ({beat_chroma.shape[1]}) than original ({original_frames})"

    # Check that beat times match the number of beat frames
    assert len(beat_times) == beat_chroma.shape[1], \
        f"Number of beat times ({len(beat_times)}) should match beat-synced frames ({beat_chroma.shape[1]})"

    # Check that beat times are monotonically increasing
    assert np.all(np.diff(beat_times) > 0), "Beat times should be monotonically increasing"


def test_beat_sync_chroma_invalid_shape():
    """Test that beat_sync_chroma raises ValueError for invalid chroma shape."""
    sr = 22050
    y = generate_sine_wave(440.0, 5.0, sr)

    # Wrong shape: 1D array
    with pytest.raises(ValueError, match="must have shape"):
        beat_sync_chroma(np.zeros(100), y, sr)

    # Wrong shape: (10, T) instead of (12, T)
    with pytest.raises(ValueError, match="must have shape"):
        beat_sync_chroma(np.zeros((10, 100)), y, sr)


def test_frames_to_times():
    """Test frame-to-time conversion."""
    sr = 22050
    hop_length = 512
    n_frames = 100

    times = frames_to_times(n_frames, sr, hop_length)

    # Check shape
    assert times.shape == (n_frames,), f"Expected shape ({n_frames},), got {times.shape}"

    # Check that times are monotonically increasing
    assert np.all(np.diff(times) > 0), "Times should be monotonically increasing"

    # Check first time is close to 0
    assert times[0] >= 0, "First time should be >= 0"

    # Check approximate time step
    expected_step = hop_length / sr
    actual_steps = np.diff(times)
    assert np.allclose(actual_steps, expected_step, rtol=0.01), \
        f"Time steps should be approximately {expected_step:.4f}s"


def test_progression_extraction():
    """Test chord progression extraction from segments."""
    from engine.postprocess import extract_progression, to_roman_numerals

    # Create test segments with roman numerals
    segments = [
        {"chord": "C", "start": 0.0, "end": 1.0, "duration": 1.0},
        {"chord": "C", "start": 1.0, "end": 2.0, "duration": 1.0},
        {"chord": "Am", "start": 2.0, "end": 3.0, "duration": 1.0},
        {"chord": "F", "start": 3.0, "end": 4.0, "duration": 1.0},
        {"chord": "G", "start": 4.0, "end": 5.0, "duration": 1.0}
    ]

    # Add roman numerals (in C major)
    segments_with_roman = to_roman_numerals(segments, "C major")

    # Extract progression
    progression = extract_progression(segments_with_roman)

    # Should return a progression string
    assert isinstance(progression, str)
    assert len(progression) > 0
    assert " -> " in progression or len(segments) <= 2


def test_merge_segments():
    """Test merging consecutive identical chords into segments."""
    from engine.postprocess import merge_segments

    # Test basic merging
    chords = ["C", "C", "C", "Am", "Am", "F", "F", "F", "F"]
    times = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])

    segments = merge_segments(chords, times)

    assert len(segments) == 3  # C, Am, F
    assert segments[0]["chord"] == "C"
    assert segments[0]["start"] == 0.0
    assert segments[0]["end"] == 1.5
    assert segments[0]["duration"] == 1.5

    assert segments[1]["chord"] == "Am"
    assert segments[1]["start"] == 1.5
    assert segments[1]["end"] == 2.5
    assert segments[1]["duration"] == 1.0

    assert segments[2]["chord"] == "F"
    assert segments[2]["start"] == 2.5
    assert abs(segments[2]["end"] - 4.5) < 0.1  # Approximate end time
    assert abs(segments[2]["duration"] - 2.0) < 0.1


def test_merge_segments_single():
    """Test merging with single chord."""
    from engine.postprocess import merge_segments

    chords = ["C"]
    times = np.array([0.0])

    segments = merge_segments(chords, times)

    assert len(segments) == 1
    assert segments[0]["chord"] == "C"
    assert segments[0]["start"] == 0.0


def test_infer_key_c_major():
    """Test key inference for C major progression."""
    from engine.postprocess import infer_key

    # Typical C major progression: C - F - G - C
    segments = [
        {"chord": "C", "duration": 2.0},
        {"chord": "F", "duration": 1.5},
        {"chord": "G", "duration": 1.5},
        {"chord": "C", "duration": 2.0}
    ]

    key = infer_key(segments)

    # Should detect C major (or possibly G major/A minor)
    assert "major" in key or "minor" in key
    # C, F, G strongly suggest C major
    assert key in ["C major", "G major", "F major"]


def test_infer_key_a_minor():
    """Test key inference for A minor progression."""
    from engine.postprocess import infer_key

    # Typical A minor progression: Am - Dm - E - Am
    segments = [
        {"chord": "A:min", "duration": 2.0},
        {"chord": "D:min", "duration": 1.5},
        {"chord": "E", "duration": 1.5},
        {"chord": "A:min", "duration": 2.0}
    ]

    key = infer_key(segments)

    # Should detect A minor (or possibly C major, the relative major)
    assert "A" in key or "C" in key


def test_infer_key_empty():
    """Test key inference with no segments."""
    from engine.postprocess import infer_key

    key = infer_key([])

    # Should return default
    assert key == "C major"


def test_build_output():
    """Test building complete output dictionary."""
    from engine.output import build_output

    # Create mock segments
    segments = [
        {
            "chord": "C",
            "roman": "I",
            "start": 0.0,
            "end": 2.0,
            "duration": 2.0
        },
        {
            "chord": "G",
            "roman": "V",
            "start": 2.0,
            "end": 4.0,
            "duration": 2.0
        }
    ]

    # Create mock audio (2 seconds of sine wave)
    sr = 22050
    t = np.linspace(0, 2.0, sr * 2)
    y = np.sin(2 * np.pi * 440 * t)
    audio_dict = {"y": y, "sr": sr}

    # Mock data
    key = "C major"
    progression = "I → V"
    raw_chords = ["C", "C", "G", "G"]
    confidences = [0.95, 0.93]

    # Build output
    result = build_output(segments, key, progression, audio_dict, raw_chords, confidences)

    # Assert all required keys are present
    assert "key" in result
    assert "progression" in result
    assert "tempo_bpm" in result
    assert "duration_seconds" in result
    assert "chord_count" in result
    assert "segments" in result
    assert "raw_chords" in result

    # Assert values
    assert result["key"] == "C major"
    assert result["progression"] == "I → V"
    assert result["tempo_bpm"] >= 0  # Tempo detection (0 if no beats detected)
    assert abs(result["duration_seconds"] - 2.0) < 0.1  # ~2 seconds
    assert result["chord_count"] == 2  # C and G
    assert len(result["segments"]) == 2
    assert len(result["raw_chords"]) == 4

    # Check confidence scores were added to segments
    assert result["segments"][0]["confidence"] == 0.95
    assert result["segments"][1]["confidence"] == 0.93
