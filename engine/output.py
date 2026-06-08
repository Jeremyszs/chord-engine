"""Output formatting and export for chord recognition results."""
import json
import librosa
import numpy as np
from pathlib import Path


def build_output(segments: list[dict], key: str, progression: str,
                 audio_dict: dict, raw_chords: list[str],
                 confidences: list[float] = None, beats: np.ndarray = None) -> dict:
    """
    Build comprehensive output dictionary from chord recognition results.

    Args:
        segments: List of chord segments with timing and roman numerals
        key: Detected key (e.g., "G major")
        progression: Chord progression string (e.g., "I → V → vi → IV")
        audio_dict: Dictionary with 'y' (waveform) and 'sr' (sample rate)
        raw_chords: Raw chord predictions before segmentation
        confidences: Optional confidence scores for each segment
        beats: Optional array of beat times in seconds

    Returns:
        Dictionary with complete chord recognition results
    """
    # Extract audio data
    y = audio_dict['y']
    sr = audio_dict['sr']

    # Calculate tempo from beat intervals if beats provided, else use librosa
    if beats is not None and len(beats) > 1:
        # Calculate tempo from mean beat interval
        beat_intervals = np.diff(beats)
        mean_interval = np.mean(beat_intervals)
        tempo_bpm = 60.0 / mean_interval if mean_interval > 0 else 120.0
    else:
        # Fallback to librosa beat tracking
        # librosa 2.x returns tempo as a numpy array; handle scalar, 0-d, and 1-d arrays
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo_bpm = float(np.atleast_1d(tempo)[0])

    # Calculate duration
    duration_seconds = float(len(y) / sr)

    # Count unique chords (excluding non-chord labels)
    unique_chords = set()
    for segment in segments:
        chord = segment['chord']
        if chord not in ['N', 'X', '-']:
            unique_chords.add(chord)
    chord_count = len(unique_chords)

    # Add confidence scores to segments if provided
    if confidences is not None and len(confidences) > 0:
        # Match confidences to segments (use first frame of each segment)
        for i, segment in enumerate(segments):
            if i < len(confidences):
                segment['confidence'] = float(confidences[i])
            else:
                segment['confidence'] = 0.0
    else:
        # Default confidence if not provided
        for segment in segments:
            if 'confidence' not in segment:
                segment['confidence'] = 1.0

    # Build result dictionary
    result = {
        "key": key,
        "progression": progression,
        "tempo_bpm": tempo_bpm,
        "duration_seconds": duration_seconds,
        "chord_count": chord_count,
        "segments": segments,
        "raw_chords": raw_chords
    }

    # Add beats array if provided
    if beats is not None:
        result["beats"] = [float(t) for t in beats]

    return result


def save_output(result: dict, path: str) -> None:
    """
    Save chord recognition result to JSON file.

    Args:
        result: Output dictionary from build_output
        path: File path to save JSON (will be created/overwritten)
    """
    output_path = Path(path)

    # Create parent directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write JSON with readable formatting
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)


def print_summary(result: dict) -> None:
    """
    Print a human-readable summary of chord recognition results.

    Displays key, progression, tempo, duration, and a timeline of
    chord segments with proportional duration bars.

    Args:
        result: Output dictionary from build_output
    """
    # Extract result data
    key = result['key']
    progression = result['progression']
    tempo_bpm = result['tempo_bpm']
    duration_seconds = result['duration_seconds']
    chord_count = result['chord_count']
    segments = result['segments']

    # Format duration as MM:SS
    minutes = int(duration_seconds // 60)
    seconds = int(duration_seconds % 60)
    duration_str = f"{minutes}:{seconds:02d}"

    # Print header
    print("=" * 50)
    print(" CHORD RECOGNITION RESULT".center(50))
    print("=" * 50)

    # Print metadata
    print(f" Key          : {key}")
    print(f" Progression  : {progression}")
    print(f" Tempo        : {int(tempo_bpm)} BPM")
    print(f" Duration     : {duration_str}")
    print(f" Unique chords: {chord_count}")

    print("-" * 50)

    # Print segments timeline
    for segment in segments:
        chord = segment['chord']
        roman = segment.get('roman', '?')
        start = segment['start']
        end = segment['end']
        duration = segment['duration']
        confidence = segment.get('confidence', 1.0)

        # Format timestamps as M:SS
        start_min = int(start // 60)
        start_sec = int(start % 60)
        end_min = int(end // 60)
        end_sec = int(end % 60)

        start_str = f"{start_min}:{start_sec:02d}"
        end_str = f"{end_min}:{end_sec:02d}"

        # Create proportional duration bar (max 8 blocks)
        # Scale based on longest segment
        max_duration = max(s['duration'] for s in segments)
        bar_length = int((duration / max_duration) * 8)
        bar_length = max(1, min(8, bar_length))  # Clamp to 1-8
        bar = "#" * bar_length

        # Format confidence as percentage
        conf_str = f"{confidence:.2f}"

        # Print segment line
        print(f" {start_str} – {end_str}   {chord:12s} ({roman:4s})  {bar:8s}  {conf_str}")

    print("=" * 50)


def format_lab_file(segments: list[dict]) -> str:
    """
    Format segments as a .lab file (Audacity label format).

    Args:
        segments: List of chord segments

    Returns:
        Lab file content as string (one line per segment)
    """
    lines = []
    for segment in segments:
        start = segment['start']
        end = segment['end']
        chord = segment['chord']
        line = f"{start:.3f}\t{end:.3f}\t{chord}"
        lines.append(line)
    return "\n".join(lines)


def save_lab_file(segments: list[dict], path: str) -> None:
    """
    Save segments as Audacity label file (.lab format).

    Args:
        segments: List of chord segments
        path: Output file path
    """
    lab_content = format_lab_file(segments)

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write(lab_content)
