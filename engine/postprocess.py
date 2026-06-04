"""Post-processing for chord predictions."""
import numpy as np
from scipy import stats
from collections import Counter, defaultdict
from hmmlearn import hmm

from config import POSTPROCESS


def smooth_chords(chord_labels: list[str], method: str = None,
                  confidences: list[float] = None) -> list[str]:
    """
    Smooth chord predictions to remove single-frame noise.

    Args:
        chord_labels: Raw chord predictions, one per frame
        method: Smoothing method - "median" or "hmm" (default from config)
        confidences: Optional confidence scores for HMM method

    Returns:
        Smoothed chord label sequence
    """
    if len(chord_labels) == 0:
        return []

    if method is None:
        method = POSTPROCESS["smooth_method"]

    if method == "median":
        return _smooth_median(chord_labels)
    elif method == "hmm":
        return _smooth_hmm(chord_labels, confidences)
    else:
        raise ValueError(f"Unknown smoothing method: {method}")


def _smooth_median(chord_labels: list[str], window: int = None) -> list[str]:
    """Apply sliding median filter to chord sequence."""
    if window is None:
        window = POSTPROCESS["median_window"]

    if len(chord_labels) < window:
        return chord_labels

    smoothed = []
    half_window = window // 2

    for i in range(len(chord_labels)):
        # Get window around current position
        start = max(0, i - half_window)
        end = min(len(chord_labels), i + half_window + 1)
        window_labels = chord_labels[start:end]

        # Find most common chord in window (mode)
        counts = Counter(window_labels)
        most_common = counts.most_common(1)[0][0]
        smoothed.append(most_common)

    return smoothed


def _smooth_hmm(chord_labels: list[str], confidences: list[float] = None) -> list[str]:
    """Smooth using Hidden Markov Model with Viterbi decoding."""
    if len(chord_labels) < 2:
        return chord_labels

    # Get unique chords and create mapping
    unique_chords = sorted(set(chord_labels))
    chord_to_idx = {chord: i for i, chord in enumerate(unique_chords)}
    n_states = len(unique_chords)

    if n_states == 1:
        return chord_labels  # All same chord, no smoothing needed

    # Convert labels to indices
    observations = np.array([chord_to_idx[c] for c in chord_labels])

    # Build transition matrix from bigram counts
    transitions = np.ones((n_states, n_states))  # Laplace smoothing
    for i in range(len(observations) - 1):
        transitions[observations[i], observations[i + 1]] += 1

    # Normalize to probabilities
    transition_matrix = transitions / transitions.sum(axis=1, keepdims=True)

    # Start probability: uniform
    start_prob = np.ones(n_states) / n_states

    # Build HMM model
    model = hmm.CategoricalHMM(n_components=n_states)
    model.startprob_ = start_prob
    model.transmat_ = transition_matrix

    # Emission probabilities: identity matrix (each state emits itself)
    # with some noise for robustness
    emission_matrix = np.eye(n_states) * 0.9 + 0.1 / n_states
    model.emissionprob_ = emission_matrix

    # Decode using Viterbi algorithm
    observations_2d = observations.reshape(-1, 1)
    decoded_indices = model.predict(observations_2d)

    # Convert back to chord labels
    idx_to_chord = {i: chord for chord, i in chord_to_idx.items()}
    smoothed = [idx_to_chord[idx] for idx in decoded_indices]

    return smoothed


def merge_segments(chord_labels: list[str], times: np.ndarray) -> list[dict]:
    """
    Collapse consecutive identical chord labels into segments.

    Args:
        chord_labels: Chord labels, one per frame
        times: Time in seconds for each frame

    Returns:
        List of segment dicts with chord, start, end, duration
    """
    if len(chord_labels) == 0:
        return []

    if len(chord_labels) != len(times):
        raise ValueError(
            f"Length mismatch: {len(chord_labels)} labels, {len(times)} times"
        )

    segments = []
    current_chord = chord_labels[0]
    start_time = times[0]

    for i in range(1, len(chord_labels)):
        if chord_labels[i] != current_chord:
            # End of current segment
            end_time = times[i]
            segments.append({
                "chord": current_chord,
                "start": float(start_time),
                "end": float(end_time),
                "duration": float(end_time - start_time)
            })
            # Start new segment
            current_chord = chord_labels[i]
            start_time = times[i]

    # Add final segment (use last time + average duration as end)
    if len(times) > 1:
        avg_frame_time = (times[-1] - times[0]) / (len(times) - 1)
        end_time = times[-1] + avg_frame_time
    else:
        end_time = times[-1] + 1.0  # Default 1 second

    segments.append({
        "chord": current_chord,
        "start": float(start_time),
        "end": float(end_time),
        "duration": float(end_time - start_time)
    })

    return segments


# Krumhansl-Schmuckler key profiles (from cognitive psychology research)
# Values represent the perceptual stability of each pitch class in major/minor keys
KS_MAJOR_PROFILE = np.array([
    6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88
])

KS_MINOR_PROFILE = np.array([
    6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17
])


def infer_key(segments: list[dict]) -> str:
    """
    Infer musical key using Krumhansl-Schmuckler algorithm.

    Analyzes chord root distribution weighted by duration and correlates
    against major/minor key profiles.

    Args:
        segments: List of segment dicts with chord and duration

    Returns:
        Detected key as string (e.g., "C major", "A minor")
    """
    if len(segments) == 0:
        return "C major"  # Default

    # Extract pitch class histogram weighted by duration
    pitch_class_histogram = np.zeros(12)

    # Mapping from root names to pitch classes
    root_to_pc = {
        'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
        'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
        'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11
    }

    for segment in segments:
        chord_label = segment['chord']
        duration = segment['duration']

        # Skip non-chord labels
        if chord_label in ['N', 'X', '-']:
            continue

        # Parse root from chord label (format: "C:maj", "A:min", "C", etc.)
        if ':' in chord_label:
            root = chord_label.split(':')[0]
        else:
            root = chord_label

        # Get pitch class
        if root in root_to_pc:
            pc = root_to_pc[root]
            pitch_class_histogram[pc] += duration

    # Normalize histogram
    if pitch_class_histogram.sum() > 0:
        pitch_class_histogram /= pitch_class_histogram.sum()
    else:
        return "C major"  # Default if no valid chords

    # Correlate with all 24 major and minor keys
    best_correlation = -1
    best_key = "C major"

    pc_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    # Try all major keys
    for tonic_pc in range(12):
        # Rotate key profile to match this tonic
        rotated_profile = np.roll(KS_MAJOR_PROFILE, tonic_pc)
        correlation = np.corrcoef(pitch_class_histogram, rotated_profile)[0, 1]

        if correlation > best_correlation:
            best_correlation = correlation
            best_key = f"{pc_names[tonic_pc]} major"

    # Try all minor keys
    for tonic_pc in range(12):
        # Rotate key profile to match this tonic
        rotated_profile = np.roll(KS_MINOR_PROFILE, tonic_pc)
        correlation = np.corrcoef(pitch_class_histogram, rotated_profile)[0, 1]

        if correlation > best_correlation:
            best_correlation = correlation
            best_key = f"{pc_names[tonic_pc]} minor"

    return best_key


def to_roman_numerals(segments: list[dict], key: str) -> list[dict]:
    """
    Add roman numeral analysis to segments relative to a key.

    Args:
        segments: List of segment dicts with chord labels
        key: Key string (e.g., "C major", "A minor")

    Returns:
        Updated segments with "roman" field added
    """
    # Parse key
    key_parts = key.split()
    if len(key_parts) != 2:
        return segments  # Invalid key format

    tonic_name, mode = key_parts[0], key_parts[1]

    # Map tonic to pitch class
    root_to_pc = {'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5,
                  'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11}

    if tonic_name not in root_to_pc:
        return segments

    tonic_pc = root_to_pc[tonic_name]

    # Scale degrees to roman numerals
    if mode == "major":
        major_scale = [0, 2, 4, 5, 7, 9, 11]  # Major scale intervals
        roman_maj = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII']
        roman_min = ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii°']
        qualities = ['maj', 'min', 'min', 'maj', 'maj', 'min', 'dim']
    else:  # minor
        major_scale = [0, 2, 3, 5, 7, 8, 10]  # Natural minor scale intervals
        roman_maj = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII']
        roman_min = ['i', 'ii', 'iii°', 'iv', 'v', 'VI', 'VII']
        qualities = ['min', 'dim', 'maj', 'min', 'min', 'maj', 'maj']

    for segment in segments:
        chord_label = segment['chord']

        # Skip non-chords
        if chord_label in ['N', 'X', '-']:
            segment['roman'] = '?'
            continue

        # Parse chord root and quality
        if ':' in chord_label:
            root, quality = chord_label.split(':', 1)
            if quality == 'min':
                quality = 'min'
            else:
                quality = 'maj'
        else:
            root = chord_label
            quality = 'maj'

        if root not in root_to_pc:
            segment['roman'] = '?'
            continue

        chord_pc = root_to_pc[root]

        # Calculate interval from tonic
        interval = (chord_pc - tonic_pc) % 12

        # Find scale degree
        if interval in major_scale:
            degree = major_scale.index(interval)
            expected_quality = qualities[degree]

            # Choose roman numeral based on quality match
            if quality == expected_quality or quality == 'maj':
                numeral = roman_maj[degree] if quality == 'maj' else roman_min[degree]
            else:
                numeral = roman_min[degree] if quality == 'min' else roman_maj[degree]

            segment['roman'] = numeral
        else:
            segment['roman'] = '?'  # Chromatic/out-of-key

    return segments


def extract_progression(segments: list[dict]) -> str:
    """
    Extract the main chord progression pattern.

    Finds the most repeated n-gram (length 3-6) as the core loop.

    Args:
        segments: List of segment dicts with roman numerals

    Returns:
        Progression string (e.g., "I → V → vi → IV")
    """
    if len(segments) < 3:
        return " → ".join([s.get('roman', '?') for s in segments])

    # Extract roman numerals
    numerals = [s.get('roman', '?') for s in segments if s.get('roman') != '?']

    if len(numerals) < 3:
        return " → ".join(numerals) if numerals else "?"

    # Find most common n-gram (length 3-6)
    best_ngram = []
    best_count = 0

    for n in range(min(6, len(numerals)), 2, -1):
        ngram_counts = Counter()
        for i in range(len(numerals) - n + 1):
            ngram = tuple(numerals[i:i+n])
            ngram_counts[ngram] += 1

        if ngram_counts:
            most_common_ngram, count = ngram_counts.most_common(1)[0]
            if count > best_count:
                best_count = count
                best_ngram = list(most_common_ngram)

    if best_ngram and best_count > 1:
        return " -> ".join(best_ngram)

    # Fallback: return unique sequence
    unique_progression = []
    for num in numerals:
        if not unique_progression or num != unique_progression[-1]:
            unique_progression.append(num)

    return " -> ".join(unique_progression[:8])  # Limit to first 8 for readability
