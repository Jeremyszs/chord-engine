"""Post-processing for chord predictions."""
import numpy as np
import librosa
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


def _build_musical_transition_matrix(unique_chords: list[str], chord_to_idx: dict) -> np.ndarray:
    """
    Build music-theoretic transition matrix based on harmonic relationships.
    
    Uses circle of fifths and common chord progressions rather than
    circular bigram counts from predictions.
    """
    n_states = len(unique_chords)
    transitions = np.ones((n_states, n_states)) * 0.01  # Base small probability
    
    # Root to pitch class mapping
    root_to_pc = {
        'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
        'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
        'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11
    }
    
    for i, chord_i in enumerate(unique_chords):
        # Self-transition (chord holds)
        transitions[i, i] = 0.7
        
        # Skip special labels
        if chord_i in ['N', 'X', '-']:
            transitions[i, :] = 1.0 / n_states  # Uniform from N/X
            continue
        
        # Parse chord i
        if ':' in chord_i:
            root_i, quality_i = chord_i.split(':', 1)
        else:
            root_i = chord_i
            quality_i = 'maj'
        
        if root_i not in root_to_pc:
            continue
        
        pc_i = root_to_pc[root_i]
        
        # Build transition weights based on interval relationships
        for j, chord_j in enumerate(unique_chords):
            if i == j:
                continue  # Already set self-transition
            
            if chord_j in ['N', 'X', '-']:
                transitions[i, j] = 0.02  # Low probability to silence
                continue
            
            # Parse chord j
            if ':' in chord_j:
                root_j, quality_j = chord_j.split(':', 1)
            else:
                root_j = chord_j
                quality_j = 'maj'
            
            if root_j not in root_to_pc:
                continue
            
            pc_j = root_to_pc[root_j]
            
            # Calculate interval distance
            interval = (pc_j - pc_i) % 12
            
            # Weight based on interval (circle of fifths)
            if interval == 7:  # Perfect fifth (dominant movement)
                weight = 0.15
            elif interval == 5:  # Perfect fourth (subdominant movement)
                weight = 0.12
            elif interval == 9:  # Major sixth / relative minor
                weight = 0.08
            elif interval == 3:  # Minor third / relative major
                weight = 0.08
            elif interval == 2 or interval == 10:  # Stepwise
                weight = 0.05
            else:
                weight = 0.02
            
            transitions[i, j] = weight
    
    # Normalize rows to sum to 1
    row_sums = transitions.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    transitions = transitions / row_sums
    
    return transitions


def _smooth_hmm(chord_labels: list[str], confidences: list[float] = None) -> list[str]:
    """
    Smooth using HMM with music-theoretic transitions and confidence-based emissions.
    
    This fixes the circular smoothing issue where transitions were built from
    the predictions themselves. Now uses musical knowledge (circle of fifths).
    """
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

    # Build music-theoretic transition matrix
    transition_matrix = _build_musical_transition_matrix(unique_chords, chord_to_idx)

    # Start probability: uniform
    start_prob = np.ones(n_states) / n_states

    # Build emission probabilities from confidences if available
    if confidences is not None and len(confidences) == len(chord_labels):
        # Use confidence scores to build emission matrix
        # High confidence -> strong emission, low confidence -> diffuse
        emission_matrix = np.zeros((n_states, n_states))
        for obs_idx, conf in zip(observations, confidences):
            # Observed state emits itself with probability = confidence
            emission_matrix[obs_idx, obs_idx] += conf
            # Distribute remaining probability to other states
            remaining = 1.0 - conf
            for other_idx in range(n_states):
                if other_idx != obs_idx:
                    emission_matrix[obs_idx, other_idx] += remaining / (n_states - 1)
        
        # Normalize
        emission_matrix = emission_matrix / (emission_matrix.sum(axis=1, keepdims=True) + 1e-10)
    else:
        # Fallback: identity-based emissions with noise
        emission_matrix = np.eye(n_states) * 0.85 + 0.15 / n_states

    # Build HMM model
    model = hmm.CategoricalHMM(n_components=n_states)
    model.startprob_ = start_prob
    model.transmat_ = transition_matrix
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


def refine_boundaries(segments: list[dict], y: np.ndarray, sr: int, 
                      hop_length: int, max_shift: float = 0.15) -> list[dict]:
    """
    Snap chord boundaries to nearest onset positions for better timing accuracy.
    
    Uses librosa's onset detector to find note attack positions, then aligns
    each segment boundary to the nearest onset within max_shift seconds.
    This fixes timing drift caused by fixed-frame boundaries.
    
    Args:
        segments: List of segment dicts with start/end times
        y: Audio waveform (for onset detection)
        sr: Sample rate
        hop_length: Hop length used in feature extraction
        max_shift: Maximum boundary adjustment in seconds (default 0.15)
    
    Returns:
        Segments with refined boundaries snapped to onsets
    """
    if len(segments) == 0:
        return segments
    
    # Detect onsets in the audio
    onset_frames = librosa.onset.onset_detect(
        y=y, sr=sr, hop_length=hop_length, backtrack=True
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)
    
    if len(onset_times) == 0:
        return segments  # No onsets detected, return unchanged
    
    refined = []
    for i, seg in enumerate(segments):
        refined_seg = seg.copy()
        
        # Refine start boundary (skip first segment's start)
        if i > 0:
            start_time = seg['start']
            # Find nearest onset to start boundary
            distances = np.abs(onset_times - start_time)
            nearest_idx = np.argmin(distances)
            nearest_onset = onset_times[nearest_idx]
            
            # Only snap if within max_shift threshold
            if distances[nearest_idx] < max_shift:
                refined_seg['start'] = float(nearest_onset)
        
        # Refine end boundary (skip last segment's end)
        if i < len(segments) - 1:
            end_time = seg['end']
            # Find nearest onset to end boundary
            distances = np.abs(onset_times - end_time)
            nearest_idx = np.argmin(distances)
            nearest_onset = onset_times[nearest_idx]
            
            # Only snap if within max_shift threshold
            if distances[nearest_idx] < max_shift:
                refined_seg['end'] = float(nearest_onset)
        
        # Recalculate duration
        refined_seg['duration'] = refined_seg['end'] - refined_seg['start']
        refined.append(refined_seg)
    
    return refined


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


def normalize_chord_label(chord: str) -> str:
    """
    Clean and normalize a chord label to standard form.

    Normalizations:
        - Unicode flat (♭) → 'b'
        - Unicode sharp (♯) → '#'
        - Strip leading/trailing whitespace
        - Normalize 'min' variations to 'min'
        - Normalize 'maj' to bare root (e.g. 'C:maj' → 'C')

    Args:
        chord: Raw chord label string

    Returns:
        Normalized chord label
    """
    chord = chord.strip()
    chord = chord.replace('♭', 'b').replace('♯', '#')

    # Normalize quality suffixes
    if ':' in chord:
        root, quality = chord.split(':', 1)
        quality = quality.strip()
        if quality in ('maj', 'M', ''):
            return root  # bare root = major
        if quality in ('min', 'm', 'minor'):
            quality = 'min'
        return f'{root}:{quality}'

    return chord


def filter_short_chords(segments: list[dict],
                         min_duration: float = 0.1) -> list[dict]:
    """
    Remove segments shorter than min_duration, merging them into neighbors.

    Short segments are usually spurious noise predictions. Each short segment
    is merged into the longer adjacent neighbor to preserve continuous coverage
    with no gaps.

    Args:
        segments: List of segment dicts with chord/start/end/duration
        min_duration: Minimum segment duration in seconds (default 0.1)

    Returns:
        Filtered segments with short ones merged into neighbors
    """
    if not segments:
        return []

    # First pass: collect non-short segments
    kept = []
    for seg in segments:
        if seg['duration'] >= min_duration:
            kept.append(seg.copy())
        elif kept:
            # Merge short into previous neighbor
            kept[-1]['end'] = seg['end']
            kept[-1]['duration'] = kept[-1]['end'] - kept[-1]['start']

    # Handle case where first segment(s) are short (no previous to merge into):
    # absorb them into the first non-short segment by adjusting its start
    if kept and kept[0]['start'] > segments[0]['start']:
        kept[0]['start'] = segments[0]['start']
        kept[0]['duration'] = kept[0]['end'] - kept[0]['start']

    # If all segments were short, return the longest one
    if not kept:
        longest = max(segments, key=lambda s: s['duration'])
        return [longest.copy()]

    return kept


def merge_consecutive_chords(segments: list[dict],
                              tolerance: float = 0.01) -> list[dict]:
    """
    Merge adjacent segments with same chord when boundary gap is below tolerance.

    Args:
        segments: List of segment dicts
        tolerance: Max gap in seconds to consider for merging (default 0.01)

    Returns:
        Merged segment list
    """
    if len(segments) < 2:
        return segments

    merged = [segments[0].copy()]
    for seg in segments[1:]:
        prev = merged[-1]
        gap = seg['start'] - prev['end']
        if seg['chord'] == prev['chord'] and gap >= 0 and gap <= tolerance:
            # Merge: extend previous segment
            prev['end'] = seg['end']
            prev['duration'] = prev['end'] - prev['start']
        else:
            merged.append(seg.copy())

    return merged


def sync_chords_to_beats(segments: list[dict], beat_times: np.ndarray,
                         sr: int, hop_length: int) -> list[dict]:
    """
    Snap chord boundaries to nearest beat positions using two-pointer algorithm.

    Implements ChordMiniApp's beat synchronization approach:
    - For each chord segment boundary, find the closest beat
    - Use 50% midpoint rule: snap to nearest beat within threshold
    - Maintain forward-fill semantics (chord sustains until next change)

    Args:
        segments: List of segment dicts with start/end times
        beat_times: Array of beat times in seconds from beat detection
        sr: Sample rate
        hop_length: Hop length used in feature extraction

    Returns:
        Segments with beat-aligned boundaries
    """
    if len(segments) == 0 or len(beat_times) == 0:
        return segments

    synced = []

    for i, seg in enumerate(segments):
        synced_seg = seg.copy()

        # Snap start boundary to nearest beat (skip first segment)
        if i > 0:
            start_time = seg['start']
            # Find nearest beat using two-pointer approach
            distances = np.abs(beat_times - start_time)
            nearest_idx = np.argmin(distances)
            nearest_beat = beat_times[nearest_idx]

            # Apply 50% midpoint rule
            # Only snap if the boundary is closer to this beat than the frame resolution
            frame_time = hop_length / sr
            if distances[nearest_idx] < frame_time:
                synced_seg['start'] = float(nearest_beat)

        # Snap end boundary to nearest beat (skip last segment)
        if i < len(segments) - 1:
            end_time = seg['end']
            # Find nearest beat
            distances = np.abs(beat_times - end_time)
            nearest_idx = np.argmin(distances)
            nearest_beat = beat_times[nearest_idx]

            # Apply 50% midpoint rule
            frame_time = hop_length / sr
            if distances[nearest_idx] < frame_time:
                synced_seg['end'] = float(nearest_beat)

        # Recalculate duration
        synced_seg['duration'] = synced_seg['end'] - synced_seg['start']

        # Ensure positive duration (in case beats are very close)
        if synced_seg['duration'] <= 0:
            synced_seg = seg.copy()  # Revert to original if invalid

        synced.append(synced_seg)

    return synced
