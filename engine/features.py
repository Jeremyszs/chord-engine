"""Audio feature extraction for chord recognition."""
import librosa
import numpy as np
from sklearn.preprocessing import normalize
from config import FEATURES


def frames_to_times(n_frames: int, sr: int, hop_length: int) -> np.ndarray:
    """
    Convert frame indices to time in seconds.

    Args:
        n_frames: Number of frames
        sr: Sample rate in Hz
        hop_length: Number of samples between successive frames

    Returns:
        Array of time values in seconds for each frame
    """
    return librosa.frames_to_time(
        frames=np.arange(n_frames),
        sr=sr,
        hop_length=hop_length
    )


def extract_chroma(audio_dict: dict, config: dict = None) -> np.ndarray:
    """
    Extract chroma features from audio using Constant-Q Transform.

    Uses harmonic-percussive source separation to remove drum noise,
    then computes chroma features with fine pitch resolution. Each
    chroma frame is normalized to unit L2 norm.

    Args:
        audio_dict: Dictionary containing audio data with keys:
                   - 'y': Audio waveform (np.ndarray)
                   - 'sr': Sample rate (int)
        config: Optional configuration dictionary with keys:
               - 'hop_length': Hop length in samples (default: 512)
               - 'bins_per_octave': Bins per octave (default: 36)

    Returns:
        Chroma feature matrix of shape (12, T) where T is the number
        of time frames. Values are normalized to [0, 1] range.

    Raises:
        ValueError: If audio_dict is missing required keys
    """
    # Validate input
    if 'y' not in audio_dict or 'sr' not in audio_dict:
        raise ValueError(
            "audio_dict must contain 'y' (waveform) and 'sr' (sample rate)"
        )

    y = audio_dict['y']
    sr = audio_dict['sr']

    # Parse config or use defaults
    if config is None:
        config = {}
    hop_length = config.get('hop_length', FEATURES["hop_length"])
    bins_per_octave = config.get('bins_per_octave', FEATURES["bins_per_octave"])

    # Apply harmonic-percussive source separation
    # This removes percussive elements (drums) and keeps harmonic content (chords)
    if FEATURES["use_hpss"]:
        y_harmonic, _ = librosa.effects.hpss(y)
    else:
        y_harmonic = y

    # Extract chroma features using Constant-Q Transform
    # CQT provides better low-frequency resolution than STFT
    chroma = librosa.feature.chroma_cqt(
        y=y_harmonic,
        sr=sr,
        hop_length=hop_length,
        bins_per_octave=bins_per_octave
    )

    # Normalize each chroma frame to unit L2 norm
    # This makes the features invariant to volume changes
    chroma = normalize(chroma, norm='l2', axis=0)

    return chroma


def beat_sync_chroma(
    chroma: np.ndarray,
    y: np.ndarray,
    sr: int,
    hop_length: int = 512
) -> tuple[np.ndarray, np.ndarray]:
    """
    Synchronize chroma features to detected beat positions.

    Detects beats in the audio and aggregates chroma frames at beat
    boundaries using median pooling. This reduces temporal resolution
    while aligning features to musically meaningful positions.

    Args:
        chroma: Chroma feature matrix of shape (12, T)
        y: Audio waveform
        sr: Sample rate in Hz
        hop_length: Hop length used for chroma extraction (default: 512)

    Returns:
        Tuple containing:
        - beat_synced_chroma: Beat-synchronized chroma of shape (12, B)
                             where B is the number of detected beats
        - beat_times: Array of beat times in seconds, shape (B,)

    Raises:
        ValueError: If chroma shape is invalid or no beats are detected
    """
    # Validate chroma shape
    if chroma.ndim != 2 or chroma.shape[0] != 12:
        raise ValueError(
            f"chroma must have shape (12, T), got {chroma.shape}"
        )

    # Detect beats in the audio
    # Returns tempo (ignored) and beat frame indices
    _, beat_frames = librosa.beat.beat_track(
        y=y,
        sr=sr,
        hop_length=hop_length
    )

    # Check if beats were detected
    if len(beat_frames) == 0:
        raise ValueError(
            "No beats detected in audio. The audio may be too short or "
            "lack a clear rhythmic structure."
        )

    # Aggregate chroma features at beat boundaries using median pooling
    # librosa.util.sync creates segments between boundaries, so it produces
    # one more frame than the number of beat boundaries (includes segments
    # before first beat and after last beat)
    beat_chroma = librosa.util.sync(
        chroma,
        idx=beat_frames,
        aggregate=np.median
    )

    # Create segment boundaries: [0, beat1, beat2, ..., beatN, T]
    # This matches the segments created by librosa.util.sync
    segment_boundaries = np.concatenate([[0], beat_frames, [chroma.shape[1]]])

    # Calculate the center time of each segment
    # This gives us one time value per beat-synced chroma frame
    segment_centers = (segment_boundaries[:-1] + segment_boundaries[1:]) / 2.0
    beat_times = librosa.frames_to_time(
        frames=segment_centers,
        sr=sr,
        hop_length=hop_length
    )

    return beat_chroma, beat_times


def extract_features(audio_dict: dict, config: dict = None) -> dict:
    """
    Extract all features needed for chord recognition.

    Args:
        audio_dict: Dictionary containing audio data with keys:
                   - 'y': Audio waveform (np.ndarray)
                   - 'sr': Sample rate (int)
        config: Optional configuration dictionary for feature extraction

    Returns:
        Dictionary containing extracted features:
        - 'chroma': Chroma feature matrix (12, T)
        - 'n_frames': Number of time frames
        - 'frame_times': Time in seconds for each frame
    """
    # Extract chroma features
    chroma = extract_chroma(audio_dict, config)

    # Calculate frame timing information
    n_frames = chroma.shape[1]
    hop_length = config.get('hop_length', FEATURES["hop_length"]) if config else FEATURES["hop_length"]
    frame_times = frames_to_times(n_frames, audio_dict['sr'], hop_length)

    return {
        'chroma': chroma,
        'n_frames': n_frames,
        'frame_times': frame_times
    }
