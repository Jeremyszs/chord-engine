"""Evaluation script for chord recognition using mir_eval."""
import argparse
import sys
import numpy as np
from pathlib import Path
import mir_eval

from engine.loader import load_audio
from engine.model import load_detector
from engine.postprocess import smooth_chords, merge_segments
from config import MODEL


def load_lab_file(lab_path: str) -> tuple[np.ndarray, list[str]]:
    """
    Load a .lab reference file with chord annotations.

    Args:
        lab_path: Path to .lab file (tab-separated: start end label)

    Returns:
        Tuple of (intervals, labels) where:
        - intervals: numpy array of shape (N, 2) with start/end times
        - labels: list of N chord label strings
    """
    intervals = []
    labels = []

    with open(lab_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split('\t')
            if len(parts) < 3:
                parts = line.split()  # Try space-separated as fallback

            if len(parts) >= 3:
                start = float(parts[0])
                end = float(parts[1])
                label = parts[2]

                intervals.append([start, end])
                labels.append(label)

    if not intervals:
        raise ValueError(f"No valid annotations found in {lab_path}")

    return np.array(intervals), labels


def evaluate(audio_path: str, reference_lab_path: str,
             device: str = "cpu", smooth_method: str = "hmm") -> dict:
    """
    Evaluate chord recognition against reference annotations.

    Runs the full chord recognition pipeline and compares results
    against ground truth using mir_eval metrics.

    Args:
        audio_path: Path to audio file
        reference_lab_path: Path to reference .lab annotation file
        device: Device for model ("cpu" or "cuda")
        smooth_method: Smoothing method ("hmm" or "median")

    Returns:
        Dictionary with evaluation scores:
        - root: Root note accuracy (0-1)
        - thirds: Thirds accuracy (0-1)
        - triads: Triads accuracy (0-1)
        - sevenths: Sevenths accuracy (0-1)
        - tetrads: Tetrads accuracy (0-1)
        - mirex: MIREX score (0-1) - main benchmark metric
    """
    # Load reference annotations
    ref_intervals, ref_labels = load_lab_file(reference_lab_path)

    # Run chord recognition pipeline
    # Step 1: Load audio
    audio_dict = load_audio(audio_path)
    y = audio_dict['y']
    sr = audio_dict['sr']

    # Step 2: Load model and predict
    detector = load_detector(device=device)
    raw_chords = detector.predict(y, sr)

    # Step 3: Calculate frame times (BTC uses hop_length from config)
    hop_length = MODEL["hop_length"]
    frame_times = np.array([i * hop_length / sr for i in range(len(raw_chords))])

    # Step 4: Smooth predictions
    smoothed_chords = smooth_chords(raw_chords, method=smooth_method)

    # Step 5: Merge into segments
    segments = merge_segments(smoothed_chords, frame_times)

    # Convert segments to mir_eval format (intervals, labels)
    est_intervals = []
    est_labels = []
    for segment in segments:
        est_intervals.append([segment['start'], segment['end']])
        est_labels.append(segment['chord'])

    est_intervals = np.array(est_intervals)

    # Run mir_eval evaluation
    # mir_eval expects intervals and labels for both reference and estimate
    scores = mir_eval.chord.evaluate(
        ref_intervals=ref_intervals,
        ref_labels=ref_labels,
        est_intervals=est_intervals,
        est_labels=est_labels
    )

    # Extract the main metrics
    result = {
        'root': float(scores['root']),
        'thirds': float(scores['thirds']),
        'triads': float(scores['triads']),
        'sevenths': float(scores['sevenths']),
        'tetrads': float(scores['tetrads']),
        'mirex': float(scores['mirex'])
    }

    return result


def print_scores(scores: dict) -> None:
    """
    Print evaluation scores in a formatted table.

    Args:
        scores: Dictionary of evaluation scores
    """
    print()
    print("=" * 40)
    print("  EVALUATION SCORES".center(40))
    print("=" * 40)
    print(f"  Root accuracy    :  {scores['root']:.3f}")
    print(f"  Thirds accuracy  :  {scores['thirds']:.3f}")
    print(f"  Triads accuracy  :  {scores['triads']:.3f}")
    print(f"  Sevenths accuracy:  {scores['sevenths']:.3f}")
    print(f"  Tetrads accuracy :  {scores['tetrads']:.3f}")
    print("-" * 40)
    print(f"  MIREX score      :  {scores['mirex']:.3f}")
    print("=" * 40)
    print()


def main():
    """Parse arguments and run evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate chord recognition against reference annotations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python eval.py --audio song.wav --ref song.lab
  python eval.py --audio song.mp3 --ref song.lab --device cuda
        """
    )

    parser.add_argument(
        '--audio',
        type=str,
        required=True,
        help='Path to input audio file'
    )

    parser.add_argument(
        '--ref',
        type=str,
        required=True,
        help='Path to reference .lab annotation file'
    )

    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda'],
        default='cpu',
        help='Device to run model on (default: cpu)'
    )

    parser.add_argument(
        '--smooth',
        type=str,
        choices=['hmm', 'median'],
        default='hmm',
        help='Chord smoothing method (default: hmm)'
    )

    args = parser.parse_args()

    # Validate files exist
    audio_path = Path(args.audio)
    ref_path = Path(args.ref)

    if not audio_path.exists():
        print(f"[ERROR] Audio file not found: {args.audio}", file=sys.stderr)
        sys.exit(1)

    if not ref_path.exists():
        print(f"[ERROR] Reference file not found: {args.ref}", file=sys.stderr)
        sys.exit(1)

    # Run evaluation
    try:
        print(f"Evaluating: {args.audio}")
        print(f"Reference: {args.ref}")
        print("Running chord recognition pipeline...")

        scores = evaluate(
            audio_path=str(audio_path),
            reference_lab_path=str(ref_path),
            device=args.device,
            smooth_method=args.smooth
        )

        print_scores(scores)

    except Exception as e:
        print(f"\n[ERROR] Evaluation failed: {type(e).__name__}: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
