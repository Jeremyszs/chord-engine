"""Main CLI entry point for chord-engine."""
import argparse
import sys
import io
import numpy as np
from pathlib import Path

# Fix encoding for Windows console - use UTF-8 for all output
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from engine.loader import load_audio
from engine.model import load_detector
import librosa
from engine.postprocess import (smooth_chords, merge_segments, infer_key,
                                 to_roman_numerals, extract_progression,
                                 refine_boundaries, filter_short_chords,
                                 merge_consecutive_chords, normalize_chord_label,
                                 sync_chords_to_beats)
from engine.output import build_output, save_output, print_summary
from config import MODEL, POSTPROCESS


def run_pipeline(audio_path: str, output_path: str, device: str,
                 smooth_method: str, save_result: bool, verbose: bool,
                 use_chordmini: bool = False) -> int:
    """
    Run the complete chord recognition pipeline.

    Args:
        audio_path: Path to input audio file
        output_path: Path to save JSON output
        device: Device to run model on ("cpu" or "cuda")
        smooth_method: Smoothing method ("hmm" or "median")
        save_result: Whether to save JSON output
        verbose: Whether to print progress messages

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    total_steps = 14  # Updated for new post-processing steps
    current_step = 0

    def log_progress(message: str):
        """Print progress message if verbose mode is enabled."""
        nonlocal current_step
        current_step += 1
        if verbose:
            print(f"[{current_step}/{total_steps}] {message}")

    try:
        # Step 1: Load audio
        log_progress("Loading audio file...")
        audio_dict = load_audio(audio_path)
        y = audio_dict['y']
        sr = audio_dict['sr']

        if verbose:
            print(f"  → Loaded {audio_dict['duration']:.2f}s audio at {sr} Hz")

        # Step 2: Load BTC model with large vocabulary
        log_progress("Loading BTC chord recognition model...")
        detector = load_detector(device=device, large_voca=True, use_chordmini=use_chordmini)

        model_type = "ChordMini BTC-PL" if use_chordmini else "BTC baseline"
        if verbose:
            print(f"  → Model loaded on {device} (170-chord vocabulary, {model_type})")

        # Step 3: Predict chords with confidence scores
        # Note: BTC model performs its own CQT feature extraction internally
        # (144-bin CQT with hop_length=2048), so we pass raw audio directly
        log_progress("Running chord recognition (this may take a moment)...")
        predictions = detector.predict_with_confidence(y, sr)
        raw_chords = [p['chord'] for p in predictions]
        confidences = [p['confidence'] for p in predictions]

        if verbose:
            avg_conf = sum(confidences) / len(confidences) if confidences else 0
            print(f"  → Predicted {len(raw_chords)} chord frames (avg confidence: {avg_conf:.3f})")

        # Step 4: Calculate frame times
        # BTC uses hop_length from MODEL config
        log_progress("Calculating frame timestamps...")
        hop_length = MODEL["hop_length"]
        frame_times = np.array([i * hop_length / sr for i in range(len(raw_chords))])

        # Step 5: Smooth predictions with confidence-aware HMM
        log_progress(f"Smoothing predictions ({smooth_method} method)...")
        smoothed_chords = smooth_chords(raw_chords, method=smooth_method, confidences=confidences)

        if verbose:
            print(f"  → Smoothed {len(smoothed_chords)} frames")

        # Step 6: Merge into segments
        log_progress("Merging consecutive chords into segments...")
        segments = merge_segments(smoothed_chords, frame_times)

        if verbose:
            print(f"  → Created {len(segments)} segments")

        # Step 6.5: Beat tracking for timing synchronization
        log_progress("Detecting beats for timing alignment...")
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=512)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=512)
        
        if verbose:
            print(f"  → Detected {len(beat_times)} beats at {float(np.atleast_1d(tempo)[0]):.1f} BPM")

        # Step 6.6: Sync chord boundaries to beats
        log_progress("Synchronizing chords to beat grid...")
        segments = sync_chords_to_beats(segments, beat_times, sr, hop_length)
        
        if verbose:
            print(f"  → Synchronized {len(segments)} segments to beats")

        # Step 6.7: Refine boundaries using onset detection
        log_progress("Refining boundaries with onset detection...")
        segments = refine_boundaries(segments, y, sr, hop_length, max_shift=0.15)

        if verbose:
            print(f"  → Refined {len(segments)} segment boundaries")

        # Step 6.6: Normalize chord labels (unicode flats/sharps, quality format)
        log_progress("Normalizing chord labels...")
        for seg in segments:
            seg['chord'] = normalize_chord_label(seg['chord'])

        # Step 6.7: Filter short segments (< 100ms spurious noise)
        log_progress("Filtering short chord segments...")
        segments = filter_short_chords(segments, min_duration=0.1)

        if verbose:
            print(f"  → After filtering: {len(segments)} segments")

        # Step 6.8: Merge consecutive same-chord segments with small gaps
        log_progress("Merging consecutive same-chord segments...")
        segments = merge_consecutive_chords(segments, tolerance=0.01)

        if verbose:
            print(f"  → After merging: {len(segments)} segments")

        # Step 7: Infer key
        log_progress("Detecting musical key...")
        key = infer_key(segments)

        if verbose:
            print(f"  → Detected key: {key}")

        # Step 8: Add roman numerals
        log_progress("Adding roman numeral analysis...")
        segments = to_roman_numerals(segments, key)

        # Step 9: Extract progression
        log_progress("Extracting chord progression pattern...")
        progression = extract_progression(segments)

        if verbose:
            print(f"  → Progression: {progression}")

        # Step 10: Build complete output
        log_progress("Building final output...")
        result = build_output(
            segments=segments,
            key=key,
            progression=progression,
            audio_dict=audio_dict,
            raw_chords=raw_chords,
            beats=beat_times
        )

        if verbose:
            print(f"  → Output complete")

        # Print summary
        print()  # Blank line before summary
        print_summary(result)

        # Save output if requested
        if save_result:
            save_output(result, output_path)
            print()
            print(f"[OK] Results saved to: {output_path}")

        return 0

    except FileNotFoundError as e:
        print(f"\n[ERROR] Step [{current_step}/{total_steps}]: File not found", file=sys.stderr)
        error_msg = str(e).encode('ascii', 'replace').decode('ascii')
        print(f"  {error_msg}", file=sys.stderr)
        return 1

    except ValueError as e:
        print(f"\n[ERROR] Step [{current_step}/{total_steps}]: Invalid input", file=sys.stderr)
        error_msg = str(e).encode('ascii', 'replace').decode('ascii')
        print(f"  {error_msg}", file=sys.stderr)
        return 1

    except RuntimeError as e:
        print(f"\n[ERROR] Step [{current_step}/{total_steps}]: Runtime error", file=sys.stderr)
        error_msg = str(e).encode('ascii', 'replace').decode('ascii')
        print(f"  {error_msg}", file=sys.stderr)
        return 1

    except Exception as e:
        print(f"\n[ERROR] Step [{current_step}/{total_steps}]: Unexpected error", file=sys.stderr)
        error_msg = f"{type(e).__name__}: {str(e)}".encode('ascii', 'replace').decode('ascii')
        print(f"  {error_msg}", file=sys.stderr)
        return 1


def main():
    """Parse arguments and run chord recognition pipeline."""
    parser = argparse.ArgumentParser(
        description="Chord recognition engine using BTC pretrained model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python main.py --audio song.wav

  # With custom output path
  python main.py --audio song.mp3 --output results/song.json

  # Print progress and don't save
  python main.py --audio song.wav --verbose --no-save

  # Use GPU acceleration with median smoothing
  python main.py --audio song.wav --device cuda --smooth median
        """
    )

    parser.add_argument(
        '--audio',
        type=str,
        required=True,
        help='Path to input audio file (MP3, WAV, FLAC, OGG, M4A)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='result.json',
        help='Path to save JSON result (default: result.json)'
    )

    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda'],
        default=MODEL["device"],
        help=f'Device to run model on (default: {MODEL["device"]})'
    )

    parser.add_argument(
        '--smooth',
        type=str,
        choices=['hmm', 'median'],
        default=POSTPROCESS["smooth_method"],
        help=f'Chord smoothing method (default: {POSTPROCESS["smooth_method"]})'
    )

    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save JSON output, only print summary'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print progress messages for each step'
    )

    parser.add_argument(
        '--use-chordmini',
        action='store_true',
        help='Use ChordMiniApp\'s superior BTC-PL model (pseudo-labeling + knowledge distillation)'
    )

    args = parser.parse_args()

    # Validate audio file exists
    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"[ERROR] Audio file not found: {args.audio}", file=sys.stderr)
        sys.exit(1)

    # Run pipeline
    exit_code = run_pipeline(
        audio_path=str(audio_path),
        output_path=args.output,
        device=args.device,
        smooth_method=args.smooth,
        save_result=not args.no_save,
        verbose=args.verbose,
        use_chordmini=args.use_chordmini
    )

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
