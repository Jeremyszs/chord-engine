"""BTC pretrained chord recognition model wrapper."""
import os
import sys
from pathlib import Path
import numpy as np
import torch
import librosa

from config import MODEL

# Add btc_model directory to Python path to import BTC modules
_btc_model_path = Path(__file__).parent.parent / MODEL["checkpoint_dir"]
if str(_btc_model_path) not in sys.path:
    sys.path.insert(0, str(_btc_model_path))

from btc_model import BTC_model
from utils.hparams import HParams
from utils.mir_eval_modules import idx2chord, idx2voca_chord


def extract_btc_features(y: np.ndarray, sr: int, config: dict) -> np.ndarray:
    """
    Extract CQT features for BTC model.

    The BTC model was trained on 144-bin CQT features with specific
    parameters. This function replicates that exact feature extraction.

    Args:
        y: Audio waveform
        sr: Sample rate (should be 22050 Hz)
        config: Configuration dict with feature parameters

    Returns:
        Log-transformed CQT features of shape (n_bins, T)
    """
    # Extract CQT features with BTC parameters
    cqt = librosa.cqt(
        y=y,
        sr=sr,
        n_bins=config['n_bins'],
        bins_per_octave=config['bins_per_octave'],
        hop_length=config['hop_length']
    )

    # Log transform (as done in original BTC code)
    feature = np.log(np.abs(cqt) + 1e-6)

    return feature


class ChordDetector:
    """
    Wrapper for BTC (Bidirectional Transformer for Chord Recognition) model.

    Loads pretrained BTC model and provides inference methods for chord
    recognition from audio.
    """

    def __init__(self, checkpoint_path: str, device: str = "cpu", large_voca: bool = False):
        """
        Initialize the BTC chord detector.

        Args:
            checkpoint_path: Path to BTC model checkpoint (.pt file)
            device: Device to run model on ("cpu" or "cuda")
            large_voca: If True, use large vocabulary (170 chords),
                       otherwise use standard (25 chords: maj/min only)

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            RuntimeError: If model loading fails
        """
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint file not found: {checkpoint_path}\n"
                f"Please ensure the BTC model weights are downloaded."
            )

        self.device = torch.device(device)
        self.large_voca = large_voca

        # Load configuration
        config_path = _btc_model_path / "run_config.yaml"
        self.config = HParams.load(str(config_path))

        # Update config for large vocabulary if needed
        if large_voca:
            self.config.feature['large_voca'] = True
            self.config.model['num_chords'] = 170
            self.idx_to_chord = idx2voca_chord()
        else:
            self.config.model['num_chords'] = 25
            self.idx_to_chord = idx2chord

        # Initialize model
        self.model = BTC_model(config=self.config.model).to(self.device)

        # Load checkpoint
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            self.mean = checkpoint['mean']
            self.std = checkpoint['std']
            self.model.load_state_dict(checkpoint['model'])
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load model checkpoint: {str(e)}")

        # Model parameters
        self.timestep = self.config.model['timestep']
        self.feature_size = self.config.model['feature_size']

    def _preprocess_features(self, feature: np.ndarray) -> torch.Tensor:
        """
        Preprocess features for model input.

        Args:
            feature: CQT features of shape (n_bins, T)

        Returns:
            Preprocessed tensor ready for model input
        """
        # Transpose to (T, n_bins)
        feature = feature.T

        # Normalize with checkpoint mean/std
        feature = (feature - self.mean) / self.std

        # Pad to be divisible by timestep
        num_pad = self.timestep - (feature.shape[0] % self.timestep)
        if num_pad != self.timestep:
            feature = np.pad(feature, ((0, num_pad), (0, 0)), mode="constant", constant_values=0)

        # Convert to tensor
        feature = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).to(self.device)

        return feature, num_pad

    def predict(self, y: np.ndarray, sr: int = 22050) -> list[str]:
        """
        Predict chord labels from audio.

        Args:
            y: Audio waveform (mono)
            sr: Sample rate (should be 22050 Hz)

        Returns:
            List of chord label strings, one per time frame
            (e.g., ["C", "C:min", "G", "A:min", ...])
        """
        # Extract BTC features
        cqt_feature = extract_btc_features(
            y, sr, self.config.feature
        )

        # Preprocess for model
        feature, num_pad = self._preprocess_features(cqt_feature)
        num_instance = feature.shape[1] // self.timestep

        # Run inference
        predictions = []
        with torch.no_grad():
            for t in range(num_instance):
                # Extract chunk
                chunk = feature[:, self.timestep * t:self.timestep * (t + 1), :]

                # Forward pass through self-attention layers
                self_attn_output, _ = self.model.self_attn_layers(chunk)

                # Get predictions from output layer
                prediction, _ = self.model.output_layer(self_attn_output)
                prediction = prediction.squeeze().cpu().numpy()

                # Convert indices to chord labels
                for i in range(self.timestep):
                    # Skip padding frames at the end
                    if t == num_instance - 1 and i >= self.timestep - num_pad:
                        break
                    chord_idx = int(prediction[i])
                    predictions.append(self.idx_to_chord[chord_idx])

        return predictions

    def predict_with_confidence(self, y: np.ndarray, sr: int = 22050) -> list[dict]:
        """
        Predict chord labels with confidence scores.

        Args:
            y: Audio waveform (mono)
            sr: Sample rate (should be 22050 Hz)

        Returns:
            List of dicts with keys:
                - "chord": Chord label string
                - "confidence": Confidence score (0-1)
        """
        # Extract BTC features
        cqt_feature = extract_btc_features(
            y, sr, self.config.feature
        )

        # Preprocess for model
        feature, num_pad = self._preprocess_features(cqt_feature)
        num_instance = feature.shape[1] // self.timestep

        # Run inference
        results = []
        with torch.no_grad():
            for t in range(num_instance):
                # Extract chunk
                chunk = feature[:, self.timestep * t:self.timestep * (t + 1), :]

                # Forward pass through self-attention layers
                self_attn_output, _ = self.model.self_attn_layers(chunk)

                # Get raw logits from output layer
                # We need to access the layer directly to get probabilities
                logits = self.model.output_layer.output_projection(self_attn_output)
                probs = torch.softmax(logits, dim=-1)
                probs = probs.squeeze().cpu().numpy()

                # Get predictions
                prediction = np.argmax(probs, axis=-1)

                # Convert to chord labels with confidence
                for i in range(self.timestep):
                    # Skip padding frames at the end
                    if t == num_instance - 1 and i >= self.timestep - num_pad:
                        break
                    chord_idx = int(prediction[i])
                    confidence = float(probs[i, chord_idx])
                    results.append({
                        "chord": self.idx_to_chord[chord_idx],
                        "confidence": confidence
                    })

        return results


def load_detector(device: str = "cpu", large_voca: bool = False) -> ChordDetector:
    """
    Load BTC chord detector with automatic checkpoint discovery.

    Searches for the appropriate checkpoint file in the btc_model/test
    directory and loads the pretrained model.

    Args:
        device: Device to run model on ("cpu" or "cuda")
        large_voca: If True, load large vocabulary model (170 chords),
                   otherwise load standard model (25 chords)

    Returns:
        Initialized ChordDetector instance

    Raises:
        FileNotFoundError: If checkpoint file is not found
    """
    # Determine checkpoint filename
    if large_voca:
        checkpoint_name = "btc_model_large_voca.pt"
    else:
        checkpoint_name = "btc_model.pt"

    # Search for checkpoint in btc_model/test directory
    checkpoint_path = _btc_model_path / "test" / checkpoint_name

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"BTC model checkpoint not found: {checkpoint_path}\n"
            f"Please ensure the BTC-ISMIR19 repository is cloned to {MODEL['checkpoint_dir']}\n"
            f"and contains the pretrained weights in {MODEL['checkpoint_dir']}/test/"
        )

    # Load and return detector
    detector = ChordDetector(
        checkpoint_path=str(checkpoint_path),
        device=device,
        large_voca=large_voca
    )

    return detector
