# ChordMiniApp Model Integration

## Overview

This chord-engine now supports ChordMiniApp's superior **BTC-PL model** (Pseudo-Labeling + Knowledge Distillation), which provides state-of-the-art chord recognition accuracy.

## What is ChordMini BTC-PL?

ChordMiniApp's BTC-PL model uses a two-stage training pipeline:

1. **Stage 1: Pseudo-Labeling** - Student model trained on 1,072+ hours of unlabeled audio using teacher-generated pseudo-labels
2. **Stage 2: Continual Learning** - Fine-tuned on labeled data with selective knowledge distillation

### Performance Improvements (from arXiv:2602.19778)

Compared to baseline BTC model:
- **Root**: +1.51% (81.52% → 83.03%)
- **Thirds**: +2.17% (78.00% → 80.17%)
- **Triads**: +2.21% (76.12% → 78.33%)
- **7ths**: +3.45% (65.93% → 69.38%)
- **MIREX**: +2.37% (77.79% → 80.16%)
- **Rare chords**: Dim7 (0% → 45.6%), Maj7 (28.2% → 42.3%)

## Installation

The ChordMini model checkpoint is already included in this repository:
- Location: `btc_model/test/btc_model_chordmini.pth`
- Size: 35.9 MB
- Source: https://github.com/ptnghia-j/ChordMini

## Usage

### CLI

```bash
# Use baseline BTC model (default)
python main.py --audio song.wav

# Use ChordMini BTC-PL model (superior accuracy)
python main.py --audio song.wav --use-chordmini

# With other options
python main.py --audio song.wav --use-chordmini --verbose --device cuda
```

### Python API

```python
from engine.model import load_detector

# Load baseline BTC model
detector = load_detector(device="cpu", large_voca=True)

# Load ChordMini BTC-PL model
detector = load_detector(device="cpu", use_chordmini=True)

# Run inference
predictions = detector.predict_with_confidence(audio, sr=22050)
```

## Model Comparison

### Test: samples/demo_chords.wav (16s, C major I-IV-V-I)

| Metric | Baseline BTC | ChordMini BTC-PL |
|--------|:------------:|:----------------:|
| Segments | 10 | 7 |
| Avg Confidence | 0.762 | 0.723 |
| Progression | I→IV→V→I | I→IV→V→I |
| Key Detection | C major | C major |

Both models correctly detected the progression, but ChordMini produced cleaner output with fewer segments.

## Technical Details

### Checkpoint Format

ChordMini uses a different checkpoint format than baseline BTC:

**Baseline BTC** (`.pt`):
```python
{
    'model': state_dict,
    'mean': float,
    'std': float
}
```

**ChordMini** (`.pth`):
```python
{
    'model_state_dict': state_dict,
    'normalization': {
        'mean': tensor,
        'std': tensor
    },
    'epoch': int,
    'optimizer_state_dict': dict,
    'chord_mapping': dict,
    'idx_to_chord': dict
}
```

### Normalization Stats

- **Baseline BTC**: mean=varies, std=varies (per checkpoint)
- **ChordMini BTC-PL**: mean=-2.3698, std=1.9627

These are critical for correct feature normalization during inference.

## References

- Paper: Phan et al. "Enhancing Automatic Chord Recognition via Pseudo-Labeling and Knowledge Distillation" arXiv:2602.19778, 2026
- ChordMini Repository: https://github.com/ptnghia-j/ChordMini
- ChordMiniApp: https://github.com/ptnghia-j/ChordMiniApp
- Live Demo: https://www.chordmini.me
