## Gunshot Audio Classification

![Python](https://img.shields.io/badge/Python-3.10-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1+cu121-orange) ![CUDA](https://img.shields.io/badge/CUDA-RTX%204060-green) ![License](https://img.shields.io/badge/License-MIT-lightgrey)

Audio-based firearm identification system that classifies gunshot recordings into 8 weapon classes using deep learning on MFCC spectrograms.

---

## Overview

- What it does: classifies gunshot audio into weapon classes using deep learning on MFCC spectrograms.
- Why it matters: useful for law enforcement, smart city surveillance, and detection in low-visibility conditions where audio is available but visual data is not.
- Input: raw `.wav` audio file or microphone recording
- Output: identified weapon class + confidence score + consensus from 2 models

---

## Dataset

- 851 labelled `.wav` recordings across 9 folders → 8 classes
- M16 and M4 merged into M-Family due to acoustic similarity

| Class | Label | Samples |
|---|---:|---:|
| AK-12 | 0 | 98 |
| AK-47 | 1 | 72 |
| IMI Desert Eagle | 2 | 100 |
| M-Family (M16 + M4) | 3 | 200 |
| M249 | 4 | 99 |
| MG-42 | 5 | 100 |
| MP5 | 6 | 100 |
| Zastava M92 | 7 | 82 |
| **Total** |  | **851** |

---

## Audio Preprocessing Pipeline

1. Mono conversion (stereo averaged across channels)
2. Resample to 48,000 Hz
3. Pad or crop to 2 seconds (96,000 samples)
4. Pre-emphasis filter (coefficient = 0.97)
5. Amplitude normalisation to [-1.0, 1.0]
6. MFCC extraction → output tensor [1, 40, 188] (n_mfcc=40, n_fft=1024, hop_length=512, n_mels=64)

---

## Models

Three experiment folders, each self-contained.

### `basic_cnn_model/` — Baseline CNN

- 2 conv blocks (1→32→64 filters), Dropout(0.3), FC(30080→128→8)
- Adam lr=0.001, ReduceLROnPlateau, 75/25 split, 25 epochs
- Test Accuracy: 82.17%

### `better_cnn_model/` — Improved CNN ★ Best Architecture

- 2 conv blocks (1→32→128 filters), AdaptiveAvgPool2d(4×4), Dropout(0.4), FC(2048→256→8)
- Adam lr=0.0005 wd=2e-3, CosineAnnealingLR(T_max=50), stratified 70/15/15 split
- WeightedRandomSampler + SpecAugment (FreqMask=8, TimeMask=15) + Gaussian noise + time shift
- GPU: CUDA RTX 4060
- Test Accuracy: 73.44% | Inference Test: 77.8%

### `transfer_learning_model/` — 5 Model Comparison

| Model | Trainable Params | Train Acc | Val Acc | Test Acc |
|---|---:|---:|---:|---:|
| GunshotCNN | 169,288 | 81.34% | 69.53% | 71.88% |
| EfficientNet-B0 | 10,536 | 77.98% | 64.84% | 67.19% |
| MobileNetV2 | 10,536 | 72.44% | 68.75% | 63.28% |
| DeepGunshotCNN | 634,632 | 80.17% | 64.84% | 73.44% |
| ViT-B/16 | 6,152 | 66.72% | 48.44% | 43.75% |

---

## Key Findings

- Custom CNNs outperform all transfer learning models on this dataset size
- ViT-B/16 failed (43.75%) — too small a dataset for attention patterns to generalise
- M-Family is the weakest class — merging M16+M4 creates acoustic inconsistency
- AK-12 and Zastava M92 are the most distinctive classes — near-perfect identification
- MP5 is the hardest class — acoustic profile overlaps with several other weapons

---

## Project Structure

```
Gunshot_detection_pytorch/
├── basic_cnn_model/
│   ├── gunshot_classification.ipynb
│   ├── inference.py
│   └── inference_test.py
├── better_cnn_model/
│   ├── better_gunshot_classification.ipynb
│   ├── inference.py
│   └── inference_test.py
├── transfer_learning_model/
│   ├── gunshot_transfer_learning.ipynb
│   ├── inference.py          ← dual model consensus
│   └── inference_test.py     ← tests all 5 models
├── gunshot-audio-dataset/    ← not tracked in git
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Inference

Run from `transfer_learning_model/`:

```bash
python inference.py
```

Menu:

- `[R]` Record from microphone (2 seconds)
- `[U]` Upload a `.wav` file path

Output example:

```text
==================================================
  GUNSHOT CLASSIFICATION RESULTS
==================================================
  Model                  Prediction         Confidence
  ------------------------------------------------
  BetterCNN              MG-42               53.88%
  DeepGunshotCNN         MG-42               47.53%
  ------------------------------------------------
  ✓ CONSENSUS: MG-42
==================================================
```

Consensus logic:

- ✓ Both models agree → CONSENSUS
- ~ Models disagree → higher confidence wins
- ✗ Both below 40% threshold → Unknown / Uncertain

---

## Setup

```bash
# Clone the repo
git clone https://github.com/gaurangpatil97/Gunshot_detection_pytorch.git
cd Gunshot_detection_pytorch

# Create venv with Python 3.10 (required — newer versions have no CUDA wheels)
py -3.10 -m venv gunshotvenv310
gunshotvenv310\scripts\activate

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
pip install -r requirements.txt
```

Note: Dataset (`gunshot-audio-dataset/`) is not included in the repo. Place it in the root before running any notebook.

Each model folder is self-contained — run its notebook top to bottom to preprocess, train, and export the model files. Then run `inference.py` for predictions.

---

## Reporting & Presentation

[📄 Experiment Report (PDF)](Gunshot_Classification_Report-GaurangPatil.pdf)

---

## Tech Stack

| Component | Technology |
|---|---|
| Framework | PyTorch 2.5.1+cu121 |
| Audio Loading | wave (Python built-in) + soundfile |
| Features | MFCC — 40 coefficients, 48kHz |
| Augmentation | SpecAugment (torchaudio), Gaussian noise, time shift |
| Transfer Models | torchvision (EfficientNet-B0, MobileNetV2, ViT-B/16) |
| Evaluation | scikit-learn (classification_report, confusion_matrix) |
| Environment | Python 3.10, CUDA GPU (RTX 4060) |

---

## Future Work

- Expand dataset to 500+ samples per class
- Separate M16 and M4 back into distinct classes
- Try audio-specific pretrained models (PANNs, YAMNet)
- Test on real-world recordings with background noise
- Weighted ensemble of BetterCNN + DeepGunshotCNN outputs

---
