"""
inference_test.py — Batch inference tester for basic_cnn_model
Picks 2 random audio files from each class folder in the dataset,
runs inference on each, and prints actual vs predicted.

Usage:
    python inference_test.py
"""

import os, pickle, random, wave as wave_module
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T

# ── CONFIG ───────────────────────────────────────────────────
DATASET_PATH    = Path("../gunshot-audio-dataset")
FILES_PER_CLASS = 2
CONFIDENCE_THRESHOLD = 40.0

# ── 1. LOAD CONFIG & MAPPING ────────────────────────────────
with open("preprocessing_config.pkl", "rb") as f:
    cfg = pickle.load(f)

with open("class_mapping.pkl", "rb") as f:
    mapping = pickle.load(f)

idx_to_class = mapping["idx_to_class"]
num_classes  = len(idx_to_class)

# ── 2. MODEL ────────────────────────────────────────────────
class GunshotCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1   = nn.Conv2d(1, 32, 3, 1, 1)
        self.bn1     = nn.BatchNorm2d(32)
        self.pool1   = nn.MaxPool2d(2)
        self.conv2   = nn.Conv2d(32, 64, 3, 1, 1)
        self.bn2     = nn.BatchNorm2d(64)
        self.pool2   = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.3)
        self.fc1     = nn.Linear(64 * 10 * 47, 128)
        self.fc2     = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = GunshotCNN(num_classes).to(device)
model.load_state_dict(torch.load("best_gunshot_model.pth", map_location=device, weights_only=True))
model.eval()

# ── 3. MFCC TRANSFORM ───────────────────────────────────────
mfcc_transform = T.MFCC(
    sample_rate=cfg["target_sr"],
    n_mfcc=cfg["n_mfcc"],
    melkwargs={"n_fft": cfg["n_fft"], "hop_length": cfg["hop_length"], "n_mels": cfg["n_mels"]}
)

# ── 4. LOAD & PREPROCESS ────────────────────────────────────
def load_wav(file_path):
    with wave_module.open(str(file_path), 'rb') as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        b  = wf.readframes(wf.getnframes())
        w  = torch.from_numpy(np.frombuffer(b, dtype=np.int16).copy()).float()
        w  = w.view(-1, ch).T / 32768.0
    return w, sr

def preprocess(waveform, sr):
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    if sr != cfg["target_sr"]:
        waveform = T.Resample(sr, cfg["target_sr"])(waveform)
    target_samples = cfg["target_sr"] * cfg["target_sec"]
    if waveform.shape[1] > target_samples:
        waveform = waveform[:, :target_samples]
    else:
        waveform = F.pad(waveform, (0, target_samples - waveform.shape[1]))
    coeff    = cfg["pre_emphasis_coeff"]
    waveform = torch.cat((waveform[:, 0:1], waveform[:, 1:] - coeff * waveform[:, :-1]), dim=1)
    threshold = cfg["noise_gate_threshold"]
    waveform  = waveform * (waveform.abs() >= threshold).float()
    waveform  = waveform / (waveform.abs().max() + 1e-8)
    return waveform

def predict(file_path):
    waveform, sr = load_wav(file_path)
    waveform     = preprocess(waveform, sr)
    mfcc         = mfcc_transform(waveform).unsqueeze(0).to(device)
    with torch.no_grad():
        prob      = F.softmax(model(mfcc), dim=1)
        conf, idx = torch.max(prob, 1)
    label      = idx_to_class[idx.item()]
    confidence = conf.item() * 100
    return ("Unknown / Uncertain", confidence) if confidence < CONFIDENCE_THRESHOLD else (label, confidence)

# ── 5. BATCH TEST ────────────────────────────────────────────
print("\n" + "="*70)
print("  BATCH INFERENCE TEST — basic_cnn_model")
print(f"  Dataset : {DATASET_PATH}")
print(f"  Files   : {FILES_PER_CLASS} per class")
print(f"  Device  : {device}")
print("="*70)

class_folders = sorted([d for d in DATASET_PATH.iterdir() if d.is_dir()])

correct = 0
total   = 0
results = []

for folder in class_folders:
    actual_class = folder.name
    wav_files    = list(folder.glob("*.wav"))

    if not wav_files:
        print(f"\n  [!] No .wav files found in {folder.name}, skipping")
        continue

    sampled = random.sample(wav_files, min(FILES_PER_CLASS, len(wav_files)))

    for file_path in sampled:
        predicted, confidence = predict(file_path)

        actual_display = "M-Family" if actual_class in ["M16", "M4"] else actual_class
        is_correct     = predicted == actual_display

        if is_correct:
            correct += 1
        total += 1

        results.append({
            "file":       file_path.name,
            "folder":     actual_class,
            "actual":     actual_display,
            "predicted":  predicted,
            "confidence": confidence,
            "correct":    is_correct
        })

# ── 6. PRINT RESULTS ─────────────────────────────────────────
print(f"\n{'FILE':<25} {'FOLDER':<18} {'ACTUAL':<18} {'PREDICTED':<22} {'CONF':>7}  {'✓/✗'}")
print("-"*100)

for r in results:
    status = "✓" if r["correct"] else "✗"
    print(
        f"{r['file']:<25} "
        f"{r['folder']:<18} "
        f"{r['actual']:<18} "
        f"{r['predicted']:<22} "
        f"{r['confidence']:>6.1f}%  "
        f"{status}"
    )

print("-"*100)
print(f"\n  Correct : {correct} / {total}")
print(f"  Accuracy: {100 * correct / total:.1f}%")
print("="*70 + "\n")