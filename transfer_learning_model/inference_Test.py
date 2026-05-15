"""
inference_test.py — Batch inference tester for transfer_learning_model
Tests all 5 models on 2 random files per class and prints a comparison table.

Models tested:
    - GunshotCNN          → best_gunshot_cnn.pth
    - EfficientNet-B0     → best_gunshot_efficientnet.pth
    - MobileNetV2         → best_gunshot_mobilenet.pth
    - DeepGunshotCNN      → best_gunshot_deep_cnn.pth
    - ViT-B/16            → best_gunshot_vit.pth

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
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision.models import vit_b_16, ViT_B_16_Weights

# ── CONFIG ───────────────────────────────────────────────────
DATASET_PATH     = Path("../gunshot-audio-dataset")
FILES_PER_CLASS  = 2
CONFIDENCE_THRESHOLD = 40.0

# ── 1. LOAD CONFIG & MAPPING ────────────────────────────────
with open("preprocessing_config.pkl", "rb") as f:
    cfg = pickle.load(f)

with open("class_mapping.pkl", "rb") as f:
    mapping = pickle.load(f)

idx_to_class = mapping["idx_to_class"]
num_classes  = len(idx_to_class)
device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[+] Device: {device}")
print(f"[+] Classes: {idx_to_class}\n")

# ── 2. MODEL ARCHITECTURES ──────────────────────────────────
class GunshotCNN(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.dropout = nn.Dropout(0.5)
        self.fc1     = nn.Linear(128 * 4 * 4, 64)
        self.fc2     = nn.Linear(64, num_classes)
        self.relu    = nn.ReLU()

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


class EfficientNetAudio(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.model.classifier     = nn.Linear(1280, num_classes)
        for param in self.model.features[0][0].parameters():
            param.requires_grad = True
        for param in self.model.classifier.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)


class MobileNetAudio(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        self.model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.model.classifier     = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, num_classes))
        for param in self.model.features[0][0].parameters():
            param.requires_grad = True
        for param in self.model.classifier.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)


class DeepGunshotCNN(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.pool     = nn.AdaptiveAvgPool2d((4, 4))
        self.dropout1 = nn.Dropout(0.4)
        self.fc1      = nn.Linear(128 * 4 * 4, 256)
        self.dropout2 = nn.Dropout(0.3)
        self.fc2      = nn.Linear(256, 64)
        self.fc3      = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.pool(x)
        x = self.dropout1(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class ViTAudio(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        self.vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        for param in self.vit.parameters():
            param.requires_grad = False
        self.vit.heads = nn.Linear(768, num_classes)
        for param in self.vit.heads.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        x = x.repeat(1, 3, 1, 1)
        return self.vit(x)


# ── 3. LOAD ALL MODELS ───────────────────────────────────────
def load_model(model, path):
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model

models = {
    "GunshotCNN":      load_model(GunshotCNN(),       "best_gunshot_cnn.pth"),
    "EfficientNet-B0": load_model(EfficientNetAudio(), "best_gunshot_efficientnet.pth"),
    "MobileNetV2":     load_model(MobileNetAudio(),   "best_gunshot_mobilenet.pth"),
    "DeepGunshotCNN":  load_model(DeepGunshotCNN(),   "best_gunshot_deep_cnn.pth"),
    "ViT-B/16":        load_model(ViTAudio(),          "best_gunshot_vit.pth"),
}
print(f"[+] Loaded {len(models)} models\n")

# ── 4. MFCC TRANSFORM ───────────────────────────────────────
mfcc_transform = T.MFCC(
    sample_rate=cfg["target_sr"],
    n_mfcc=cfg["n_mfcc"],
    melkwargs={"n_fft": cfg["n_fft"], "hop_length": cfg["hop_length"], "n_mels": cfg["n_mels"]}
)

# ── 5. LOAD & PREPROCESS ────────────────────────────────────
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
    waveform = waveform / (waveform.abs().max() + 1e-8)
    return waveform

def get_mfcc(file_path):
    waveform, sr = load_wav(file_path)
    waveform     = preprocess(waveform, sr)
    return mfcc_transform(waveform).unsqueeze(0).to(device)

def predict(model, mfcc):
    with torch.no_grad():
        prob      = F.softmax(model(mfcc), dim=1)
        conf, idx = torch.max(prob, 1)
    label      = idx_to_class[idx.item()]
    confidence = conf.item() * 100
    return ("Unknown / Uncertain", confidence) if confidence < CONFIDENCE_THRESHOLD else (label, confidence)

# ── 6. SAMPLE FILES ──────────────────────────────────────────
class_folders = sorted([d for d in DATASET_PATH.iterdir() if d.is_dir()])
test_files    = []

for folder in class_folders:
    wav_files = list(folder.glob("*.wav"))
    if not wav_files:
        continue
    sampled = random.sample(wav_files, min(FILES_PER_CLASS, len(wav_files)))
    for f in sampled:
        actual = "M-Family" if folder.name in ["M16", "M4"] else folder.name
        test_files.append((f, folder.name, actual))

# ── 7. RUN ALL MODELS ON SAME FILES ─────────────────────────
model_names = list(models.keys())
results     = []

for file_path, folder, actual in test_files:
    mfcc = get_mfcc(file_path)
    row  = {"file": file_path.name, "folder": folder, "actual": actual}
    for name, model in models.items():
        pred, conf = predict(model, mfcc)
        row[name]  = (pred, conf, pred == actual)
    results.append(row)

# ── 8. PRINT PER-MODEL RESULTS ───────────────────────────────
for name in model_names:
    correct = sum(1 for r in results if r[name][2])
    total   = len(results)
    print(f"\n{'='*90}")
    print(f"  {name}  —  {correct}/{total}  ({100*correct/total:.1f}%)")
    print(f"{'='*90}")
    print(f"  {'FILE':<25} {'FOLDER':<18} {'ACTUAL':<18} {'PREDICTED':<22} {'CONF':>7}  ✓/✗")
    print(f"  {'-'*88}")
    for r in results:
        pred, conf, ok = r[name]
        status = "✓" if ok else "✗"
        print(f"  {r['file']:<25} {r['folder']:<18} {r['actual']:<18} {pred:<22} {conf:>6.1f}%  {status}")

# ── 9. FINAL COMPARISON TABLE ────────────────────────────────
print(f"\n\n{'='*55}")
print(f"  FINAL COMPARISON — {FILES_PER_CLASS} files per class ({len(test_files)} total)")
print(f"{'='*55}")
print(f"  {'Model':<22} {'Correct':>8} {'Accuracy':>10}")
print(f"  {'-'*42}")

for name in model_names:
    correct = sum(1 for r in results if r[name][2])
    total   = len(results)
    print(f"  {name:<22} {correct:>5}/{total:<3} {100*correct/total:>9.1f}%")

print(f"{'='*55}\n")