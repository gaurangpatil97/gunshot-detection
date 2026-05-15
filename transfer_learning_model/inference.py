"""
inference.py — Dual model gunshot classification
Runs both BetterCNN and DeepGunshotCNN on the same input
and prints both predictions side by side.

Place this file in: transfer_learning_model/inference.py

Requirements:
    ../better_cnn_model/best of the best.pth
    ../better_cnn_model/preprocessing_config.pkl
    ../better_cnn_model/class_mapping.pkl
    best_gunshot_deep_cnn.pth
    preprocessing_config.pkl
    class_mapping.pkl

Usage:
    python inference.py
"""

import os, pickle, tempfile, wave as wave_module
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T

# ── CONFIG ───────────────────────────────────────────────────
BETTER_CNN_DIR   = Path("../better_cnn_model")
DEEP_CNN_DIR     = Path(".")
CONFIDENCE_THRESHOLD = 40.0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── 1. LOAD CONFIGS ──────────────────────────────────────────
with open(BETTER_CNN_DIR / "preprocessing_config.pkl", "rb") as f:
    cfg_better = pickle.load(f)
with open(BETTER_CNN_DIR / "class_mapping.pkl", "rb") as f:
    mapping_better = pickle.load(f)

with open(DEEP_CNN_DIR / "preprocessing_config.pkl", "rb") as f:
    cfg_deep = pickle.load(f)
with open(DEEP_CNN_DIR / "class_mapping.pkl", "rb") as f:
    mapping_deep = pickle.load(f)

idx_to_class_better = mapping_better["idx_to_class"]
idx_to_class_deep   = mapping_deep["idx_to_class"]

# ── 2. MODEL ARCHITECTURES ───────────────────────────────────
class BetterCNN(nn.Module):
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
        self.dropout = nn.Dropout(0.4)
        self.fc1     = nn.Linear(128 * 4 * 4, 256)
        self.fc2     = nn.Linear(256, num_classes)
        self.relu    = nn.ReLU()

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


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

# ── 3. LOAD MODELS ───────────────────────────────────────────
better_model = BetterCNN(num_classes=8).to(device)
better_model.load_state_dict(torch.load(
    BETTER_CNN_DIR / "best of the best.pth", map_location=device, weights_only=True
))
better_model.eval()

deep_model = DeepGunshotCNN(num_classes=8).to(device)
deep_model.load_state_dict(torch.load(
    DEEP_CNN_DIR / "best_gunshot_deep_cnn.pth", map_location=device, weights_only=True
))
deep_model.eval()

print(f"[+] Both models loaded on {device}")

# ── 4. MFCC TRANSFORMS ──────────────────────────────────────
def make_mfcc_transform(cfg):
    return T.MFCC(
        sample_rate=cfg["target_sr"],
        n_mfcc=cfg["n_mfcc"],
        melkwargs={"n_fft": cfg["n_fft"], "hop_length": cfg["hop_length"], "n_mels": cfg["n_mels"]}
    )

mfcc_better = make_mfcc_transform(cfg_better)
mfcc_deep   = make_mfcc_transform(cfg_deep)

# ── 5. LOAD WAV ──────────────────────────────────────────────
def load_wav(file_path):
    with wave_module.open(str(file_path), 'rb') as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        b  = wf.readframes(wf.getnframes())
        w  = torch.from_numpy(np.frombuffer(b, dtype=np.int16).copy()).float()
        w  = w.view(-1, ch).T / 32768.0
    return w, sr

# ── 6. PREPROCESS ────────────────────────────────────────────
def preprocess(waveform, sr, cfg):
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

# ── 7. PREDICT ───────────────────────────────────────────────
def predict(file_path, model, cfg, mfcc_transform, idx_to_class):
    waveform, sr = load_wav(file_path)
    waveform     = preprocess(waveform, sr, cfg)
    mfcc         = mfcc_transform(waveform).unsqueeze(0).to(device)
    with torch.no_grad():
        prob      = F.softmax(model(mfcc), dim=1)
        conf, idx = torch.max(prob, 1)
    label      = idx_to_class[idx.item()]
    confidence = conf.item() * 100
    return ("Unknown / Uncertain", confidence) if confidence < CONFIDENCE_THRESHOLD else (label, confidence)

def run_both(file_path):
    b_label, b_conf = predict(file_path, better_model, cfg_better, mfcc_better, idx_to_class_better)
    d_label, d_conf = predict(file_path, deep_model,   cfg_deep,   mfcc_deep,   idx_to_class_deep)
    return b_label, b_conf, d_label, d_conf

def print_results(b_label, b_conf, d_label, d_conf):
    print(f"\n{'='*50}")
    print(f"  GUNSHOT CLASSIFICATION RESULTS")
    print(f"{'='*50}")
    print(f"  {'Model':<22} {'Prediction':<18} {'Confidence':>10}")
    print(f"  {'-'*48}")
    print(f"  {'BetterCNN':<22} {b_label:<18} {b_conf:>9.2f}%")
    print(f"  {'DeepGunshotCNN':<22} {d_label:<18} {d_conf:>9.2f}%")
    print(f"  {'-'*48}")

    if b_label == d_label and b_label != "Unknown / Uncertain":
        print(f"  ✓ CONSENSUS: {b_label}")
    elif b_label == "Unknown / Uncertain" and d_label == "Unknown / Uncertain":
        print(f"  ✗ Both models uncertain — try a cleaner recording")
    else:
        if b_conf >= d_conf:
            print(f"  ~ No consensus — BetterCNN favours: {b_label} ({b_conf:.1f}%)")
        else:
            print(f"  ~ No consensus — DeepGunshotCNN favours: {d_label} ({d_conf:.1f}%)")
    print(f"{'='*50}\n")

# ── 8. MENU ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*50)
    print("     GUNSHOT CLASSIFICATION SYSTEM")
    print("       BetterCNN + DeepGunshotCNN")
    print("="*50)
    print(" [R]  Record from microphone (2 seconds)")
    print(" [U]  Upload a .wav file path")
    choice = input("\nSelect option (R/U): ").strip().upper()

    if choice == "R":
        import sounddevice as sd
        from scipy.io.wavfile import write as wav_write

        fs      = cfg_better["target_sr"]
        seconds = cfg_better["target_sec"]
        print(f"\n[!] Recording for {seconds}s... make some noise!")
        recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype="int16")
        sd.wait()
        print("[+] Done recording.")

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        wav_write(tmp.name, fs, recording)
        b_label, b_conf, d_label, d_conf = run_both(tmp.name)
        os.unlink(tmp.name)
        print_results(b_label, b_conf, d_label, d_conf)

    elif choice == "U":
        file_path = input("Enter path to .wav file: ").strip().strip("'\"")
        if not os.path.exists(file_path):
            print(f"[!] File not found: {file_path}")
        else:
            b_label, b_conf, d_label, d_conf = run_both(file_path)
            print_results(b_label, b_conf, d_label, d_conf)

    else:
        print("[!] Invalid choice.")