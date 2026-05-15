import os, pickle, tempfile, wave as wave_module
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T

# ── 1. LOAD CONFIG & MAPPING ────────────────────────────────
with open("preprocessing_config.pkl", "rb") as f:
    cfg = pickle.load(f)

with open("class_mapping.pkl", "rb") as f:
    mapping = pickle.load(f)

idx_to_class = mapping["idx_to_class"]
num_classes  = len(idx_to_class)

print(f"[+] Config loaded: sr={cfg['target_sr']}, n_mfcc={cfg['n_mfcc']}, n_fft={cfg['n_fft']}")
print(f"[+] Classes: {idx_to_class}")

# ── 2. MODEL ARCHITECTURE ───────────────────────────────────
class GunshotCNN(nn.Module):
    def __init__(self, num_classes=8):
        super(GunshotCNN, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
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

# ── 3. LOAD MODEL ───────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = GunshotCNN(num_classes).to(device)
model.load_state_dict(torch.load("best of the best.pth", map_location=device, weights_only=True))
model.eval()
print(f"[+] Model loaded on {device}")

# ── 4. MFCC TRANSFORM (matches training exactly) ────────────
mfcc_transform = T.MFCC(
    sample_rate=cfg["target_sr"],
    n_mfcc=cfg["n_mfcc"],
    melkwargs={
        "n_fft":      cfg["n_fft"],
        "hop_length": cfg["hop_length"],
        "n_mels":     cfg["n_mels"]
    }
)

# ── 5. LOAD WAV ──────────────────────────────────────────────
def load_wav(file_path):
    with wave_module.open(str(file_path), 'rb') as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        b  = wf.readframes(wf.getnframes())
        w  = torch.from_numpy(np.frombuffer(b, dtype=np.int16).copy()).float()
        w  = w.view(-1, ch).T / 32768.0
    return w, sr

# ── 6. PREPROCESS (mirrors training pipeline exactly) ────────
def preprocess(waveform, sr):
    # Mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    # Resample
    if sr != cfg["target_sr"]:
        waveform = T.Resample(sr, cfg["target_sr"])(waveform)
    # Pad / crop
    target_samples = cfg["target_sr"] * cfg["target_sec"]
    if waveform.shape[1] > target_samples:
        waveform = waveform[:, :target_samples]
    else:
        waveform = F.pad(waveform, (0, target_samples - waveform.shape[1]))
    # Pre-emphasis
    coeff    = cfg["pre_emphasis_coeff"]
    waveform = torch.cat((waveform[:, 0:1], waveform[:, 1:] - coeff * waveform[:, :-1]), dim=1)
    # Normalize
    waveform = waveform / (waveform.abs().max() + 1e-8)
    return waveform

# ── 7. PREDICT ───────────────────────────────────────────────
def predict(file_path):
    waveform, sr = load_wav(file_path)
    waveform     = preprocess(waveform, sr)
    mfcc         = mfcc_transform(waveform).unsqueeze(0).to(device)

    with torch.no_grad():
        prob      = F.softmax(model(mfcc), dim=1)
        conf, idx = torch.max(prob, 1)

    label      = idx_to_class[idx.item()]
    confidence = conf.item() * 100
    return ("Unknown / Uncertain", confidence) if confidence < 40.0 else (label, confidence)

# ── 8. MENU ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*45)
    print("     GUNSHOT CLASSIFICATION SYSTEM")
    print("      [ better_cnn_model ]")
    print("="*45)
    print(" [R]  Record from microphone (2 seconds)")
    print(" [U]  Upload a .wav file path")
    choice = input("\nSelect option (R/U): ").strip().upper()

    label, confidence = None, 0

    if choice == "R":
        import sounddevice as sd
        from scipy.io.wavfile import write as wav_write

        fs, seconds = cfg["target_sr"], cfg["target_sec"]
        print(f"\n[!] Recording for {seconds}s... make some noise!")
        recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype="int16")
        sd.wait()
        print("[+] Done recording.")

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        wav_write(tmp.name, fs, recording)
        label, confidence = predict(tmp.name)
        os.unlink(tmp.name)

    elif choice == "U":
        file_path = input("Enter path to .wav file: ").strip().strip("'\"")
        if not os.path.exists(file_path):
            print(f"[!] File not found: {file_path}")
        else:
            label, confidence = predict(file_path)

    else:
        print("[!] Invalid choice.")

    if label:
        print(f"\n{'='*45}")
        print(f"  IDENTIFIED : {label}")
        print(f"  CONFIDENCE : {confidence:.2f}%")
        print(f"{'='*45}\n")