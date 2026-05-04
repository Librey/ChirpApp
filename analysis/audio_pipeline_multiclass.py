"""
SensEat 1.5s Chirp-Level Multi-class Classification Pipeline
============================================================
Phase 2: Multi-class classification — Predicting specific food items.
Loads EATING data from 20 participants. 
Uses best features from Phase 1/Tuning:
- STFT (64x64)
- MFCC (20 coeffs -> 20x64)
"""

import os
import re
import random
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import librosa
import cv2
from scipy.signal import butter, lfilter
from scipy.fft import dct as scipy_dct
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
SAMPLE_RATE    = 44100
LOWCUT         = 17500.0
HIGHCUT        = 20500.0
FILTER_ORDER   = 6
CHIRP_PERIOD_S = 1.5
SAMPLES_PER_CHUNK = int(SAMPLE_RATE * CHIRP_PERIOD_S)  # 66 150
DIRECT_PATH_SAMPLES = int((0.30 / 343.0) * SAMPLE_RATE)  # ~38

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

SCRIPT_DIR = Path(__file__).resolve().parent
RAW_DATA   = SCRIPT_DIR / "raw_data"

# All 20 participants
EATING_FOLDERS = [f"{i:03d}" for i in range(1, 21)]

# IRB filename pattern: W_XXX_YY_ZZ
# YY is the food code
IRB_RE = re.compile(r"^(\d+)_(\d{3})_(\d{2})_(\d{2})$")

FOOD_NAMES = {
    1:  "Tortilla",
    2:  "Mandarin",
    3:  "Chicken_Breast",
    4:  "Cheeze_It",
    5:  "Carrots",
    6:  "Chocolate",
    7:  "Yogurt",
    8:  "Noodles",
    9:  "Water",
    10: "Coke"
}

# ─────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────
def butter_bandpass_filter(data, lowcut, highcut, fs, order=6):
    nyquist = 0.5 * fs
    low  = lowcut / nyquist
    high = highcut / nyquist
    if low <= 0 or high >= 1 or low >= high: return data
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)

def load_pcm_left_channel(filepath):
    raw = np.fromfile(filepath, dtype=np.int16)
    if raw.size % 2 != 0: raw = raw[:-1]
    stereo = raw.reshape(-1, 2)
    return stereo[:, 0].astype(np.float32) / 32768.0

def preprocess_and_segment(filepath):
    """Filter to 17.5-20.5kHz and split into 1.5s non-overlapping chunks."""
    signal = load_pcm_left_channel(filepath)
    signal = signal[DIRECT_PATH_SAMPLES:]
    signal = butter_bandpass_filter(signal, LOWCUT, HIGHCUT, SAMPLE_RATE, FILTER_ORDER)
    
    chunks = []
    # Peak Normalization (To prevent magnitude variations from affecting model)
    HOP_SAMPLES = int(SAMPLE_RATE * 0.5) 
    for i in range(0, len(signal) - SAMPLES_PER_CHUNK + 1, HOP_SAMPLES):
        seg = signal[i : i + SAMPLES_PER_CHUNK]
        if len(seg) == SAMPLES_PER_CHUNK:
            # Normalize peak to 1.0
            max_val = np.max(np.abs(seg))
            if max_val > 0:
                seg = seg / max_val
            chunks.append(seg)
    return chunks

# ─────────────────────────────────────────
# FEATURE EXTRACION 
# ─────────────────────────────────────────
def get_stft_image(segment, n_fft=2048, fixed_size=(64, 64)):
    """64×64×1 STFT spectrogram image."""
    try:
        D = np.abs(librosa.stft(segment, n_fft=n_fft))
        S_db = librosa.amplitude_to_db(D, ref=np.max)
        S_db = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)
        resized = cv2.resize(S_db, fixed_size, interpolation=cv2.INTER_CUBIC)
        return resized.astype(np.float32)[..., np.newaxis]
    except Exception:
        return np.zeros((*fixed_size, 1), dtype=np.float32)

def get_mfcc_fixed(segment, sr=SAMPLE_RATE, n_mfcc=80, fixed_frames=64):
    """80×64×1 MFCC image (Increased detail from Phase 1)."""
    hop = max(64, int(np.floor(len(segment) / (fixed_frames - 1))))
    mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=n_mfcc, hop_length=hop)
    if mfcc.shape[1] < fixed_frames:
        mfcc = np.pad(mfcc, ((0, 0), (0, fixed_frames - mfcc.shape[1])),
                      mode='constant', constant_values=(mfcc.min(),))
    elif mfcc.shape[1] > fixed_frames:
        mfcc = mfcc[:, :fixed_frames]
    
    # Optional MinMax scale to match STFT bounds
    mfcc_norm = (mfcc - mfcc.min()) / (mfcc.max() - mfcc.min() + 1e-8)
    return mfcc_norm.astype(np.float32)[..., np.newaxis]

def get_gfcc_fixed(segment, sr=SAMPLE_RATE, num_ceps=20, fixed_frames=64):
    """20x64x1 Gammatone Frequency Cepstral Coefficient Image."""
    # spafe expects window lengths rather than hop samples
    # We set frame length and step to roughly match 64 frames over 1.5s (step ~ 23ms)
    try:
        gfccs = gfcc_feature.gfcc(segment, fs=sr, num_ceps=num_ceps,
                                  nfilts=64, win_len=0.046, win_hop=0.023)
        # spafe returns (frames, coeffs), so we transpose to (coeffs, frames)
        gfccs = gfccs.T
        if gfccs.shape[1] < fixed_frames:
            gfccs = np.pad(gfccs, ((0, 0), (0, fixed_frames - gfccs.shape[1])),
                           mode='constant', constant_values=(gfccs.min(),))
        elif gfccs.shape[1] > fixed_frames:
            gfccs = gfccs[:, :fixed_frames]
            
        gfccs_norm = (gfccs - gfccs.min()) / (gfccs.max() - gfccs.min() + 1e-8)
        return gfccs_norm.astype(np.float32)[..., np.newaxis]
    except Exception:
        return np.zeros((num_ceps, fixed_frames, 1), dtype=np.float32)

# ─────────────────────────────────────────
# DATASET BUILDER
# ─────────────────────────────────────────
def build_multiclass_dataset():
    feat_stft = []
    feat_mfcc = []
    feat_gfcc = []
    labels    = []
    
    print("\n📦 Loading and segmenting Multi-Class EATING files (T = 1.5s) ...")

    for folder_name in EATING_FOLDERS:
        folder_path = RAW_DATA / folder_name
        if not folder_path.exists():
            print(f"  ⚠ Eating folder not found: {folder_path} (Skipping)")
            continue

        pcm_files = sorted([
            f for f in folder_path.glob("*.pcm")
            if not f.stem.endswith("_idleTail") and not f.stem.endswith("_meta")
        ])
        if len(pcm_files) == 0:
            continue
            
        print(f"  📂 Participant {folder_name}: {len(pcm_files)} files processed")

        for pcm in pcm_files:
            m = IRB_RE.match(pcm.stem)
            if not m:
                continue
            
            food_code = int(m.group(3))
            if food_code not in FOOD_NAMES:
                continue # Skip invalid codes

            chunks = preprocess_and_segment(str(pcm))
            for seg in chunks:
                feat_stft.append(get_stft_image(seg))
                feat_mfcc.append(get_mfcc_fixed(seg))
                feat_gfcc.append(get_gfcc_fixed(seg))
                labels.append(food_code)

    return np.array(feat_stft), np.array(feat_mfcc), np.array(feat_gfcc), np.array(labels)

# ─────────────────────────────────────────
# MODEL TRAINING (MULTI-CLASS CNN)
# ─────────────────────────────────────────
def build_multiclass_cnn(input_shape, num_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def plot_confusion_matrix(y_true, y_pred, feature_name):
    unique_labels = sorted(list(set(y_true)))
    display_labels = [FOOD_NAMES[k] for k in unique_labels]
    
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_percent, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=display_labels, yticklabels=display_labels)
    plt.title(f"Confusion Matrix (Accuracy %): CNN + {feature_name}")
    plt.ylabel('True Food Type')
    plt.xlabel('Predicted Food Type')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    out_dir = SCRIPT_DIR / "figures"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"multiclass_confusion_{feature_name}.png"
    plt.savefig(out_path, dpi=300)
    print(f"    [+] Saved Phase 2 Confusion Matrix: {out_path.name}")
    plt.close()

def train_and_eval(X, y, feature_name):
    # Relabel labels from food codes (1-11) to zero-indexed classes (0-10) for softmax
    unique_codes = sorted(list(set(y)))
    code_to_idx = {code: idx for idx, code in enumerate(unique_codes)}
    idx_to_code = {idx: code for idx, code in enumerate(unique_codes)}
    
    y_mapped = np.array([code_to_idx[code] for code in y])
    num_classes = len(unique_codes)
    
    # 80/20 Holdout Split for final validation and confusion matrix
    X_tr, X_te, y_tr, y_te = train_test_split(X, y_mapped, test_size=0.2, random_state=SEED, stratify=y_mapped)
    
    # Normalize features
    mean = X_tr.mean()
    std  = X_tr.std() if X_tr.std() > 0 else 1.0
    X_tr = (X_tr - mean) / std
    X_te = (X_te - mean) / std

    model = build_multiclass_cnn(X[0].shape, num_classes)
    es = EarlyStopping(patience=5, restore_best_weights=True)
    
    print(f"\n  Training CNN on {feature_name} (Shape: {X[0].shape}, Classes: {num_classes})")
    model.fit(X_tr, y_tr, epochs=40, batch_size=32, validation_split=0.1, callbacks=[es], verbose=0)
    
    # Evaluate
    y_pred_probs = model.predict(X_te, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    acc = accuracy_score(y_te, y_pred)
    f1 = f1_score(y_te, y_pred, average='weighted', zero_division=0)
    
    print(f"    --> Test Accuracy : {acc:.4f}")
    print(f"    --> Test F1 Score : {f1:.4f}")
    
    # Plot Confusion Matrix (map back to real codes)
    y_te_original = [idx_to_code[idx] for idx in y_te]
    y_pred_original = [idx_to_code[idx] for idx in y_pred]
    plot_confusion_matrix(y_te_original, y_pred_original, feature_name)
    
    tf.keras.backend.clear_session()
    return acc, f1

def main():
    print("=" * 70)
    print("  SensEat — Phase 2: Multi-class Classification Pipeline")
    print("=" * 70)

    X_stft, X_mfcc, X_gfcc, y = build_multiclass_dataset()
    
    if len(y) == 0:
        print("❌ No valid segments found.")
        return
        
    print(f"\n✅ Total overlapping multi-class segments extracted: {len(y)}")
    
    results = []
    
    # 1. Pipeline Test: STFT
    acc_s, f1_s = train_and_eval(X_stft, y, "STFT")
    results.append({"Feature": "STFT (64x64)", "Accuracy": round(acc_s, 4), "F1 (Weighted)": round(f1_s, 4)})
    
    # 2. Pipeline Test: MFCC (80)
    acc_m, f1_m = train_and_eval(X_mfcc, y, "MFCC_80")
    results.append({"Feature": "MFCC (80x64)", "Accuracy": round(acc_m, 4), "F1 (Weighted)": round(f1_m, 4)})
    
    # 3. Pipeline Test: GFCC (20)
    acc_g, f1_g = train_and_eval(X_gfcc, y, "GFCC_20")
    results.append({"Feature": "GFCC (20x64)", "Accuracy": round(acc_g, 4), "F1 (Weighted)": round(f1_g, 4)})
    
    df = pd.DataFrame(results).sort_values('Accuracy', ascending=False)
    out_csv = SCRIPT_DIR / "pipeline_multiclass_results.csv"
    df.to_csv(out_csv, index=False)
    
    print("\n" + "=" * 70)
    print("  MULTI-CLASS RANKED RESULTS")
    print("=" * 70)
    print(df.to_string(index=False))
    print(f"\n💾 Results saved to {out_csv}")

if __name__ == '__main__':
    main()
