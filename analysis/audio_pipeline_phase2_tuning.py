"""
SensEat Phase 2 — MFCC vs GFCC Tuning Pipeline
=================================================
Tests MFCC and GFCC variations (10, 20, 40, 60, 80, 100 coeffs) 
using the CNN to find the fairest comparison to STFT (64x64).
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
import pywt
import cv2
from spafe.features.gfcc import gfcc

from scipy.signal import butter, lfilter
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
EATING_FOLDERS = ["001", "009", "020"]
IDLE_FOLDERS   = ["idle-gp"]  # Pixel data only
IRB_RE = re.compile(r"^(\d+)_(\d{3})_(\d{2})_(\d{2})$")

# Number of coefficients to test
COEFF_SIZES = [20, 40, 60, 80, 100]
FIXED_FRAMES = 64

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
    signal = load_pcm_left_channel(filepath)
    signal = signal[DIRECT_PATH_SAMPLES:]
    signal = butter_bandpass_filter(signal, LOWCUT, HIGHCUT, SAMPLE_RATE, FILTER_ORDER)
    chunks = []
    for i in range(0, len(signal), SAMPLES_PER_CHUNK):
        seg = signal[i : i + SAMPLES_PER_CHUNK]
        if len(seg) == SAMPLES_PER_CHUNK:
            chunks.append(seg)
    return chunks

# ─────────────────────────────────────────
# 2D FEATURE EXTRACION
# ─────────────────────────────────────────
def get_mfcc_fixed(segment, n_mfcc, fixed_frames=FIXED_FRAMES):
    hop = max(64, int(np.floor(len(segment) / (fixed_frames - 1))))
    mfcc_feat = librosa.feature.mfcc(y=segment, sr=SAMPLE_RATE, n_mfcc=n_mfcc, hop_length=hop)
    if mfcc_feat.shape[1] < fixed_frames:
        mfcc_feat = np.pad(mfcc_feat, ((0, 0), (0, fixed_frames - mfcc_feat.shape[1])),
                      mode='constant', constant_values=(mfcc_feat.min(),))
    elif mfcc_feat.shape[1] > fixed_frames:
        mfcc_feat = mfcc_feat[:, :fixed_frames]
    return mfcc_feat.astype(np.float32)[..., np.newaxis]

def get_gfcc_fixed(segment, n_gfcc, fixed_frames=FIXED_FRAMES):
    try:
        g = gfcc(segment, fs=SAMPLE_RATE, num_ceps=n_gfcc)
        g = g.T # Transpose to (num_ceps, num_frames) to match MFCC
        if g.shape[1] < fixed_frames:
            g = np.pad(g, ((0, 0), (0, fixed_frames - g.shape[1])),
                       mode='constant', constant_values=(g.min(),))
        elif g.shape[1] > fixed_frames:
            g = g[:, :fixed_frames]
        return g.astype(np.float32)[..., np.newaxis]
    except Exception as e:
        return np.zeros((n_gfcc, fixed_frames, 1), dtype=np.float32)

def get_stft_image(segment, n_fft=2048, fixed_size=(64, 64)):
    try:
        D = np.abs(librosa.stft(segment, n_fft=n_fft))
        S_db = librosa.amplitude_to_db(D, ref=np.max)
        S_db = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)
        resized = cv2.resize(S_db, fixed_size, interpolation=cv2.INTER_CUBIC)
        return resized.astype(np.float32)[..., np.newaxis]
    except Exception:
        return np.zeros((*fixed_size, 1), dtype=np.float32)

# ─────────────────────────────────────────
# DATASET BUILDER
# ─────────────────────────────────────────
def build_dataset_tuning():
    features_dict = {"STFT": []}
    for n in COEFF_SIZES:
        features_dict[f"MFCC_{n}"] = []
        features_dict[f"GFCC_{n}"] = []

    labels = []

    # Eating
    for folder_name in EATING_FOLDERS:
        folder_path = RAW_DATA / folder_name
        if not folder_path.exists(): continue
        pcm_files = sorted([f for f in folder_path.glob("*.pcm") if not f.stem.endswith("_idleTail") and not f.stem.endswith("_meta")])
        print(f"  📂 {folder_name}: {len(pcm_files)} eating files")
        for pcm in pcm_files:
            chunks = preprocess_and_segment(str(pcm))
            for seg in chunks:
                features_dict["STFT"].append(get_stft_image(seg))
                for n in COEFF_SIZES:
                    features_dict[f"MFCC_{n}"].append(get_mfcc_fixed(seg, n))
                    features_dict[f"GFCC_{n}"].append(get_gfcc_fixed(seg, n))
                labels.append(1)

    # Idle
    for idle_folder_name in IDLE_FOLDERS:
        idle_path = RAW_DATA / idle_folder_name
        if not idle_path.exists(): continue
        idle_files = sorted([f for f in idle_path.glob("*.pcm") if not f.stem.endswith("_idleTail") and not f.stem.endswith("_meta")])
        print(f"  📂 {idle_folder_name}: {len(idle_files)} idle files")
        for pcm in idle_files:
            chunks = preprocess_and_segment(str(pcm))
            for seg in chunks:
                features_dict["STFT"].append(get_stft_image(seg))
                for n in COEFF_SIZES:
                    features_dict[f"MFCC_{n}"].append(get_mfcc_fixed(seg, n))
                    features_dict[f"GFCC_{n}"].append(get_gfcc_fixed(seg, n))
                labels.append(0)

    for k in features_dict:
        features_dict[k] = np.array(features_dict[k], dtype=np.float32)
    return features_dict, np.array(labels)

# ─────────────────────────────────────────
# MODEL TRAINING
# ─────────────────────────────────────────
def build_simple_cnn(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(16, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_cnn_holdout(X, y, input_shape, epochs=20, batch_size=16):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    mean = X_tr.mean()
    std  = X_tr.std() if X_tr.std() > 0 else 1.0
    X_tr = (X_tr - mean) / std
    X_te = (X_te - mean) / std

    model = build_simple_cnn(input_shape)
    es = EarlyStopping(patience=3, restore_best_weights=True)
    model.fit(X_tr, y_tr, epochs=epochs, batch_size=batch_size, validation_split=0.1, callbacks=[es], verbose=0)
    
    y_pred = (model.predict(X_te, verbose=0).ravel() > 0.5).astype(int)
    tf.keras.backend.clear_session()
    return f1_score(y_te, y_pred, zero_division=0)

def main():
    print("=" * 70)
    print("  SensEat — Phase 2: MFCC vs GFCC Parameter Tuning")
    print("=" * 70)

    print("\n📦 Loading dataset ...")
    X_dict, y = build_dataset_tuning()
    
    # Stratified 5-Fold setup for CV testing
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    
    results = []

    print(f"\n🧠 Evaluating combinations (Total: {len(y)} segments)...")
    for feat_name, X in X_dict.items():
        print(f"  Testing {feat_name:10s} | Shape: {X.shape}")
        
        # We will just do a quick 80/20 split test to save time, CV takes too long for tuning 11 models
        f1_holdout = train_cnn_holdout(X, y, input_shape=X[0].shape)
        results.append({'Feature': feat_name, 'Shape': f"{X.shape[1]}x{X.shape[2]}", 'Holdout_F1': round(f1_holdout, 4)})
        print(f"    --> F1 Score: {f1_holdout:.4f}")

    df = pd.DataFrame(results).sort_values('Holdout_F1', ascending=False)
    out_csv = SCRIPT_DIR / "phase2_tuning_results.csv"
    df.to_csv(out_csv, index=False)
    
    print("\n" + "=" * 70)
    print("  RANKED TUNING RESULTS")
    print("=" * 70)
    print(df.to_string(index=False))
    print(f"\n💾 Results saved to {out_csv}")

if __name__ == '__main__':
    main()
