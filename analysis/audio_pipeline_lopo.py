"""
SensEat — Leave-One-Participant-Out (LOPO) Binary Classification Pipeline
==========================================================================
Participants 022-041 only (clean data).
Each fold: train on 19 participants, test on 1.

Idle handling (shared pool):
  70% of idle files -> always training
  30% of idle files -> always test
  No fake participant IDs.

Segmentation: 1.5s window, 0.5s hop (overlapping, matches multiclass pipeline).

Usage:
    python audio_pipeline_lopo.py
Output:
    pipeline_lopo_binary_results.csv
    pipeline_lopo_binary_foldwise.csv
    figures/binary/lopo_binary_cm_*.png
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

from scipy.signal import butter, lfilter, hilbert
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
SAMPLE_RATE   = 44100
LOWCUT        = 17500.0
HIGHCUT       = 20500.0
FILTER_ORDER  = 6

CHIRP_PERIOD_S    = 1.5
SAMPLES_PER_CHUNK = int(SAMPLE_RATE * CHIRP_PERIOD_S)   # 66150

HOP_S      = 0.5
HOP_SAMPLES = int(SAMPLE_RATE * HOP_S)

DIRECT_PATH_SAMPLES = int((0.30 / 343.0) * SAMPLE_RATE) # ~38

# Reference chirp parameters (from Android app — setting 2)
CHIRP_START_HZ    = 18000.0
CHIRP_END_HZ      = 20000.0
CHIRP_DUR_SAMPLES = int(SAMPLE_RATE * 1.0)   # 1000ms chirp = 44100 samples

# Target echo range: 10–50 cm round-trip
SOUND_SPEED    = 343.0
TARGET_TAP_MIN = int(2 * 0.10 / SOUND_SPEED * SAMPLE_RATE)  # ~26
TARGET_TAP_MAX = int(2 * 0.50 / SOUND_SPEED * SAMPLE_RATE)  # ~129

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR   = SCRIPT_DIR / "data"
IDLE_DIR   = SCRIPT_DIR / "idle"
FIG_DIR    = SCRIPT_DIR / "figures" / "binary"
FIG_DIR.mkdir(parents=True, exist_ok=True)

EATING_FOLDERS = [f"{i:03d}" for i in range(22, 42)]

IDLE_TRAIN_GROUP = -1   # 70% idle -> always in training
IDLE_TEST_GROUP  = -2   # 30% idle -> always in test
IDLE_TRAIN_FRAC  = 0.70

# N_SPLITS controls cross-validation:
#   20 = full LOPO (1 participant per fold, gold standard, slow)
#    5 = 5-fold group (4 participants per fold, 4x faster, still person-independent)
N_SPLITS = 5

# Set to False to skip CNN (CNN is slow — classical ML F1=0.86 is already good)
RUN_CNN = False

IRB_RE = re.compile(r"^(\d+)_(\d{3})_(\d{2})_(\d{2})(?: \(\d+\))?$")


# ─────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────
def butter_bandpass(data):
    nyq  = 0.5 * SAMPLE_RATE
    low  = LOWCUT  / nyq
    high = HIGHCUT / nyq
    if low <= 0 or high >= 1 or low >= high:
        return data
    b, a = butter(FILTER_ORDER, [low, high], btype="band")
    return lfilter(b, a, data)


def generate_reference_chirp():
    """Synthesize the transmitted chirp exactly as the Android app does (setting 2)."""
    n     = CHIRP_DUR_SAMPLES
    dur_s = n / SAMPLE_RATE
    chirp = np.zeros(n, dtype=np.float32)
    for i in range(n):
        t        = i / SAMPLE_RATE
        freq     = CHIRP_START_HZ + (CHIRP_END_HZ - CHIRP_START_HZ) * (t / dur_s)
        chirp[i] = np.sin(2 * np.pi * freq * t)
    return butter_bandpass(chirp)   # bandpass to match received signal


def tap_profile_features(segment, ref_chirp):
    """FFT cross-correlation tap profile — echo strength vs distance (10–50 cm)."""
    rx  = segment[:CHIRP_DUR_SAMPLES].astype(np.float64)
    ref = ref_chirp.astype(np.float64)

    Nfft = 1 << (len(rx) + len(ref) - 1).bit_length()
    taps = np.abs(np.fft.ifft(np.fft.fft(rx, Nfft) * np.conj(np.fft.fft(ref, Nfft))))

    target = taps[TARGET_TAP_MIN:TARGET_TAP_MAX].astype(np.float32)
    peak   = target.max()
    if peak > 0:
        target = target / peak   # normalize to [0,1]

    stats = np.array([
        float(np.max(target)),
        float(np.mean(target)),
        float(np.std(target)),
        float(np.argmax(target)),          # peak echo distance bin
        float(np.sum(target)),             # total echo energy in range
        float(np.percentile(target, 75)),
        float(np.percentile(target, 25)),
    ], dtype=np.float32)

    return np.concatenate([target, stats])  # ~103 + 7 = 110 values


def load_pcm(filepath):
    raw = np.fromfile(filepath, dtype=np.int16)
    if raw.size % 2 != 0:
        raw = raw[:-1]
    stereo = raw.reshape(-1, 2)
    return stereo[:, 0].astype(np.float32) / 32768.0


def compute_reference_direct_path(idle_files, max_files=5):
    """
    Average several idle segments to get a stable reference direct path.
    Idle = person present but not eating → signal is mostly direct path + room.
    """
    all_segs = []
    for f in idle_files[:max_files]:
        sig = butter_bandpass(load_pcm(str(f)))
        for i in range(0, len(sig) - SAMPLES_PER_CHUNK + 1, HOP_SAMPLES):
            seg = sig[i : i + SAMPLES_PER_CHUNK]
            if len(seg) == SAMPLES_PER_CHUNK:
                all_segs.append(seg.astype(np.float64))
    if not all_segs:
        return None
    return np.mean(all_segs, axis=0).astype(np.float32)


def cancel_direct_path(segment, ref_direct_path):
    """
    AIM Stage 1: AGC-scaled direct path subtraction.
    Scaling coefficient c = (s · d) / (d · d) minimises ||s - c*d||²
    Accounts for phone-hold variation and AGC differences between participants.
    """
    s = segment.astype(np.float64)
    d = ref_direct_path.astype(np.float64)
    c = np.dot(s, d) / (np.dot(d, d) + 1e-8)
    return (s - c * d).astype(np.float32)


def preprocess_and_segment(filepath, ref_direct_path=None):
    signal = load_pcm(filepath)
    signal = butter_bandpass(signal)
    segments = []
    for i in range(0, len(signal) - SAMPLES_PER_CHUNK + 1, HOP_SAMPLES):
        seg = signal[i : i + SAMPLES_PER_CHUNK]
        if len(seg) != SAMPLES_PER_CHUNK:
            continue
        # Direct path removal: basic crop
        seg = seg[DIRECT_PATH_SAMPLES:]
        seg = np.pad(seg, (0, DIRECT_PATH_SAMPLES))
        # AIM Stage 1: AGC-scaled direct path cancellation
        if ref_direct_path is not None:
            seg = cancel_direct_path(seg, ref_direct_path)
        segments.append(seg)
    return segments


# ─────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────
def stat_features(x):
    zcr = librosa.feature.zero_crossing_rate(x)[0]
    env = np.abs(hilbert(x))
    return np.array([
        np.mean(x),   np.std(x),
        np.max(x),    np.min(x),
        np.mean(x**2),
        np.mean(zcr),
        np.mean(env), np.std(env), np.max(env),
    ], dtype=np.float32)


def spectral_features(x):
    S = np.abs(librosa.stft(x, n_fft=2048))
    freqs = librosa.fft_frequencies(sr=SAMPLE_RATE, n_fft=2048)
    centroid  = librosa.feature.spectral_centroid(S=S,  sr=SAMPLE_RATE)[0]
    bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=SAMPLE_RATE)[0]
    rolloff   = librosa.feature.spectral_rolloff(S=S,   sr=SAMPLE_RATE)[0]
    flux      = librosa.onset.onset_strength(S=librosa.amplitude_to_db(S), sr=SAMPLE_RATE)
    # Energy in ultrasonic band only
    band_mask = (freqs >= LOWCUT) & (freqs <= HIGHCUT)
    band_energy = np.mean(S[band_mask, :] ** 2)
    return np.array([
        np.mean(centroid),  np.std(centroid),
        np.mean(bandwidth), np.std(bandwidth),
        np.mean(rolloff),   np.std(rolloff),
        np.mean(flux),      np.std(flux),
        band_energy,
    ], dtype=np.float32)


def wavelet_features(x):
    try:
        coeffs = pywt.wavedec(x, "db4", level=4)
    except Exception:
        coeffs = pywt.wavedec(x, "db4", level=1)
    feats = []
    for c in coeffs[1:]:
        feats.extend([np.mean(c), np.std(c), np.max(c), np.min(c), np.mean(c**2)])
    return np.array(feats, dtype=np.float32)


def combined_flat(x, ref_chirp):
    return np.concatenate([
        stat_features(x),
        spectral_features(x),
        wavelet_features(x),
        tap_profile_features(x, ref_chirp),
    ])


def stft_image(x, fixed_size=(64, 64)):
    try:
        D = np.abs(librosa.stft(x, n_fft=2048))
        D = librosa.amplitude_to_db(D, ref=np.max)
        D = (D - D.min()) / (D.max() - D.min() + 1e-8)
        return cv2.resize(D, fixed_size)[..., np.newaxis].astype(np.float32)
    except Exception:
        return np.zeros((*fixed_size, 1), dtype=np.float32)


def mfcc_image(x, n_mfcc=40, fixed_frames=64):
    hop = max(64, int(len(x) / (fixed_frames - 1)))
    M   = librosa.feature.mfcc(y=x, sr=SAMPLE_RATE, n_mfcc=n_mfcc, hop_length=hop)
    if M.shape[1] < fixed_frames:
        M = np.pad(M, ((0, 0), (0, fixed_frames - M.shape[1])), constant_values=M.min())
    else:
        M = M[:, :fixed_frames]
    M = (M - M.min()) / (M.max() - M.min() + 1e-8)
    return M.astype(np.float32)[..., np.newaxis]


# ─────────────────────────────────────────
# DATASET BUILDER  (with disk cache)
# ─────────────────────────────────────────
CACHE_FILE = SCRIPT_DIR / "feature_cache_binary.npz"

def build_dataset():
    if CACHE_FILE.exists():
        print(f"  Loading cached features from {CACHE_FILE.name} ...")
        d = np.load(CACHE_FILE, allow_pickle=False)
        return d["X_flat"], d["X_stft"], d["X_mfcc"], d["y"], d["groups"]

    print("  No cache found — extracting features (will be cached for next run) ...")
    ref_chirp = generate_reference_chirp()
    print(f"  Reference chirp: {CHIRP_START_HZ/1000:.0f}–{CHIRP_END_HZ/1000:.0f} kHz  |  "
          f"Target taps {TARGET_TAP_MIN}–{TARGET_TAP_MAX} "
          f"({TARGET_TAP_MIN*SOUND_SPEED/2/SAMPLE_RATE*100:.0f}–"
          f"{TARGET_TAP_MAX*SOUND_SPEED/2/SAMPLE_RATE*100:.0f} cm)")

    # Build reference direct path from idle train pool (Stage 1 AIM)
    idle_files_all = sorted(IDLE_DIR.glob("*.pcm"))
    random.shuffle(idle_files_all)
    split = max(1, int(len(idle_files_all) * IDLE_TRAIN_FRAC))
    train_idle_files = idle_files_all[:split]
    test_idle_files  = idle_files_all[split:]

    ref_direct_path = compute_reference_direct_path(train_idle_files)
    if ref_direct_path is not None:
        print(f"  Reference direct path computed from {min(5, len(train_idle_files))} idle files")
    else:
        print("  Warning: could not compute reference direct path — skipping Stage 1")

    X_flat, X_stft, X_mfcc = [], [], []
    y, groups = [], []

    # Eating (label=1) + idleTail (label=0) per participant
    for folder in EATING_FOLDERS:
        path = DATA_DIR / folder
        if not path.exists():
            print(f"  ! Missing: {path}")
            continue

        pid = int(folder)

        eating_files = [
            f for f in path.glob("*.pcm")
            if IRB_RE.match(f.stem) and "_idleTail" not in f.stem
        ]
        idle_tail_files = [
            f for f in path.glob("*_idleTail.pcm")
            if f.stat().st_size > 0
        ]

        n_segs = 0
        for f in eating_files:
            for seg in preprocess_and_segment(str(f), ref_direct_path):
                X_flat.append(combined_flat(seg, ref_chirp))
                X_stft.append(stft_image(seg))
                X_mfcc.append(mfcc_image(seg))
                y.append(1)
                groups.append(pid)
                n_segs += 1

        for f in idle_tail_files:
            for seg in preprocess_and_segment(str(f), ref_direct_path):
                X_flat.append(combined_flat(seg, ref_chirp))
                X_stft.append(stft_image(seg))
                X_mfcc.append(mfcc_image(seg))
                y.append(0)
                groups.append(pid)
                n_segs += 1

        print(f"  Folder {folder}: {n_segs} segments "
              f"({len(eating_files)} eating files, {len(idle_tail_files)} idleTail)")

    # Shared idle pool — use pre-computed split from above
    print(f"\n  Idle pool: {len(idle_files_all)} files -> "
          f"{len(train_idle_files)} train, {len(test_idle_files)} test")

    for group_id, file_list in [(IDLE_TRAIN_GROUP, train_idle_files),
                                 (IDLE_TEST_GROUP,  test_idle_files)]:
        for f in file_list:
            for seg in preprocess_and_segment(str(f), ref_direct_path):
                X_flat.append(combined_flat(seg, ref_chirp))
                X_stft.append(stft_image(seg))
                X_mfcc.append(mfcc_image(seg))
                y.append(0)
                groups.append(group_id)

    X_flat  = np.array(X_flat)
    X_stft  = np.array(X_stft)
    X_mfcc  = np.array(X_mfcc)
    y       = np.array(y)
    groups  = np.array(groups)

    np.savez_compressed(CACHE_FILE,
                        X_flat=X_flat, X_stft=X_stft, X_mfcc=X_mfcc,
                        y=y, groups=groups)
    print(f"  Features cached to {CACHE_FILE.name}")

    return X_flat, X_stft, X_mfcc, y, groups


# ─────────────────────────────────────────
# HELPERS — build per-fold train/test idx
# ─────────────────────────────────────────
def fold_indices(groups):
    """
    GroupKFold over real participants + shared idle pool.
    N_SPLITS=5  → 5-fold group (4x faster, still person-independent)
    N_SPLITS=20 → full LOPO (1 participant per fold, gold standard)
    """
    idle_tr = np.where(groups == IDLE_TRAIN_GROUP)[0]
    idle_te = np.where(groups == IDLE_TEST_GROUP)[0]
    real_idx    = np.where((groups != IDLE_TRAIN_GROUP) & (groups != IDLE_TEST_GROUP))[0]
    real_groups = groups[real_idx]

    gkf = GroupKFold(n_splits=N_SPLITS)
    for fold, (tr, te) in enumerate(gkf.split(real_idx, groups=real_groups)):
        tr_idx = np.concatenate([real_idx[tr], idle_tr])
        te_idx = np.concatenate([real_idx[te], idle_te])
        yield fold, tr_idx, te_idx


# ─────────────────────────────────────────
# CONFUSION MATRIX
# ─────────────────────────────────────────
def save_cm(y_true, y_pred, tag):
    cm  = confusion_matrix(y_true, y_pred, labels=[0, 1])
    pct = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8) * 100
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(pct, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=["Idle", "Eating"],
                yticklabels=["Idle", "Eating"], ax=ax)
    ax.set_title(f"[Binary] LOPO Confusion Matrix (%)\n{tag}")
    ax.set_ylabel("True"); ax.set_xlabel("Predicted")
    plt.tight_layout()
    out = FIG_DIR / f"lopo_binary_cm_{tag}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"    Confusion matrix saved: {out.name}")


# ─────────────────────────────────────────
# LOPO — CLASSICAL ML
# ─────────────────────────────────────────
def lopo_classic(X, y, groups, model_name, feature_name):
    if model_name == "RF":
        clf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                     random_state=SEED, n_jobs=-1)
    elif model_name == "SVM":
        clf = SVC(kernel="rbf", class_weight="balanced", random_state=SEED)
    elif model_name == "kNN":
        clf = KNeighborsClassifier(n_neighbors=5)
    else:
        raise ValueError(model_name)

    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    fold_metrics = defaultdict(list)
    all_true, all_pred = [], []
    fold_rows = []

    for fold, tr_idx, te_idx in fold_indices(groups):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        if len(np.unique(y_tr)) < 2:
            continue

        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)

        acc  = accuracy_score(y_te, pred)
        prec = precision_score(y_te, pred, zero_division=0)
        rec  = recall_score(y_te, pred, zero_division=0)
        f1   = f1_score(y_te, pred, zero_division=0)

        fold_metrics["accuracy"].append(acc)
        fold_metrics["precision"].append(prec)
        fold_metrics["recall"].append(rec)
        fold_metrics["f1"].append(f1)

        all_true.extend(y_te); all_pred.extend(pred)
        fold_rows.append({"model": model_name, "feature": feature_name,
                          "fold": int(fold),
                          "accuracy": acc, "precision": prec,
                          "recall": rec, "f1": f1})

    save_cm(np.array(all_true), np.array(all_pred), f"{model_name}_{feature_name}")
    summary = {k: (np.mean(v), np.std(v)) for k, v in fold_metrics.items()}
    return summary, pd.DataFrame(fold_rows)


# ─────────────────────────────────────────
# LOPO — CNN
# ─────────────────────────────────────────
def build_cnn(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def lopo_cnn(X, y, groups, feature_name, epochs=15, batch_size=32):
    fold_metrics = defaultdict(list)
    all_true, all_pred = [], []
    fold_rows = []

    for fold, tr_idx, te_idx in fold_indices(groups):
        X_tr = X[tr_idx].astype(np.float32)
        X_te = X[te_idx].astype(np.float32)
        y_tr, y_te = y[tr_idx], y[te_idx]

        if len(np.unique(y_tr)) < 2:
            continue

        mn, sd = X_tr.mean(), X_tr.std() or 1.0
        X_tr = (X_tr - mn) / sd
        X_te = (X_te - mn) / sd

        n0, n1 = np.sum(y_tr == 0), np.sum(y_tr == 1)
        total  = n0 + n1
        cw = {0: total / (2 * n0 + 1e-8), 1: total / (2 * n1 + 1e-8)}

        model = build_cnn(X_tr[0].shape)
        es = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
        model.fit(X_tr, y_tr, epochs=epochs, batch_size=batch_size,
                  validation_split=0.1, callbacks=[es],
                  class_weight=cw, verbose=0)

        pred = (model.predict(X_te, verbose=0).ravel() > 0.5).astype(int)
        tf.keras.backend.clear_session()

        acc  = accuracy_score(y_te, pred)
        prec = precision_score(y_te, pred, zero_division=0)
        rec  = recall_score(y_te, pred, zero_division=0)
        f1   = f1_score(y_te, pred, zero_division=0)

        fold_metrics["accuracy"].append(acc)
        fold_metrics["precision"].append(prec)
        fold_metrics["recall"].append(rec)
        fold_metrics["f1"].append(f1)

        all_true.extend(y_te); all_pred.extend(pred)
        fold_rows.append({"model": "CNN", "feature": feature_name,
                          "fold": int(fold),
                          "accuracy": acc, "precision": prec,
                          "recall": rec, "f1": f1})

    save_cm(np.array(all_true), np.array(all_pred), f"CNN_{feature_name}")
    summary = {k: (np.mean(v), np.std(v)) for k, v in fold_metrics.items()}
    return summary, pd.DataFrame(fold_rows)


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def main():
    cv_label = "LOPO (20-fold)" if N_SPLITS == 20 else f"{N_SPLITS}-Fold Group K-Fold"
    print("=" * 72)
    print(f"  SensEat - Binary Classification (Participants 022-041)")
    print(f"  Evaluation: {cv_label}  |  Segmentation: 1.5s window, 0.5s hop")
    print("=" * 72)

    print("\nLoading and segmenting PCM files ...\n")
    X_flat, X_stft, X_mfcc, y, groups = build_dataset()

    n_eating = int(np.sum(y == 1))
    n_idle   = int(np.sum(y == 0))
    n_parts  = len(np.unique(groups[(groups != IDLE_TRAIN_GROUP) & (groups != IDLE_TEST_GROUP)]))
    print(f"\nDataset: {len(y)} total segments")
    print(f"  Eating: {n_eating}  |  Idle: {n_idle}")
    print(f"  Participants: {n_parts}  (LOPO folds = {n_parts})")

    if n_eating == 0 or n_idle == 0:
        print("Need both eating and idle segments. Aborting.")
        return

    results_rows = []
    foldwise_rows = []

    # ── Classical ML ──
    experiments = [
        ("RF",  "combined"),
        ("SVM", "combined"),
        ("kNN", "combined"),
    ]
    for model_name, feat_name in experiments:
        print(f"\n{model_name} + {feat_name}")
        res, fold_df = lopo_classic(X_flat, y, groups, model_name, feat_name)
        print(f"  Acc={res['accuracy'][0]:.4f}+-{res['accuracy'][1]:.4f}  "
              f"P={res['precision'][0]:.4f}  R={res['recall'][0]:.4f}  "
              f"F1={res['f1'][0]:.4f}+-{res['f1'][1]:.4f}")
        results_rows.append({
            "model": model_name, "feature": feat_name,
            "accuracy":  round(res["accuracy"][0],  4),
            "acc_std":   round(res["accuracy"][1],  4),
            "precision": round(res["precision"][0], 4),
            "recall":    round(res["recall"][0],    4),
            "f1":        round(res["f1"][0],        4),
            "f1_std":    round(res["f1"][1],        4),
        })
        foldwise_rows.append(fold_df)

    # ── CNN ──
    cnn_experiments = [
        ("STFT", X_stft),
        ("MFCC", X_mfcc),
    ]
    if not RUN_CNN:
        print("\n[CNN skipped — RUN_CNN=False]")
    for feat_name, X_img in (cnn_experiments if RUN_CNN else []):
        print(f"\nCNN + {feat_name}")
        res, fold_df = lopo_cnn(X_img, y, groups, feat_name)
        print(f"  Acc={res['accuracy'][0]:.4f}+-{res['accuracy'][1]:.4f}  "
              f"P={res['precision'][0]:.4f}  R={res['recall'][0]:.4f}  "
              f"F1={res['f1'][0]:.4f}+-{res['f1'][1]:.4f}")
        results_rows.append({
            "model": "CNN", "feature": feat_name,
            "accuracy":  round(res["accuracy"][0],  4),
            "acc_std":   round(res["accuracy"][1],  4),
            "precision": round(res["precision"][0], 4),
            "recall":    round(res["recall"][0],    4),
            "f1":        round(res["f1"][0],        4),
            "f1_std":    round(res["f1"][1],        4),
        })
        foldwise_rows.append(fold_df)

    # ── Save ──
    df = pd.DataFrame(results_rows).sort_values("f1", ascending=False)
    out_csv = SCRIPT_DIR / "pipeline_lopo_binary_results.csv"
    df.to_csv(out_csv, index=False)

    foldwise_df = pd.concat(foldwise_rows, ignore_index=True)
    fold_csv = SCRIPT_DIR / "pipeline_lopo_binary_foldwise.csv"
    foldwise_df.to_csv(fold_csv, index=False)

    print("\n" + "=" * 72)
    print("  RANKED RESULTS - LOPO Binary")
    print("=" * 72)
    print(df.to_string(index=False))
    print(f"\nResults saved: {out_csv}")
    print(f"Fold-wise saved: {fold_csv}")
    print("\nBinary LOPO pipeline complete.")


if __name__ == "__main__":
    main()
