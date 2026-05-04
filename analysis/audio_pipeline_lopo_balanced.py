"""
SensEat — LOPO Binary Pipeline WITH Per-Fold Undersampling
===========================================================
Identical to audio_pipeline_lopo.py EXCEPT:
  - Training set: eating class is randomly undersampled to match idle count
  - Test set: left untouched (real-world distribution)

This fixes the class-imbalance bias (model predicting Eating 100% of the time).
Reuses the same feature cache (feature_cache_binary.npz) — no re-extraction needed.

Output:
    pipeline_lopo_binary_balanced_results.csv
    pipeline_lopo_binary_balanced_foldwise.csv
    figures/binary_balanced/lopo_binary_cm_*.png
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
from spafe.features import gfcc as gfcc_feature

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
SAMPLES_PER_CHUNK = int(SAMPLE_RATE * CHIRP_PERIOD_S)

HOP_S       = 0.5
HOP_SAMPLES = int(SAMPLE_RATE * HOP_S)

DIRECT_PATH_SAMPLES = int((0.30 / 343.0) * SAMPLE_RATE)

CHIRP_START_HZ    = 18000.0
CHIRP_END_HZ      = 20000.0
CHIRP_DUR_SAMPLES = int(SAMPLE_RATE * 1.0)

SOUND_SPEED    = 343.0
TARGET_TAP_MIN = int(2 * 0.10 / SOUND_SPEED * SAMPLE_RATE)
TARGET_TAP_MAX = int(2 * 0.50 / SOUND_SPEED * SAMPLE_RATE)

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)
RNG = np.random.default_rng(SEED)

SCRIPT_DIR    = Path(__file__).resolve().parent
DATA_DIR_OLD  = SCRIPT_DIR / "old_data"   # participants 001–021
DATA_DIR_NEW  = SCRIPT_DIR / "data"       # participants 022–041
IDLE_DIR      = SCRIPT_DIR / "idle"
FIG_DIR       = SCRIPT_DIR / "figures" / "binary_balanced"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# (folder_name, data_dir) pairs for all 41 participants
EATING_SOURCES = (
    [(f"{i:03d}", DATA_DIR_OLD) for i in range(1, 22)] +
    [(f"{i:03d}", DATA_DIR_NEW) for i in range(22, 42)]
)

IDLE_TRAIN_GROUP = -1
IDLE_TEST_GROUP  = -2
IDLE_TRAIN_FRAC  = 0.70

N_SPLITS = 5
RUN_CNN  = True    # Set True to also run CNN (slower)

IRB_RE = re.compile(r"^(\d+)_(\d{3})_(\d{2})_(\d{2})(?: \(\d+\))?$")


# ─────────────────────────────────────────
# PREPROCESSING  (same as lopo.py)
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
    n     = CHIRP_DUR_SAMPLES
    dur_s = n / SAMPLE_RATE
    chirp = np.zeros(n, dtype=np.float32)
    for i in range(n):
        t        = i / SAMPLE_RATE
        freq     = CHIRP_START_HZ + (CHIRP_END_HZ - CHIRP_START_HZ) * (t / dur_s)
        chirp[i] = np.sin(2 * np.pi * freq * t)
    return butter_bandpass(chirp)


def tap_profile_features(segment, ref_chirp):
    rx  = segment[:CHIRP_DUR_SAMPLES].astype(np.float64)
    ref = ref_chirp.astype(np.float64)
    Nfft = 1 << (len(rx) + len(ref) - 1).bit_length()
    taps = np.abs(np.fft.ifft(np.fft.fft(rx, Nfft) * np.conj(np.fft.fft(ref, Nfft))))
    target = taps[TARGET_TAP_MIN:TARGET_TAP_MAX].astype(np.float32)
    peak   = target.max()
    if peak > 0:
        target = target / peak
    stats = np.array([
        float(np.max(target)), float(np.mean(target)), float(np.std(target)),
        float(np.argmax(target)), float(np.sum(target)),
        float(np.percentile(target, 75)), float(np.percentile(target, 25)),
    ], dtype=np.float32)
    return np.concatenate([target, stats])


def load_pcm(filepath):
    raw = np.fromfile(filepath, dtype=np.int16)
    if raw.size % 2 != 0:
        raw = raw[:-1]
    stereo = raw.reshape(-1, 2)
    return stereo[:, 0].astype(np.float32) / 32768.0


def compute_reference_direct_path(idle_files, max_files=5):
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
        seg = seg[DIRECT_PATH_SAMPLES:]
        seg = np.pad(seg, (0, DIRECT_PATH_SAMPLES))
        if ref_direct_path is not None:
            seg = cancel_direct_path(seg, ref_direct_path)
        segments.append(seg)
    return segments


# ─────────────────────────────────────────
# FEATURE EXTRACTION  (same as lopo.py)
# ─────────────────────────────────────────
def stat_features(x):
    zcr = librosa.feature.zero_crossing_rate(x)[0]
    env = np.abs(hilbert(x))
    return np.array([
        np.mean(x), np.std(x), np.max(x), np.min(x), np.mean(x**2),
        np.mean(zcr), np.mean(env), np.std(env), np.max(env),
    ], dtype=np.float32)


def spectral_features(x):
    S = np.abs(librosa.stft(x, n_fft=2048))
    freqs = librosa.fft_frequencies(sr=SAMPLE_RATE, n_fft=2048)
    centroid  = librosa.feature.spectral_centroid(S=S,  sr=SAMPLE_RATE)[0]
    bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=SAMPLE_RATE)[0]
    rolloff   = librosa.feature.spectral_rolloff(S=S,   sr=SAMPLE_RATE)[0]
    flux      = librosa.onset.onset_strength(S=librosa.amplitude_to_db(S), sr=SAMPLE_RATE)
    band_mask   = (freqs >= LOWCUT) & (freqs <= HIGHCUT)
    band_energy = np.mean(S[band_mask, :] ** 2)
    return np.array([
        np.mean(centroid), np.std(centroid), np.mean(bandwidth), np.std(bandwidth),
        np.mean(rolloff),  np.std(rolloff),  np.mean(flux),      np.std(flux),
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
        stat_features(x), spectral_features(x),
        wavelet_features(x), tap_profile_features(x, ref_chirp),
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


def mel_image(x, n_mels=64, fixed_frames=64):
    """
    Mel spectrogram: STFT magnitude mapped to mel-scale frequency axis.
    Mel scale is logarithmic (mimics human hearing): mel = 2595 * log10(1 + f/700).
    64 mel bins x 64 time frames, normalised 0-1, returned as (64,64,1) for CNN.
    """
    hop = max(64, int(len(x) / (fixed_frames - 1)))
    S   = librosa.feature.melspectrogram(
              y=x, sr=SAMPLE_RATE, n_mels=n_mels, hop_length=hop)
    S_db = librosa.power_to_db(S, ref=np.max)
    if S_db.shape[1] < fixed_frames:
        S_db = np.pad(S_db, ((0, 0), (0, fixed_frames - S_db.shape[1])),
                      constant_values=S_db.min())
    else:
        S_db = S_db[:, :fixed_frames]
    S_db = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)
    return S_db.astype(np.float32)[..., np.newaxis]


def gfcc_image(x, num_ceps=20, fixed_frames=64):
    """
    Gammatone Frequency Cepstral Coefficients (GFCC).
    Uses a gammatone filterbank (models the human cochlea more accurately than mel).
    Mel uses triangular filters; gammatone uses rounded asymmetric filters from auditory science.
    nfilts=64 gammatone filters, win_len=46ms, win_hop=23ms → ~64 frames over 1.5s.
    Returns (20, 64, 1) array for CNN input.
    """
    try:
        g = gfcc_feature.gfcc(x, fs=SAMPLE_RATE, num_ceps=num_ceps,
                               nfilts=64, win_len=0.046, win_hop=0.023)
        g = g.T  # spafe returns (frames, coeffs) → transpose to (coeffs, frames)
        if g.shape[1] < fixed_frames:
            g = np.pad(g, ((0, 0), (0, fixed_frames - g.shape[1])),
                       constant_values=g.min())
        else:
            g = g[:, :fixed_frames]
        g = (g - g.min()) / (g.max() - g.min() + 1e-8)
        return g.astype(np.float32)[..., np.newaxis]
    except Exception:
        return np.zeros((num_ceps, fixed_frames, 1), dtype=np.float32)


# ─────────────────────────────────────────
# DATASET — reuse existing cache
# ─────────────────────────────────────────
CACHE_FILE = SCRIPT_DIR / "feature_cache_binary_v4.npz"  # v4: old+new participants (41 total)

def build_dataset():
    if CACHE_FILE.exists():
        print(f"  Loading cached features from {CACHE_FILE.name} ...")
        d = np.load(CACHE_FILE, allow_pickle=False)
        return (d["X_flat"], d["X_stft"], d["X_mfcc"], d["X_mel"],
                d["X_gfcc"], d["X_wavelet"], d["y"], d["groups"])

    print("  No cache found — extracting features ...")
    ref_chirp = generate_reference_chirp()

    idle_files_all = sorted(IDLE_DIR.glob("*.pcm"))
    random.shuffle(idle_files_all)
    split = max(1, int(len(idle_files_all) * IDLE_TRAIN_FRAC))
    train_idle_files = idle_files_all[:split]
    test_idle_files  = idle_files_all[split:]

    ref_direct_path = compute_reference_direct_path(train_idle_files)

    X_flat, X_stft, X_mfcc, X_mel, X_gfcc, X_wavelet = [], [], [], [], [], []
    y, groups = [], []

    def _extract(seg):
        X_flat.append(combined_flat(seg, ref_chirp))
        X_stft.append(stft_image(seg))
        X_mfcc.append(mfcc_image(seg))
        X_mel.append(mel_image(seg))
        X_gfcc.append(gfcc_image(seg))
        X_wavelet.append(wavelet_features(seg))

    for folder, data_dir in EATING_SOURCES:
        path = data_dir / folder
        if not path.exists():
            continue
        pid = int(folder)
        eating_files    = [f for f in path.glob("*.pcm")
                           if IRB_RE.match(f.stem) and "_idleTail" not in f.stem]
        idle_tail_files = [f for f in path.glob("*_idleTail.pcm") if f.stat().st_size > 0]

        for f in eating_files:
            for seg in preprocess_and_segment(str(f), ref_direct_path):
                _extract(seg); y.append(1); groups.append(pid)
        for f in idle_tail_files:
            for seg in preprocess_and_segment(str(f), ref_direct_path):
                _extract(seg); y.append(0); groups.append(pid)

    for group_id, file_list in [(IDLE_TRAIN_GROUP, train_idle_files),
                                 (IDLE_TEST_GROUP,  test_idle_files)]:
        for f in file_list:
            for seg in preprocess_and_segment(str(f), ref_direct_path):
                _extract(seg); y.append(0); groups.append(group_id)

    X_flat    = np.array(X_flat);    X_stft   = np.array(X_stft)
    X_mfcc    = np.array(X_mfcc);    X_mel    = np.array(X_mel)
    X_gfcc    = np.array(X_gfcc);    X_wavelet = np.array(X_wavelet)
    y         = np.array(y);         groups    = np.array(groups)

    np.savez_compressed(CACHE_FILE,
                        X_flat=X_flat, X_stft=X_stft, X_mfcc=X_mfcc,
                        X_mel=X_mel,   X_gfcc=X_gfcc, X_wavelet=X_wavelet,
                        y=y, groups=groups)
    print(f"  Cached to {CACHE_FILE.name}")
    return X_flat, X_stft, X_mfcc, X_mel, X_gfcc, X_wavelet, y, groups


# ─────────────────────────────────────────
# FOLD INDICES
# ─────────────────────────────────────────
def fold_indices(groups):
    idle_tr  = np.where(groups == IDLE_TRAIN_GROUP)[0]
    idle_te  = np.where(groups == IDLE_TEST_GROUP)[0]
    real_idx = np.where((groups != IDLE_TRAIN_GROUP) & (groups != IDLE_TEST_GROUP))[0]
    gkf = GroupKFold(n_splits=N_SPLITS)
    for fold, (tr, te) in enumerate(gkf.split(real_idx, groups=groups[real_idx])):
        tr_idx = np.concatenate([real_idx[tr], idle_tr])
        te_idx = np.concatenate([real_idx[te], idle_te])
        yield fold, tr_idx, te_idx


# ─────────────────────────────────────────
# KEY CHANGE: per-fold undersampling
# ─────────────────────────────────────────
def undersample_train(X_tr, y_tr):
    """
    Undersample eating (majority) to match idle (minority) count.
    Only applied to training set — test set is NEVER touched.
    """
    eating_idx = np.where(y_tr == 1)[0]
    idle_idx   = np.where(y_tr == 0)[0]
    n_idle = len(idle_idx)

    if len(eating_idx) <= n_idle:
        return X_tr, y_tr   # already balanced or idle is majority

    eating_sub = RNG.choice(eating_idx, size=n_idle, replace=False)
    balanced   = np.concatenate([eating_sub, idle_idx])
    RNG.shuffle(balanced)
    return X_tr[balanced], y_tr[balanced]


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
    ax.set_title(f"[Binary] Balanced LOPO Confusion Matrix (%)\n{tag}")
    ax.set_ylabel("True"); ax.set_xlabel("Predicted")
    plt.tight_layout()
    out = FIG_DIR / f"lopo_binary_balanced_cm_{tag}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"    Confusion matrix saved: {out.name}")


# ─────────────────────────────────────────
# LOPO — CLASSICAL ML  (with undersampling)
# ─────────────────────────────────────────
def lopo_classic(X, y, groups, model_name, feature_name):
    if model_name == "RF":
        clf = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
    elif model_name == "SVM":
        clf = SVC(kernel="rbf", random_state=SEED)
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

        # ── undersample eating in training only ──
        X_tr, y_tr = undersample_train(X_tr, y_tr)
        print(f"    Fold {fold}: train eating={np.sum(y_tr==1)}  idle={np.sum(y_tr==0)}  "
              f"| test eating={np.sum(y_te==1)}  idle={np.sum(y_te==0)}")

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
                          "fold": int(fold), "accuracy": acc,
                          "precision": prec, "recall": rec, "f1": f1})

    save_cm(np.array(all_true), np.array(all_pred), f"{model_name}_{feature_name}")
    summary = {k: (np.mean(v), np.std(v)) for k, v in fold_metrics.items()}
    return summary, pd.DataFrame(fold_rows)


# ─────────────────────────────────────────
# LOPO — CNN  (with undersampling)
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

        # ── undersample eating in training only ──
        X_tr, y_tr = undersample_train(X_tr, y_tr)

        mn, sd = X_tr.mean(), X_tr.std() or 1.0
        X_tr = (X_tr - mn) / sd
        X_te = (X_te - mn) / sd

        model = build_cnn(X_tr[0].shape)
        es = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
        model.fit(X_tr, y_tr, epochs=epochs, batch_size=batch_size,
                  validation_split=0.1, callbacks=[es], verbose=0)

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
                          "fold": int(fold), "accuracy": acc,
                          "precision": prec, "recall": rec, "f1": f1})

    save_cm(np.array(all_true), np.array(all_pred), f"CNN_{feature_name}")
    summary = {k: (np.mean(v), np.std(v)) for k, v in fold_metrics.items()}
    return summary, pd.DataFrame(fold_rows)


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def main():
    cv_label = "LOPO (20-fold)" if N_SPLITS == 20 else f"{N_SPLITS}-Fold Group K-Fold"
    print("=" * 72)
    print(f"  SensEat - Binary Classification WITH Balanced Undersampling")
    print(f"  Evaluation: {cv_label}  |  Undersampling: eating -> idle count per fold")
    print("=" * 72)

    print("\nLoading features ...\n")
    X_flat, X_stft, X_mfcc, X_mel, X_gfcc, X_wavelet, y, groups = build_dataset()

    n_eating = int(np.sum(y == 1))
    n_idle   = int(np.sum(y == 0))
    n_parts  = len(np.unique(groups[(groups != IDLE_TRAIN_GROUP) & (groups != IDLE_TEST_GROUP)]))
    print(f"\nDataset: {len(y)} total segments")
    print(f"  Eating: {n_eating}  |  Idle: {n_idle}  (ratio {n_eating/max(n_idle,1):.1f}:1)")
    print(f"  Participants: {n_parts}")
    print(f"  NOTE: Training folds will be balanced 1:1 — test folds stay original ratio")

    if n_eating == 0 or n_idle == 0:
        print("Need both eating and idle segments. Aborting.")
        return

    results_rows  = []
    foldwise_rows = []

    # combined = stat + spectral + wavelet + tap_profile (all in one flat vector)
    # wavelet  = wavelet DWT standalone (db4, 4 levels, 20 features)
    classic_experiments = [
        ("RF",  "combined", X_flat),
        ("SVM", "combined", X_flat),
        ("kNN", "combined", X_flat),
        ("RF",  "wavelet",  X_wavelet),
        ("SVM", "wavelet",  X_wavelet),
    ]
    for model_name, feat_name, X_feat in classic_experiments:
        print(f"\n{model_name} + {feat_name} (balanced training)")
        res, fold_df = lopo_classic(X_feat, y, groups, model_name, feat_name)
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

    cnn_experiments = [("STFT", X_stft), ("MFCC", X_mfcc), ("Mel", X_mel), ("GFCC", X_gfcc)]
    if not RUN_CNN:
        print("\n[CNN skipped — RUN_CNN=False]")
    for feat_name, X_img in (cnn_experiments if RUN_CNN else []):
        print(f"\nCNN + {feat_name} (balanced training)")
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

    df = pd.DataFrame(results_rows).sort_values("f1", ascending=False)
    out_csv  = SCRIPT_DIR / "pipeline_lopo_binary_balanced_results.csv"
    fold_csv = SCRIPT_DIR / "pipeline_lopo_binary_balanced_foldwise.csv"
    df.to_csv(out_csv, index=False)
    pd.concat(foldwise_rows, ignore_index=True).to_csv(fold_csv, index=False)

    print("\n" + "=" * 72)
    print("  RANKED RESULTS — Balanced LOPO Binary")
    print("=" * 72)
    print(df.to_string(index=False))
    print(f"\nResults saved: {out_csv}")
    print(f"Fold-wise saved: {fold_csv}")
    print("\nBalanced binary LOPO pipeline complete.")


if __name__ == "__main__":
    main()
