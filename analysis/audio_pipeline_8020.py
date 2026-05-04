"""
SensEat - 80/20 Participant-Level Split (Binary + Multiclass)
=============================================================
Split is at PARTICIPANT level (not segment level) to avoid data leakage.
  Train: 16 participants  |  Test: 4 participants

Outputs:
    pipeline_8020_results.csv
    figures/8020/cm_*.png

Usage:
    python audio_pipeline_8020.py
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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

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
HOP_SAMPLES       = int(SAMPLE_RATE * 0.5)           # 0.5s hop
DIRECT_PATH_SAMPLES = int((0.30 / 343.0) * SAMPLE_RATE)

# Reference chirp (Android app setting 2)
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

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR   = SCRIPT_DIR / "data"
IDLE_DIR   = SCRIPT_DIR / "idle"
FIG_DIR    = SCRIPT_DIR / "figures" / "8020"
FIG_DIR.mkdir(parents=True, exist_ok=True)

EATING_FOLDERS = [f"{i:03d}" for i in range(22, 42)]

VALID_FOOD_CODES = {1, 2, 4, 5, 8, 9, 10}
FOOD_NAMES = {
    1: "Tortilla", 2: "Mandarin", 4: "Cheeze_It",
    5: "Carrots",  8: "Noodles",  9: "Water", 10: "Coke",
}

IRB_RE     = re.compile(r"^(\d+)_(\d{3})_(\d{2})_(\d{2})(?: \(\d+\))?$")
OLD_FMT_RE = re.compile(r"^chirp_chips_")

CACHE_FILE = SCRIPT_DIR / "feature_cache_8020.npz"


# ─────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────
def butter_bandpass(data):
    nyq  = 0.5 * SAMPLE_RATE
    low, high = LOWCUT / nyq, HIGHCUT / nyq
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
    return raw.reshape(-1, 2)[:, 0].astype(np.float32) / 32768.0


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
    """AIM Stage 1: AGC-scaled direct path subtraction."""
    s = segment.astype(np.float64)
    d = ref_direct_path.astype(np.float64)
    c = np.dot(s, d) / (np.dot(d, d) + 1e-8)
    return (s - c * d).astype(np.float32)


def segment(filepath, ref_direct_path=None):
    sig = butter_bandpass(load_pcm(filepath))
    segs = []
    for i in range(0, len(sig) - SAMPLES_PER_CHUNK + 1, HOP_SAMPLES):
        s = sig[i : i + SAMPLES_PER_CHUNK]
        if len(s) == SAMPLES_PER_CHUNK:
            s = np.pad(s[DIRECT_PATH_SAMPLES:], (0, DIRECT_PATH_SAMPLES))
            if ref_direct_path is not None:
                s = cancel_direct_path(s, ref_direct_path)
            segs.append(s)
    return segs


# ─────────────────────────────────────────
# FEATURES
# ─────────────────────────────────────────
def flat_features(x, ref_chirp):
    zcr = librosa.feature.zero_crossing_rate(x)[0]
    env = np.abs(hilbert(x))

    S    = np.abs(librosa.stft(x, n_fft=2048))
    freq = librosa.fft_frequencies(sr=SAMPLE_RATE, n_fft=2048)
    cen  = librosa.feature.spectral_centroid(S=S,  sr=SAMPLE_RATE)[0]
    bw   = librosa.feature.spectral_bandwidth(S=S, sr=SAMPLE_RATE)[0]
    rol  = librosa.feature.spectral_rolloff(S=S,   sr=SAMPLE_RATE)[0]
    band = (freq >= LOWCUT) & (freq <= HIGHCUT)
    band_e = float(np.mean(S[band, :] ** 2))

    try:
        coeffs = pywt.wavedec(x, "db4", level=4)
    except Exception:
        coeffs = pywt.wavedec(x, "db4", level=1)
    wav = []
    for c in coeffs[1:]:
        wav.extend([np.mean(c), np.std(c), np.max(c), np.min(c), np.mean(c**2)])

    stat = [
        np.mean(x), np.std(x), np.max(x), np.min(x),
        np.mean(x**2), float(np.mean(zcr)),
        float(np.mean(env)), float(np.std(env)), float(np.max(env)),
        float(np.mean(cen)), float(np.std(cen)),
        float(np.mean(bw)),  float(np.std(bw)),
        float(np.mean(rol)), float(np.std(rol)),
        band_e,
    ]
    return np.concatenate([
        np.array(stat + wav, dtype=np.float32),
        tap_profile_features(x, ref_chirp),
    ])


def stft_img(x):
    try:
        D = librosa.amplitude_to_db(np.abs(librosa.stft(x, n_fft=2048)), ref=np.max)
        D = (D - D.min()) / (D.max() - D.min() + 1e-8)
        return cv2.resize(D, (64, 64))[..., np.newaxis].astype(np.float32)
    except Exception:
        return np.zeros((64, 64, 1), dtype=np.float32)


def mfcc_img(x, n_mfcc=40, frames=64):
    hop = max(64, int(len(x) / (frames - 1)))
    M = librosa.feature.mfcc(y=x, sr=SAMPLE_RATE, n_mfcc=n_mfcc, hop_length=hop)
    if M.shape[1] < frames:
        M = np.pad(M, ((0, 0), (0, frames - M.shape[1])), constant_values=M.min())
    else:
        M = M[:, :frames]
    M = (M - M.min()) / (M.max() - M.min() + 1e-8)
    return M.astype(np.float32)[..., np.newaxis]


# ─────────────────────────────────────────
# DATASET  (cached)
# ─────────────────────────────────────────
def parse_food_code(path):
    if OLD_FMT_RE.match(path.stem):
        return 10
    m = IRB_RE.match(path.stem)
    return int(m.group(3)) if m else -1


def build_dataset():
    if CACHE_FILE.exists():
        print(f"  Loading cache: {CACHE_FILE.name}")
        d = np.load(CACHE_FILE, allow_pickle=False)
        return (d["X_flat"], d["X_stft"], d["X_mfcc"],
                d["y_bin"], d["y_food"], d["groups"])

    print("  No cache — extracting features (saved for next run) ...")
    ref_chirp  = generate_reference_chirp()

    # Build reference direct path from idle pool (AIM Stage 1)
    idle_files = sorted(IDLE_DIR.glob("*.pcm"))
    ref_direct_path = compute_reference_direct_path(idle_files)
    if ref_direct_path is not None:
        print(f"  Reference direct path computed from {min(5, len(idle_files))} idle files")
    else:
        print("  Warning: could not compute reference direct path — skipping IC")

    X_flat, X_stft, X_mfcc = [], [], []
    y_bin, y_food, groups   = [], [], []

    # Eating files
    for folder in EATING_FOLDERS:
        path = DATA_DIR / folder
        if not path.exists():
            continue
        pid = int(folder)
        for f in path.glob("*.pcm"):
            if "_idleTail" in f.stem or "_meta" in f.stem:
                continue
            food = parse_food_code(f)
            if food not in VALID_FOOD_CODES:
                continue
            for s in segment(str(f), ref_direct_path):
                X_flat.append(flat_features(s, ref_chirp))
                X_stft.append(stft_img(s))
                X_mfcc.append(mfcc_img(s))
                y_bin.append(1)
                y_food.append(food)
                groups.append(pid)

        # idleTail per participant (if any, size > 0)
        for f in path.glob("*_idleTail.pcm"):
            if f.stat().st_size == 0:
                continue
            for s in segment(str(f), ref_direct_path):
                X_flat.append(flat_features(s, ref_chirp))
                X_stft.append(stft_img(s))
                X_mfcc.append(mfcc_img(s))
                y_bin.append(0)
                y_food.append(-1)
                groups.append(pid)

        print(f"  {folder}: done")

    # General idle pool
    print(f"  Idle pool: {len(idle_files)} files")
    for f in idle_files:
        for s in segment(str(f), ref_direct_path):
            X_flat.append(flat_features(s, ref_chirp))
            X_stft.append(stft_img(s))
            X_mfcc.append(mfcc_img(s))
            y_bin.append(0)
            y_food.append(-1)
            groups.append(-999)

    arr = (np.array(X_flat), np.array(X_stft), np.array(X_mfcc),
           np.array(y_bin), np.array(y_food), np.array(groups))

    np.savez_compressed(CACHE_FILE,
                        X_flat=arr[0], X_stft=arr[1], X_mfcc=arr[2],
                        y_bin=arr[3], y_food=arr[4], groups=arr[5])
    print(f"  Cached to {CACHE_FILE.name}")
    return arr


# ─────────────────────────────────────────
# CONFUSION MATRIX
# ─────────────────────────────────────────
def save_cm(y_true, y_pred, labels, label_names, tag):
    cm  = confusion_matrix(y_true, y_pred, labels=labels)
    pct = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8) * 100
    fig, ax = plt.subplots(figsize=(max(5, len(labels)), max(4, len(labels) - 1)))
    sns.heatmap(pct, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=label_names, yticklabels=label_names, ax=ax)
    ax.set_title(f"80/20 Confusion Matrix (%) — {tag}")
    ax.set_ylabel("True"); ax.set_xlabel("Predicted")
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    out = FIG_DIR / f"cm_{tag}.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"  Confusion matrix: {out.name}")


# ─────────────────────────────────────────
# CNN
# ─────────────────────────────────────────
def build_cnn(input_shape, n_classes):
    out_act = "sigmoid" if n_classes == 1 else "softmax"
    out_n   = 1 if n_classes == 1 else n_classes
    loss    = "binary_crossentropy" if n_classes == 1 else "sparse_categorical_crossentropy"

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
        layers.Dense(out_n, activation=out_act),
    ])
    model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
    return model


def train_cnn(X_tr, y_tr, X_te, y_te, n_classes, tag, epochs=15):
    mn, sd = X_tr.mean(), X_tr.std() or 1.0
    X_tr = (X_tr - mn) / sd
    X_te = (X_te - mn) / sd

    # class weights
    unique, counts = np.unique(y_tr, return_counts=True)
    total = len(y_tr)
    cw = {int(u): total / (len(unique) * c) for u, c in zip(unique, counts)}

    le = None
    if n_classes > 1:
        le = LabelEncoder().fit(y_tr)
        y_tr = le.transform(y_tr)
        y_te_mapped = le.transform(y_te)
    else:
        y_te_mapped = y_te

    model = build_cnn(X_tr[0].shape, n_classes)
    es = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    model.fit(X_tr, y_tr, epochs=epochs, batch_size=32,
              validation_split=0.1, callbacks=[es],
              class_weight=cw, verbose=0)

    if n_classes == 1:
        pred = (model.predict(X_te, verbose=0).ravel() > 0.5).astype(int)
    else:
        pred_idx = np.argmax(model.predict(X_te, verbose=0), axis=1)
        pred = le.inverse_transform(pred_idx)

    tf.keras.backend.clear_session()
    return pred


# ─────────────────────────────────────────
# 80/20 SPLIT HELPERS
# ─────────────────────────────────────────
def participant_split(groups, test_size=0.2):
    """
    Split at participant level — 80% train, 20% test.
    Idle pool (groups=-999) is also split 80/20 so test set contains idle samples.
    """
    real_pids = np.unique(groups[groups != -999])
    train_p, test_p = train_test_split(real_pids, test_size=test_size, random_state=SEED)

    # Split idle pool indices 80/20 (only if idle exists in this subset)
    idle_idx = np.where(groups == -999)[0]
    if len(idle_idx) > 0:
        n_idle_test = max(1, int(len(idle_idx) * test_size))
        rng = np.random.default_rng(SEED)
        idle_test_idx = rng.choice(idle_idx, size=n_idle_test, replace=False)
        idle_test_mask = np.zeros(len(groups), dtype=bool)
        idle_test_mask[idle_test_idx] = True
        train_mask = np.isin(groups, train_p) | ((groups == -999) & ~idle_test_mask)
        test_mask  = np.isin(groups, test_p)  | idle_test_mask
    else:
        train_mask = np.isin(groups, train_p)
        test_mask  = np.isin(groups, test_p)
    return train_mask, test_mask


# ─────────────────────────────────────────
# BINARY EVALUATION
# ─────────────────────────────────────────
def run_binary(X_flat, X_stft, X_mfcc, y_bin, groups):
    print("\n" + "=" * 60)
    print("  BINARY: Eating (1) vs Idle (0)  — 80/20 split")
    print("=" * 60)

    tr_mask, te_mask = participant_split(groups)
    rows = []

    # Classical ML
    for model_name in ["SVM", "RF"]:
        X_tr, X_te = X_flat[tr_mask], X_flat[te_mask]
        y_tr, y_te = y_bin[tr_mask],  y_bin[te_mask]

        if model_name == "SVM":
            clf = SVC(kernel="rbf", class_weight="balanced", random_state=SEED)
        else:
            clf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                         random_state=SEED, n_jobs=-1)

        pipe = Pipeline([("sc", StandardScaler()), ("clf", clf)])
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)

        tag = f"binary_{model_name}_flat"
        _print_and_store(y_te, pred, model_name, "flat_features", rows, tag,
                         [0, 1], ["Idle", "Eating"])

    # CNN
    for feat_name, X_img in [("STFT", X_stft), ("MFCC", X_mfcc)]:
        X_tr = X_img[tr_mask].astype(np.float32)
        X_te = X_img[te_mask].astype(np.float32)
        y_tr, y_te = y_bin[tr_mask], y_bin[te_mask]

        pred = train_cnn(X_tr, y_tr, X_te, y_te, n_classes=1, tag=f"binary_CNN_{feat_name}")
        tag  = f"binary_CNN_{feat_name}"
        _print_and_store(y_te, pred, "CNN", feat_name, rows, tag,
                         [0, 1], ["Idle", "Eating"])

    return rows


# ─────────────────────────────────────────
# MULTICLASS EVALUATION
# ─────────────────────────────────────────
def run_multiclass(X_flat, X_stft, X_mfcc, y_food, groups):
    print("\n" + "=" * 60)
    print("  MULTICLASS: Food type — 80/20 split")
    print("=" * 60)

    # Keep only eating segments (y_food != -1)
    eat_mask = y_food != -1
    Xf, Xs, Xm = X_flat[eat_mask], X_stft[eat_mask], X_mfcc[eat_mask]
    yf, gp     = y_food[eat_mask], groups[eat_mask]

    tr_mask, te_mask = participant_split(gp)

    # Filter train to only classes present in test
    test_classes = np.unique(yf[te_mask])
    tr_class_mask = np.isin(yf[tr_mask], test_classes)

    rows = []
    food_labels = [FOOD_NAMES[c] for c in sorted(test_classes)]

    # Classical ML
    for model_name in ["SVM", "RF"]:
        X_tr = Xf[tr_mask][tr_class_mask]
        y_tr = yf[tr_mask][tr_class_mask]
        X_te = Xf[te_mask]
        y_te = yf[te_mask]

        if model_name == "SVM":
            clf = SVC(kernel="rbf", class_weight="balanced", random_state=SEED)
        else:
            clf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                         random_state=SEED, n_jobs=-1)

        pipe = Pipeline([("sc", StandardScaler()), ("clf", clf)])
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)

        tag = f"multi_{model_name}_flat"
        _print_and_store(y_te, pred, model_name, "flat_features", rows, tag,
                         sorted(test_classes), food_labels)

    # CNN
    for feat_name, X_img in [("STFT", Xs), ("MFCC", Xm)]:
        X_tr = X_img[tr_mask][tr_class_mask].astype(np.float32)
        y_tr = yf[tr_mask][tr_class_mask]
        X_te = X_img[te_mask].astype(np.float32)
        y_te = yf[te_mask]

        pred = train_cnn(X_tr, y_tr, X_te, y_te,
                         n_classes=len(test_classes), tag=f"multi_CNN_{feat_name}")
        tag  = f"multi_CNN_{feat_name}"
        _print_and_store(y_te, pred, "CNN", feat_name, rows, tag,
                         sorted(test_classes), food_labels)

    return rows


# ─────────────────────────────────────────
# SHARED PRINT + STORE
# ─────────────────────────────────────────
def _print_and_store(y_te, pred, model, feature, rows, tag, labels, label_names):
    acc   = accuracy_score(y_te, pred)
    prec  = precision_score(y_te, pred, average="macro", zero_division=0)
    rec   = recall_score(y_te, pred, average="macro", zero_division=0)
    f1    = f1_score(y_te, pred, average="macro", zero_division=0)
    wf1   = f1_score(y_te, pred, average="weighted", zero_division=0)

    task = "binary" if len(labels) == 2 else "multiclass"
    print(f"  {model:4s} + {feature:<15s}  "
          f"Acc={acc:.4f}  MacroF1={f1:.4f}  WeightedF1={wf1:.4f}")

    save_cm(y_te, pred, labels, label_names, tag)
    rows.append({
        "task": task, "model": model, "feature": feature,
        "accuracy": round(acc, 4), "precision": round(prec, 4),
        "recall": round(rec, 4), "macro_f1": round(f1, 4),
        "weighted_f1": round(wf1, 4),
    })


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def main():
    print("=" * 60)
    print("  SensEat - 80/20 Participant-Level Split")
    print("  Train: 16 participants  |  Test: 4 participants")
    print("=" * 60)
    print("\nLoading dataset ...")

    X_flat, X_stft, X_mfcc, y_bin, y_food, groups = build_dataset()

    print(f"  Total segments : {len(y_bin)}")
    print(f"  Eating         : {int(np.sum(y_bin == 1))}")
    print(f"  Idle           : {int(np.sum(y_bin == 0))}")

    all_rows = []
    all_rows += run_binary(X_flat, X_stft, X_mfcc, y_bin, groups)
    all_rows += run_multiclass(X_flat, X_stft, X_mfcc, y_food, groups)

    df = pd.DataFrame(all_rows)
    out = SCRIPT_DIR / "pipeline_8020_results.csv"
    df.to_csv(out, index=False)

    print("\n" + "=" * 60)
    print("  FINAL RESULTS — 80/20 Split")
    print("=" * 60)
    print(df.to_string(index=False))
    print(f"\nResults saved: {out}")


if __name__ == "__main__":
    main()
