"""
SensEat — SAR-Style Classification Pipeline
============================================
Instead of single-chirp STFT/MFCC, this pipeline:
  1. Extracts tap profile (echo vs distance) for each individual chirp
  2. Stacks N consecutive chirp profiles → SAR matrix (N_CHIRPS × TAP_BINS)
  3. Feeds SAR matrix to CNN (as image) and SVM/RF (as flat vector)

The SAR matrix captures HOW the echo changes across chirps = chewing pattern.
This is the core discriminative signal for food-type recognition.

Output:
    pipeline_sar_binary_results.csv
    pipeline_sar_multiclass_results.csv
    figures/sar_classifier/
"""

import os
import re
import random
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import cv2

from scipy.signal import butter, lfilter
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

# Chirp structure (Android app setting 2)
CHIRP_DUR_MS      = 1000
GAP_DUR_MS        = 500
CHIRP_DUR_SAMPLES = int(SAMPLE_RATE * CHIRP_DUR_MS / 1000)   # 44100
PERIOD_SAMPLES    = int(SAMPLE_RATE * (CHIRP_DUR_MS + GAP_DUR_MS) / 1000)  # 66150

CHIRP_START_HZ    = 18000.0
CHIRP_END_HZ      = 20000.0

DIRECT_PATH_SAMPLES = int((0.30 / 343.0) * SAMPLE_RATE)  # ~38

SOUND_SPEED    = 343.0
TARGET_TAP_MIN = int(2 * 0.10 / SOUND_SPEED * SAMPLE_RATE)  # ~26
TARGET_TAP_MAX = int(2 * 0.50 / SOUND_SPEED * SAMPLE_RATE)  # ~129
TAP_BINS       = TARGET_TAP_MAX - TARGET_TAP_MIN             # 103

# SAR window: how many consecutive chirps to stack
N_CHIRPS = 5     # 5 chirps × 1.5s = 7.5s window → 2-3 chewing cycles
SAR_HOP  = 2     # slide by 2 chirps → more windows per file

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)
RNG = np.random.default_rng(SEED)

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR   = SCRIPT_DIR / "data"
IDLE_DIR   = SCRIPT_DIR / "idle"
FIG_DIR    = SCRIPT_DIR / "figures" / "sar_classifier"
FIG_DIR.mkdir(parents=True, exist_ok=True)

EATING_FOLDERS   = [f"{i:03d}" for i in range(22, 42)]
VALID_FOOD_CODES = {1, 2, 4, 5, 8, 9, 10}
FOOD_NAMES = {
    1: "Tortilla", 2: "Mandarin", 4: "Cheeze_It",
    5: "Carrots",  8: "Noodles",  9: "Water", 10: "Coke",
}

IDLE_TRAIN_GROUP = -1
IDLE_TEST_GROUP  = -2
IDLE_TRAIN_FRAC  = 0.70

N_SPLITS = 5
RUN_CNN  = True

IRB_RE = re.compile(r"^(\d+)_(\d{3})_(\d{2})_(\d{2})(?: \(\d+\))?$")
CACHE_FILE = SCRIPT_DIR / "feature_cache_sar.npz"


# ─────────────────────────────────────────
# SIGNAL PROCESSING
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


def load_pcm(filepath):
    raw = np.fromfile(filepath, dtype=np.int16)
    if raw.size % 2 != 0:
        raw = raw[:-1]
    return raw.reshape(-1, 2)[:, 0].astype(np.float32) / 32768.0


def compute_reference_direct_path(idle_files, max_files=5):
    all_segs = []
    for f in list(idle_files)[:max_files]:
        sig = butter_bandpass(load_pcm(str(f)))
        n = len(sig) // PERIOD_SAMPLES
        for i in range(n):
            seg = sig[i * PERIOD_SAMPLES : i * PERIOD_SAMPLES + CHIRP_DUR_SAMPLES]
            if len(seg) == CHIRP_DUR_SAMPLES:
                all_segs.append(seg.astype(np.float64))
    if not all_segs:
        return None
    return np.mean(all_segs, axis=0).astype(np.float32)


def cancel_direct_path(chirp_seg, ref_direct_path):
    s = chirp_seg.astype(np.float64)
    d = ref_direct_path[:len(s)].astype(np.float64)
    c = np.dot(s, d) / (np.dot(d, d) + 1e-8)
    return (s - c * d).astype(np.float32)


def compute_tap_profile(chirp_seg, ref_chirp):
    """FFT cross-correlation → echo strength per distance bin."""
    rx  = chirp_seg.astype(np.float64)
    ref = ref_chirp.astype(np.float64)
    Nfft = 1 << (len(rx) + len(ref) - 1).bit_length()
    taps = np.abs(np.fft.ifft(
        np.fft.fft(rx, Nfft) * np.conj(np.fft.fft(ref, Nfft))
    ))
    target = taps[TARGET_TAP_MIN:TARGET_TAP_MAX].astype(np.float32)
    peak   = target.max()
    if peak > 0:
        target = target / peak
    return target   # shape: (TAP_BINS,) = (103,)


# ─────────────────────────────────────────
# PER-FILE: extract chirp tap profiles
# ─────────────────────────────────────────
def extract_tap_profiles(filepath, ref_chirp, ref_direct_path=None):
    """
    For each individual chirp in the file:
      - crop direct path
      - AIM Stage 1 IC
      - compute tap profile
    Returns list of (TAP_BINS,) arrays — one per chirp.
    """
    signal  = butter_bandpass(load_pcm(filepath))
    n_chirps = len(signal) // PERIOD_SAMPLES
    profiles = []

    for i in range(n_chirps):
        seg = signal[i * PERIOD_SAMPLES : i * PERIOD_SAMPLES + CHIRP_DUR_SAMPLES]
        if len(seg) < CHIRP_DUR_SAMPLES:
            continue
        # Direct path crop
        seg = seg[DIRECT_PATH_SAMPLES:]
        seg = np.pad(seg, (0, DIRECT_PATH_SAMPLES))
        # AIM Stage 1 IC
        if ref_direct_path is not None:
            seg = cancel_direct_path(seg, ref_direct_path[:CHIRP_DUR_SAMPLES])
        profiles.append(compute_tap_profile(seg[:CHIRP_DUR_SAMPLES], ref_chirp))

    return profiles   # list of (103,) arrays


# ─────────────────────────────────────────
# BUILD SAR WINDOWS FROM PROFILES
# ─────────────────────────────────────────
def profiles_to_sar_windows(profiles, n_chirps=N_CHIRPS, hop=SAR_HOP):
    """
    Slide a window of n_chirps over the profile list.
    Returns list of (n_chirps × TAP_BINS) SAR matrices.
    """
    windows = []
    for i in range(0, len(profiles) - n_chirps + 1, hop):
        mat = np.stack(profiles[i : i + n_chirps], axis=0)  # (N_CHIRPS, TAP_BINS)
        windows.append(mat)
    return windows


# ─────────────────────────────────────────
# FEATURE EXTRACTION FROM SAR MATRIX
# ─────────────────────────────────────────
def sar_to_image(mat, fixed_size=(64, 64)):
    """
    Resize SAR matrix to fixed image for CNN input.
    Rows = time (chirp index), Cols = distance (tap bin).
    """
    img = (mat - mat.min()) / (mat.max() - mat.min() + 1e-8)
    return cv2.resize(img, fixed_size, interpolation=cv2.INTER_CUBIC
                      ).astype(np.float32)[..., np.newaxis]  # (64,64,1)


def sar_to_flat(mat):
    """
    Flatten SAR matrix + temporal/spatial statistics for SVM/RF.
    mat shape: (N_CHIRPS, TAP_BINS)
    """
    flat = mat.flatten()                        # N_CHIRPS × TAP_BINS values

    # Per-chirp (row) stats — how strong the echo is at each time step
    row_mean = np.mean(mat, axis=1)             # (N_CHIRPS,)
    row_std  = np.std(mat,  axis=1)
    row_max  = np.max(mat,  axis=1)

    # Per-tap-bin (col) stats — average echo profile across all chirps
    col_mean = np.mean(mat, axis=0)             # (TAP_BINS,)
    col_std  = np.std(mat,  axis=0)

    # Temporal variance per tap bin — HOW MUCH each distance changes over time
    # This is the key feature: high variance = jaw is moving near that distance
    temporal_var = np.var(mat, axis=0)          # (TAP_BINS,)

    # Diff between consecutive chirps — detects sudden jaw movement events
    diffs     = np.diff(mat, axis=0)            # (N_CHIRPS-1, TAP_BINS)
    diff_mean = np.mean(np.abs(diffs), axis=0)  # (TAP_BINS,)
    diff_max  = np.max(np.abs(diffs),  axis=0)  # (TAP_BINS,)

    return np.concatenate([
        flat, row_mean, row_std, row_max,
        col_mean, col_std, temporal_var,
        diff_mean, diff_max,
    ]).astype(np.float32)


# ─────────────────────────────────────────
# DATASET BUILDER  (with cache)
# ─────────────────────────────────────────
def build_dataset():
    if CACHE_FILE.exists():
        print(f"  Loading cache: {CACHE_FILE.name}")
        d = np.load(CACHE_FILE, allow_pickle=False)
        return d["X_flat"], d["X_img"], d["y_bin"], d["y_food"], d["groups"]

    print("  No cache — extracting SAR features ...")
    ref_chirp = generate_reference_chirp()

    # Build reference direct path for AIM Stage 1 IC
    idle_files_all = sorted(IDLE_DIR.glob("*.pcm"))
    random.shuffle(idle_files_all)
    split            = max(1, int(len(idle_files_all) * IDLE_TRAIN_FRAC))
    train_idle_files = idle_files_all[:split]
    test_idle_files  = idle_files_all[split:]

    ref_direct_path = compute_reference_direct_path(train_idle_files)
    if ref_direct_path is not None:
        print(f"  Reference direct path from {min(5, len(train_idle_files))} idle files")

    X_flat, X_img           = [], []
    y_bin, y_food, groups   = [], [], []

    def process_file(filepath, label_bin, label_food, group_id):
        profiles = extract_tap_profiles(str(filepath), ref_chirp, ref_direct_path)
        windows  = profiles_to_sar_windows(profiles)
        for mat in windows:
            X_flat.append(sar_to_flat(mat))
            X_img.append(sar_to_image(mat))
            y_bin.append(label_bin)
            y_food.append(label_food)
            groups.append(group_id)

    # Eating files
    for folder in EATING_FOLDERS:
        path = DATA_DIR / folder
        if not path.exists():
            continue
        pid = int(folder)
        n_win = 0
        for f in path.glob("*.pcm"):
            if "_idleTail" in f.stem or "_meta" in f.stem:
                continue
            m = IRB_RE.match(f.stem)
            if not m:
                continue
            food = int(m.group(3))
            if food not in VALID_FOOD_CODES:
                continue
            before = len(X_flat)
            process_file(f, 1, food, pid)
            n_win += len(X_flat) - before
        print(f"  {folder}: {n_win} SAR windows (eating)")

    # Idle pool
    print(f"\n  Idle pool: {len(idle_files_all)} files → "
          f"{len(train_idle_files)} train, {len(test_idle_files)} test")

    for gid, flist in [(IDLE_TRAIN_GROUP, train_idle_files),
                       (IDLE_TEST_GROUP,  test_idle_files)]:
        for f in flist:
            process_file(f, 0, -1, gid)

    X_flat  = np.array(X_flat,  dtype=np.float32)
    X_img   = np.array(X_img,   dtype=np.float32)
    y_bin   = np.array(y_bin,   dtype=np.int32)
    y_food  = np.array(y_food,  dtype=np.int32)
    groups  = np.array(groups,  dtype=np.int32)

    np.savez_compressed(CACHE_FILE,
                        X_flat=X_flat, X_img=X_img,
                        y_bin=y_bin, y_food=y_food, groups=groups)
    print(f"  Cached to {CACHE_FILE.name}")
    return X_flat, X_img, y_bin, y_food, groups


# ─────────────────────────────────────────
# FOLD INDICES  (same LOPO logic)
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


def undersample_train(X_tr, y_tr):
    """Undersample majority class to match minority (per fold, training only)."""
    cls, counts = np.unique(y_tr, return_counts=True)
    min_count   = counts.min()
    keep = []
    for c in cls:
        idx = np.where(y_tr == c)[0]
        keep.append(RNG.choice(idx, size=min_count, replace=False))
    sel = np.concatenate(keep)
    RNG.shuffle(sel)
    return X_tr[sel], y_tr[sel]


# ─────────────────────────────────────────
# CONFUSION MATRIX
# ─────────────────────────────────────────
def save_cm(y_true, y_pred, labels, label_names, tag):
    cm  = confusion_matrix(y_true, y_pred, labels=labels)
    pct = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8) * 100
    sz  = max(5, len(labels))
    fig, ax = plt.subplots(figsize=(sz, sz - 1))
    sns.heatmap(pct, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=label_names, yticklabels=label_names, ax=ax)
    ax.set_title(f"[SAR] LOPO Confusion Matrix (%)\n{tag}")
    ax.set_ylabel("True"); ax.set_xlabel("Predicted")
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    out = FIG_DIR / f"sar_cm_{tag}.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"    CM saved: {out.name}")


# ─────────────────────────────────────────
# LOPO — CLASSICAL ML
# ─────────────────────────────────────────
def lopo_classic(X, y, groups, model_name, task="binary"):
    labels      = sorted(np.unique(y))
    label_names = (["Idle", "Eating"] if task == "binary"
                   else [FOOD_NAMES.get(l, str(l)) for l in labels])

    if model_name == "RF":
        clf = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
    elif model_name == "SVM":
        clf = SVC(kernel="rbf", random_state=SEED)
    else:
        clf = KNeighborsClassifier(n_neighbors=5)

    pipe = Pipeline([("sc", StandardScaler()), ("clf", clf)])
    metrics = defaultdict(list)
    all_true, all_pred = [], []
    rows = []

    for fold, tr_idx, te_idx in fold_indices(groups):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        if len(np.unique(y_tr)) < 2:
            continue

        X_tr, y_tr = undersample_train(X_tr, y_tr)

        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)

        avg = "binary" if task == "binary" else "macro"
        acc  = accuracy_score(y_te, pred)
        prec = precision_score(y_te, pred, average=avg, zero_division=0)
        rec  = recall_score(y_te, pred, average=avg, zero_division=0)
        f1   = f1_score(y_te, pred, average=avg, zero_division=0)

        metrics["accuracy"].append(acc)
        metrics["precision"].append(prec)
        metrics["recall"].append(rec)
        metrics["f1"].append(f1)
        all_true.extend(y_te); all_pred.extend(pred)
        rows.append({"model": model_name, "fold": fold,
                     "accuracy": acc, "precision": prec, "recall": rec, "f1": f1})

    save_cm(np.array(all_true), np.array(all_pred), labels, label_names,
            f"{task}_{model_name}_SAR_flat")
    summary = {k: (np.mean(v), np.std(v)) for k, v in metrics.items()}
    return summary, pd.DataFrame(rows)


# ─────────────────────────────────────────
# LOPO — CNN
# ─────────────────────────────────────────
def build_cnn(input_shape, n_classes):
    act  = "sigmoid"  if n_classes == 1 else "softmax"
    out  = 1          if n_classes == 1 else n_classes
    loss = "binary_crossentropy" if n_classes == 1 else "sparse_categorical_crossentropy"
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
        layers.Dense(out, activation=act),
    ])
    model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
    return model


def lopo_cnn(X, y, groups, task="binary"):
    n_classes   = 1 if task == "binary" else len(np.unique(y))
    labels      = sorted(np.unique(y))
    label_names = (["Idle", "Eating"] if task == "binary"
                   else [FOOD_NAMES.get(l, str(l)) for l in labels])

    metrics = defaultdict(list)
    all_true, all_pred = [], []
    rows = []

    for fold, tr_idx, te_idx in fold_indices(groups):
        X_tr = X[tr_idx].astype(np.float32)
        X_te = X[te_idx].astype(np.float32)
        y_tr, y_te = y[tr_idx], y[te_idx]
        if len(np.unique(y_tr)) < 2:
            continue

        # Undersample training
        X_tr, y_tr = undersample_train(X_tr, y_tr)

        # Normalize
        mn, sd = X_tr.mean(), X_tr.std() or 1.0
        X_tr = (X_tr - mn) / sd
        X_te = (X_te - mn) / sd

        # Remap labels to 0-indexed for multiclass
        if task == "multiclass":
            unique_cls = sorted(np.unique(y_tr))
            cls_map    = {c: i for i, c in enumerate(unique_cls)}
            y_tr_mapped = np.array([cls_map[c] for c in y_tr])
            y_te_mapped = np.array([cls_map.get(c, -1) for c in y_te])
            valid       = y_te_mapped >= 0
            X_te, y_te_mapped = X_te[valid], y_te_mapped[valid]
            y_te        = y_te[valid]
            n_cls       = len(unique_cls)
        else:
            y_tr_mapped = y_tr
            y_te_mapped = y_te
            n_cls       = 1

        model = build_cnn(X_tr[0].shape, n_cls)
        es = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
        model.fit(X_tr, y_tr_mapped, epochs=15, batch_size=32,
                  validation_split=0.1, callbacks=[es], verbose=0)

        if n_cls == 1:
            pred_mapped = (model.predict(X_te, verbose=0).ravel() > 0.5).astype(int)
            pred = pred_mapped
        else:
            pred_mapped = np.argmax(model.predict(X_te, verbose=0), axis=1)
            rev_map = {i: c for c, i in cls_map.items()}
            pred    = np.array([rev_map[i] for i in pred_mapped])
        tf.keras.backend.clear_session()

        avg = "binary" if task == "binary" else "macro"
        acc  = accuracy_score(y_te, pred)
        prec = precision_score(y_te, pred, average=avg, zero_division=0)
        rec  = recall_score(y_te, pred, average=avg, zero_division=0)
        f1   = f1_score(y_te, pred, average=avg, zero_division=0)

        metrics["accuracy"].append(acc)
        metrics["precision"].append(prec)
        metrics["recall"].append(rec)
        metrics["f1"].append(f1)
        all_true.extend(y_te); all_pred.extend(pred)
        rows.append({"model": "CNN", "fold": fold,
                     "accuracy": acc, "precision": prec, "recall": rec, "f1": f1})

    save_cm(np.array(all_true), np.array(all_pred), labels, label_names,
            f"{task}_CNN_SAR_image")
    summary = {k: (np.mean(v), np.std(v)) for k, v in metrics.items()}
    return summary, pd.DataFrame(rows)


# ─────────────────────────────────────────
# PRINT HELPER
# ─────────────────────────────────────────
def print_res(name, res):
    print(f"  {name:<25s}  "
          f"Acc={res['accuracy'][0]:.4f}±{res['accuracy'][1]:.4f}  "
          f"P={res['precision'][0]:.4f}  "
          f"R={res['recall'][0]:.4f}  "
          f"F1={res['f1'][0]:.4f}±{res['f1'][1]:.4f}")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def main():
    print("=" * 70)
    print("  SensEat — SAR-Style Classification Pipeline")
    print(f"  SAR window: {N_CHIRPS} chirps × 1.5s = {N_CHIRPS*1.5:.0f}s  |  "
          f"Hop: {SAR_HOP} chirps  |  TAP_BINS: {TAP_BINS}")
    print(f"  SAR matrix shape: ({N_CHIRPS} × {TAP_BINS})")
    print("=" * 70)

    print("\nBuilding SAR dataset ...")
    X_flat, X_img, y_bin, y_food, groups = build_dataset()

    n_eat  = int(np.sum(y_bin == 1))
    n_idle = int(np.sum(y_bin == 0))
    n_parts = len(np.unique(groups[(groups != IDLE_TRAIN_GROUP) & (groups != IDLE_TEST_GROUP)]))
    print(f"\n  Total SAR windows : {len(y_bin)}")
    print(f"  Eating            : {n_eat}")
    print(f"  Idle              : {n_idle}")
    print(f"  Participants      : {n_parts}")
    print(f"  SAR image shape   : {X_img[0].shape}")
    print(f"  Flat feature dim  : {X_flat.shape[1]}")

    all_results = []

    # ══════════════════════════════════════
    # BINARY: Eating vs Idle
    # ══════════════════════════════════════
    print("\n" + "=" * 70)
    print("  BINARY: Eating vs Idle")
    print("=" * 70)

    for model_name in ["RF", "SVM", "kNN"]:
        print(f"\n  {model_name} + SAR flat features (balanced)")
        res, _ = lopo_classic(X_flat, y_bin, groups, model_name, task="binary")
        print_res(f"{model_name}_SAR_flat", res)
        all_results.append({"task": "binary", "model": model_name,
                             "feature": "SAR_flat", **{k: round(v[0], 4) for k, v in res.items()}})

    if RUN_CNN:
        print(f"\n  CNN + SAR image (balanced)")
        res, _ = lopo_cnn(X_img, y_bin, groups, task="binary")
        print_res("CNN_SAR_image", res)
        all_results.append({"task": "binary", "model": "CNN",
                             "feature": "SAR_image", **{k: round(v[0], 4) for k, v in res.items()}})

    # ══════════════════════════════════════
    # MULTICLASS: Food type
    # ══════════════════════════════════════
    print("\n" + "=" * 70)
    print("  MULTICLASS: Food type (eating segments only)")
    print("=" * 70)

    eat_mask   = y_bin == 1
    Xf_eat     = X_flat[eat_mask]
    Xi_eat     = X_img[eat_mask]
    yf_eat     = y_food[eat_mask]
    groups_eat = groups[eat_mask]

    for model_name in ["RF", "SVM", "kNN"]:
        print(f"\n  {model_name} + SAR flat features (balanced)")
        res, _ = lopo_classic(Xf_eat, yf_eat, groups_eat, model_name, task="multiclass")
        print_res(f"{model_name}_SAR_flat", res)
        all_results.append({"task": "multiclass", "model": model_name,
                             "feature": "SAR_flat", **{k: round(v[0], 4) for k, v in res.items()}})

    if RUN_CNN:
        print(f"\n  CNN + SAR image (balanced)")
        res, _ = lopo_cnn(Xi_eat, yf_eat, groups_eat, task="multiclass")
        print_res("CNN_SAR_image", res)
        all_results.append({"task": "multiclass", "model": "CNN",
                             "feature": "SAR_image", **{k: round(v[0], 4) for k, v in res.items()}})

    # ── Save ──
    df = pd.DataFrame(all_results)
    out_csv = SCRIPT_DIR / "pipeline_sar_results.csv"
    df.to_csv(out_csv, index=False)

    print("\n" + "=" * 70)
    print("  RANKED RESULTS — SAR Pipeline")
    print("=" * 70)
    print(df.sort_values("f1", ascending=False).to_string(index=False))
    print(f"\nResults saved: {out_csv}")
    print("\nSAR classifier pipeline complete.")


if __name__ == "__main__":
    main()
