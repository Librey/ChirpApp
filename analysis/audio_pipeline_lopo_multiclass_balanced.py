"""
SensEat — LOPO Multiclass Pipeline WITH Per-Fold Undersampling
==============================================================
Based on audio_pipeline_lopo_multiclass.py with:
  - Per-fold undersampling: all food classes balanced to minority count
    (applied to training set only — test set untouched)
  - Added features: Mel spectrogram, GFCC (real spafe implementation)
  - Balancing applied inside each fold loop, never globally

Food classes: Tortilla(1), Mandarin(2), Cheeze_It(4), Carrots(5),
              Noodles(8), Water(9), Coke(10)

Output:
    pipeline_lopo_multiclass_balanced_results.csv
    pipeline_lopo_multiclass_balanced_foldwise.csv
    figures/multi_balanced/lopo_multi_balanced_cm_*.png
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
import pywt
from spafe.features import gfcc as gfcc_feature

from scipy.signal import butter, lfilter
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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
SAMPLE_RATE       = 44100
LOWCUT            = 17500.0
HIGHCUT           = 20500.0
FILTER_ORDER      = 6
CHIRP_PERIOD_S    = 1.5
SAMPLES_PER_CHUNK = int(SAMPLE_RATE * CHIRP_PERIOD_S)
HOP_S             = 0.5
HOP_SAMPLES       = int(SAMPLE_RATE * HOP_S)
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

SCRIPT_DIR   = Path(__file__).resolve().parent
DATA_DIR_OLD = SCRIPT_DIR / "old_data"   # participants 001–021
DATA_DIR_NEW = SCRIPT_DIR / "data"       # participants 022–041
FIG_DIR      = SCRIPT_DIR / "figures" / "multi_balanced"
FIG_DIR.mkdir(parents=True, exist_ok=True)

EATING_SOURCES = (
    [(f"{i:03d}", DATA_DIR_OLD) for i in range(1, 22)] +
    [(f"{i:03d}", DATA_DIR_NEW) for i in range(22, 42)]
)

VALID_FOOD_CODES = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

FOOD_NAMES = {
    1:  "Tortilla",
    2:  "Mandarin",
    3:  "Chicken",
    4:  "Cheeze_It",
    5:  "Carrots",
    6:  "Chocolate",
    7:  "Yogurt",
    8:  "Noodles",
    9:  "Water",
    10: "Coke",
}

IRB_RE     = re.compile(r"^(\d+)_(\d{3})_(\d{2})_(\d{2})(?:\s*\(\d+\))?$")
OLD_FMT_RE = re.compile(r"^chirp_chips_")

CACHE_FILE = SCRIPT_DIR / "feature_cache_multiclass_balanced_v2.npz"  # v2: old+new (41 total)


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
    stereo = raw.reshape(-1, 2)
    return stereo[:, 0].astype(np.float32) / 32768.0


def preprocess_and_segment(filepath):
    signal = load_pcm(filepath)
    signal = butter_bandpass(signal)
    chunks = []
    for i in range(0, len(signal) - SAMPLES_PER_CHUNK + 1, HOP_SAMPLES):
        seg = signal[i : i + SAMPLES_PER_CHUNK]
        if len(seg) != SAMPLES_PER_CHUNK:
            continue
        seg = seg[DIRECT_PATH_SAMPLES:]
        seg = np.pad(seg, (0, DIRECT_PATH_SAMPLES))
        chunks.append(seg)
    return chunks


def parse_food_code(pcm_path: Path) -> int:
    stem = pcm_path.stem
    if OLD_FMT_RE.match(stem):
        return 10
    m = IRB_RE.match(stem)
    if m:
        return int(m.group(3))
    return -1


# ─────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────
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


def stat_features(x):
    zcr = librosa.feature.zero_crossing_rate(x)[0]
    return np.array([
        np.mean(x), np.std(x), np.max(x), np.min(x),
        np.mean(x ** 2), np.mean(zcr),
    ], dtype=np.float32)


def wavelet_features(x):
    try:
        coeffs = pywt.wavedec(x, "db4", level=4)
    except Exception:
        coeffs = pywt.wavedec(x, "db4", level=1)
    feats = []
    for c in coeffs[1:]:
        feats.extend([np.mean(c), np.std(c), np.max(c), np.min(c), np.mean(c ** 2)])
    return np.array(feats, dtype=np.float32)


def combined_flat(x, ref_chirp):
    return np.concatenate([
        stat_features(x), wavelet_features(x), tap_profile_features(x, ref_chirp)
    ])


def stft_image(x, fixed_size=(64, 64)):
    try:
        D    = np.abs(librosa.stft(x, n_fft=2048))
        S_db = librosa.amplitude_to_db(D, ref=np.max)
        S_db = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)
        return cv2.resize(S_db, fixed_size)[..., np.newaxis].astype(np.float32)
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
    hop  = max(64, int(len(x) / (fixed_frames - 1)))
    S    = librosa.feature.melspectrogram(y=x, sr=SAMPLE_RATE, n_mels=n_mels, hop_length=hop)
    S_db = librosa.power_to_db(S, ref=np.max)
    if S_db.shape[1] < fixed_frames:
        S_db = np.pad(S_db, ((0, 0), (0, fixed_frames - S_db.shape[1])), constant_values=S_db.min())
    else:
        S_db = S_db[:, :fixed_frames]
    S_db = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)
    return S_db.astype(np.float32)[..., np.newaxis]


def gfcc_image(x, num_ceps=20, fixed_frames=64):
    try:
        g = gfcc_feature.gfcc(x, fs=SAMPLE_RATE, num_ceps=num_ceps,
                               nfilts=64, win_len=0.046, win_hop=0.023)
        g = g.T
        if g.shape[1] < fixed_frames:
            g = np.pad(g, ((0, 0), (0, fixed_frames - g.shape[1])), constant_values=g.min())
        else:
            g = g[:, :fixed_frames]
        g = (g - g.min()) / (g.max() - g.min() + 1e-8)
        return g.astype(np.float32)[..., np.newaxis]
    except Exception:
        return np.zeros((num_ceps, fixed_frames, 1), dtype=np.float32)


# ─────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────
def build_dataset():
    if CACHE_FILE.exists():
        print(f"  Loading cached features from {CACHE_FILE.name} ...")
        d = np.load(CACHE_FILE, allow_pickle=False)
        return (d["X_flat"], d["X_stft"], d["X_mfcc"],
                d["X_mel"], d["X_gfcc"], d["y"], d["groups"])

    print("  No cache — extracting features ...")
    ref_chirp = generate_reference_chirp()

    X_flat, X_stft, X_mfcc, X_mel, X_gfcc = [], [], [], [], []
    y, groups = [], []

    for folder_name, data_dir in EATING_SOURCES:
        folder_path = data_dir / folder_name
        if not folder_path.exists():
            print(f"  Missing: {folder_path}")
            continue

        pid = int(folder_name)
        pcm_files = sorted([
            f for f in folder_path.glob("*.pcm")
            if "_idleTail" not in f.stem and "_meta" not in f.stem
        ])

        loaded = 0
        for pcm in pcm_files:
            food_code = parse_food_code(pcm)
            if food_code not in VALID_FOOD_CODES:
                continue
            for seg in preprocess_and_segment(str(pcm)):
                X_flat.append(combined_flat(seg, ref_chirp))
                X_stft.append(stft_image(seg))
                X_mfcc.append(mfcc_image(seg))
                X_mel.append(mel_image(seg))
                X_gfcc.append(gfcc_image(seg))
                y.append(food_code)
                groups.append(pid)
                loaded += 1

        print(f"  {folder_name}: {loaded} segments")

    X_flat = np.array(X_flat); X_stft = np.array(X_stft)
    X_mfcc = np.array(X_mfcc); X_mel  = np.array(X_mel)
    X_gfcc = np.array(X_gfcc); y      = np.array(y)
    groups = np.array(groups)

    np.savez_compressed(CACHE_FILE,
                        X_flat=X_flat, X_stft=X_stft, X_mfcc=X_mfcc,
                        X_mel=X_mel,   X_gfcc=X_gfcc, y=y, groups=groups)
    print(f"  Cached to {CACHE_FILE.name}")
    return X_flat, X_stft, X_mfcc, X_mel, X_gfcc, y, groups


# ─────────────────────────────────────────
# PER-FOLD UNDERSAMPLING (multiclass)
# Undersample all classes to the minority class count in training set.
# Test set is NEVER touched.
# ─────────────────────────────────────────
def undersample_train_multiclass(X_tr, y_tr):
    classes, counts = np.unique(y_tr, return_counts=True)
    min_count = counts.min()

    selected = []
    for cls in classes:
        idx = np.where(y_tr == cls)[0]
        chosen = RNG.choice(idx, size=min_count, replace=False)
        selected.append(chosen)

    balanced = np.concatenate(selected)
    RNG.shuffle(balanced)
    return X_tr[balanced], y_tr[balanced]


# ─────────────────────────────────────────
# CONFUSION MATRIX
# ─────────────────────────────────────────
def save_cm(y_true, y_pred, tag):
    unique_labels  = sorted(set(y_true))
    display_labels = [FOOD_NAMES[k] for k in unique_labels]

    cm     = confusion_matrix(y_true, y_pred, labels=unique_labels)
    cm_pct = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8) * 100

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=display_labels, yticklabels=display_labels, ax=ax)
    ax.set_title(f"[Multiclass] Balanced LOPO Confusion Matrix (%)\n{tag}")
    ax.set_ylabel("True Food")
    ax.set_xlabel("Predicted Food")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    out = FIG_DIR / f"lopo_multi_balanced_cm_{tag}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"    Confusion matrix saved: {out.name}")


# ─────────────────────────────────────────
# LOPO — CLASSICAL ML (with per-fold undersampling)
# ─────────────────────────────────────────
def lopo_classic(X, y, groups, model_name, feature_name):
    logo = LeaveOneGroupOut()
    fold_metrics = defaultdict(list)
    all_true, all_pred = [], []
    fold_rows = []

    for train_idx, test_idx in logo.split(X, y, groups):
        pid    = int(groups[test_idx[0]])
        X_te   = X[test_idx]
        y_te   = y[test_idx]

        # Restrict to food classes present in held-out participant
        test_classes = np.unique(y_te)
        train_mask   = np.isin(y[train_idx], test_classes)
        X_tr = X[train_idx][train_mask]
        y_tr = y[train_idx][train_mask]

        if len(np.unique(y_tr)) < 2:
            continue

        # ── per-fold undersampling (training only) ──
        X_tr, y_tr = undersample_train_multiclass(X_tr, y_tr)

        if model_name == "RF":
            clf = RandomForestClassifier(n_estimators=150, random_state=SEED, n_jobs=-1)
        elif model_name == "SVM":
            clf = SVC(kernel="rbf", random_state=SEED)
        else:
            raise ValueError(model_name)

        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te)

        acc        = accuracy_score(y_te, y_pred)
        macro_f1   = f1_score(y_te, y_pred, average="macro",    zero_division=0)
        weighted_f1= f1_score(y_te, y_pred, average="weighted", zero_division=0)

        fold_metrics["accuracy"].append(acc)
        fold_metrics["macro_f1"].append(macro_f1)
        fold_metrics["weighted_f1"].append(weighted_f1)

        all_true.extend(y_te)
        all_pred.extend(y_pred)
        fold_rows.append({"model": model_name, "feature": feature_name,
                          "participant": pid, "accuracy": acc,
                          "macro_f1": macro_f1, "weighted_f1": weighted_f1})

    save_cm(np.array(all_true), np.array(all_pred), f"{model_name}_{feature_name}")
    summary = {k: (np.mean(v), np.std(v)) for k, v in fold_metrics.items()}
    return summary, pd.DataFrame(fold_rows)


# ─────────────────────────────────────────
# LOPO — CNN (with per-fold undersampling)
# ─────────────────────────────────────────
def build_cnn(input_shape, num_classes):
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
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def lopo_cnn(X, y, groups, feature_name, epochs=10, batch_size=32):
    logo = LeaveOneGroupOut()
    fold_metrics = defaultdict(list)
    all_true, all_pred = [], []
    fold_rows = []

    for train_idx, test_idx in logo.split(X, y, groups):
        pid   = int(groups[test_idx[0]])
        X_te  = X[test_idx].astype(np.float32)
        y_te  = y[test_idx]

        test_classes = np.unique(y_te)
        train_mask   = np.isin(y[train_idx], test_classes)
        X_tr = X[train_idx][train_mask].astype(np.float32)
        y_tr = y[train_idx][train_mask]

        if len(np.unique(y_tr)) < 2:
            continue

        # ── per-fold undersampling (training only) ──
        X_tr, y_tr = undersample_train_multiclass(X_tr, y_tr)

        # Map food codes to 0-based indices for CNN
        code_to_idx = {c: i for i, c in enumerate(sorted(test_classes))}
        idx_to_code = {i: c for c, i in code_to_idx.items()}
        y_tr_idx = np.array([code_to_idx[c] for c in y_tr])
        num_classes = len(test_classes)

        mn, sd = X_tr.mean(), X_tr.std() or 1.0
        X_tr = (X_tr - mn) / sd
        X_te = (X_te - mn) / sd

        model = build_cnn(X_tr[0].shape, num_classes)
        es = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
        model.fit(X_tr, y_tr_idx, epochs=epochs, batch_size=batch_size,
                  validation_split=0.1, callbacks=[es], verbose=0)

        y_pred_idx  = np.argmax(model.predict(X_te, verbose=0), axis=1)
        y_pred_orig = np.array([idx_to_code[i] for i in y_pred_idx])
        tf.keras.backend.clear_session()

        acc         = accuracy_score(y_te, y_pred_orig)
        macro_f1    = f1_score(y_te, y_pred_orig, average="macro",    zero_division=0)
        weighted_f1 = f1_score(y_te, y_pred_orig, average="weighted", zero_division=0)

        fold_metrics["accuracy"].append(acc)
        fold_metrics["macro_f1"].append(macro_f1)
        fold_metrics["weighted_f1"].append(weighted_f1)

        all_true.extend(y_te)
        all_pred.extend(y_pred_orig)
        fold_rows.append({"model": "CNN", "feature": feature_name,
                          "participant": pid, "accuracy": acc,
                          "macro_f1": macro_f1, "weighted_f1": weighted_f1})

    save_cm(np.array(all_true), np.array(all_pred), f"CNN_{feature_name}")
    summary = {k: (np.mean(v), np.std(v)) for k, v in fold_metrics.items()}
    return summary, pd.DataFrame(fold_rows)


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def main():
    print("=" * 72)
    print("  SensEat — Multiclass LOPO WITH Per-Fold Undersampling")
    print(f"  Foods: {list(FOOD_NAMES.values())}")
    print("  Balancing: all food classes -> minority class count per fold")
    print("  Test folds: untouched (real distribution)")
    print("=" * 72)

    print("\nLoading features ...\n")
    X_flat, X_stft, X_mfcc, X_mel, X_gfcc, y, groups = build_dataset()

    if len(y) == 0:
        print("No segments found. Check DATA_DIR.")
        return

    print(f"\nDataset: {len(y)} total segments")
    print(f"  Participants: {len(np.unique(groups))}")
    for code in sorted(VALID_FOOD_CODES):
        count = int(np.sum(y == code))
        print(f"  {FOOD_NAMES[code]:15s} (code {code:02d}): {count} segments")

    results_rows  = []
    foldwise_rows = []

    # Stat-only slice: first 6 features of combined_flat are stat_features
    X_stat = X_flat[:, :6]
    classic_experiments = [
        ("RF",  "combined", X_flat),
        ("SVM", "combined", X_flat),
        ("RF",  "stat",     X_stat),
        ("SVM", "stat",     X_stat),
    ]

    for model_name, feat_name, X_feat in classic_experiments:
        print(f"\n{model_name} + {feat_name} (balanced per fold)")
        res, fold_df = lopo_classic(X_feat, y, groups, model_name, feat_name)
        print(f"  Acc={res['accuracy'][0]:.4f}+-{res['accuracy'][1]:.4f}  "
              f"MacroF1={res['macro_f1'][0]:.4f}+-{res['macro_f1'][1]:.4f}  "
              f"WeightedF1={res['weighted_f1'][0]:.4f}")
        results_rows.append({
            "model": model_name, "feature": feat_name,
            "accuracy":      round(res["accuracy"][0],      4),
            "acc_std":       round(res["accuracy"][1],      4),
            "macro_f1":      round(res["macro_f1"][0],      4),
            "macro_f1_std":  round(res["macro_f1"][1],      4),
            "weighted_f1":   round(res["weighted_f1"][0],   4),
        })
        foldwise_rows.append(fold_df)

    cnn_experiments = [
        ("STFT", X_stft),
        ("MFCC", X_mfcc),
        ("Mel",  X_mel),
        ("GFCC", X_gfcc),
    ]

    for feat_name, X_img in cnn_experiments:
        print(f"\nCNN + {feat_name} (balanced per fold)")
        res, fold_df = lopo_cnn(X_img, y, groups, feat_name)
        print(f"  Acc={res['accuracy'][0]:.4f}+-{res['accuracy'][1]:.4f}  "
              f"MacroF1={res['macro_f1'][0]:.4f}+-{res['macro_f1'][1]:.4f}  "
              f"WeightedF1={res['weighted_f1'][0]:.4f}")
        results_rows.append({
            "model": "CNN", "feature": feat_name,
            "accuracy":      round(res["accuracy"][0],      4),
            "acc_std":       round(res["accuracy"][1],      4),
            "macro_f1":      round(res["macro_f1"][0],      4),
            "macro_f1_std":  round(res["macro_f1"][1],      4),
            "weighted_f1":   round(res["weighted_f1"][0],   4),
        })
        foldwise_rows.append(fold_df)

    df = pd.DataFrame(results_rows).sort_values("macro_f1", ascending=False)
    out_csv  = SCRIPT_DIR / "pipeline_lopo_multiclass_balanced_results.csv"
    fold_csv = SCRIPT_DIR / "pipeline_lopo_multiclass_balanced_foldwise.csv"
    df.to_csv(out_csv, index=False)
    pd.concat(foldwise_rows, ignore_index=True).to_csv(fold_csv, index=False)

    print("\n" + "=" * 72)
    print("  RANKED RESULTS — Balanced LOPO Multiclass")
    print("=" * 72)
    print(df.to_string(index=False))
    print(f"\nResults saved: {out_csv}")
    print(f"Fold-wise saved: {fold_csv}")
    print("\nBalanced multiclass LOPO pipeline complete.")


if __name__ == "__main__":
    main()
