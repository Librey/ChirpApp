"""
SensEat — LOPO Multiclass Classification Pipeline
=================================================
Best technical implementation for method selection.

Participants:
    022–041

Food codes used:
    01, 02, 04, 05, 08, 09, 10

Design choice:
- Each participant intentionally consumed only a subset of foods.
- Therefore, in each LOPO fold, training/evaluation is restricted to
  the food classes present in the held-out participant.

Special handling:
- Participant 039 files starting with chirp_chips_... are mapped to Coke (10)
- Files with suffixes like " (1)" or " (2)" are valid and included

Outputs:
- pipeline_lopo_multiclass_results.csv
- pipeline_lopo_multiclass_foldwise.csv
- figures/multi/lopo_multiclass_cm_*.png
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

from scipy.signal import butter, lfilter
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

SAMPLE_RATE = 44100
LOWCUT = 17500.0
HIGHCUT = 20500.0
FILTER_ORDER = 6

CHIRP_PERIOD_S = 1.5
SAMPLES_PER_CHUNK = int(SAMPLE_RATE * CHIRP_PERIOD_S)

# Overlapping hop for data augmentation
HOP_S = 0.5
HOP_SAMPLES = int(SAMPLE_RATE * HOP_S)

# Approximate direct-path duration to suppress at start of each chirp
DIRECT_PATH_SAMPLES = int((0.30 / 343.0) * SAMPLE_RATE)

# Reference chirp parameters (from Android app — setting 2)
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
DATA_DIR = SCRIPT_DIR / "data"
FIG_DIR = SCRIPT_DIR / "figures" / "multi"
FIG_DIR.mkdir(parents=True, exist_ok=True)

EATING_FOLDERS = [f"{i:03d}" for i in range(22, 42)]

VALID_FOOD_CODES = {1, 2, 4, 5, 8, 9, 10}

FOOD_NAMES = {
    1: "Tortilla",
    2: "Mandarin",
    4: "Cheeze_It",
    5: "Carrots",
    8: "Noodles",
    9: "Water",
    10: "Coke",
}

IRB_RE = re.compile(r"^(\d+)_(\d{3})_(\d{2})_(\d{2})(?:\s*\(\d+\))?$")
OLD_FMT_RE = re.compile(r"^chirp_chips_")


# ─────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────
def butter_bandpass_filter(data, lowcut=LOWCUT, highcut=HIGHCUT, fs=SAMPLE_RATE, order=FILTER_ORDER):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    if low <= 0 or high >= 1 or low >= high:
        return data
    b, a = butter(order, [low, high], btype="band")
    return lfilter(b, a, data)


def generate_reference_chirp():
    """Synthesize transmitted chirp exactly as Android app (setting 2)."""
    n     = CHIRP_DUR_SAMPLES
    dur_s = n / SAMPLE_RATE
    chirp = np.zeros(n, dtype=np.float32)
    for i in range(n):
        t        = i / SAMPLE_RATE
        freq     = CHIRP_START_HZ + (CHIRP_END_HZ - CHIRP_START_HZ) * (t / dur_s)
        chirp[i] = np.sin(2 * np.pi * freq * t)
    return butter_bandpass_filter(chirp)


def tap_profile_features(segment, ref_chirp):
    """FFT cross-correlation tap profile — echo strength at 10–50 cm."""
    rx  = segment[:CHIRP_DUR_SAMPLES].astype(np.float64)
    ref = ref_chirp.astype(np.float64)

    Nfft = 1 << (len(rx) + len(ref) - 1).bit_length()
    taps = np.abs(np.fft.ifft(np.fft.fft(rx, Nfft) * np.conj(np.fft.fft(ref, Nfft))))

    target = taps[TARGET_TAP_MIN:TARGET_TAP_MAX].astype(np.float32)
    peak   = target.max()
    if peak > 0:
        target = target / peak

    stats = np.array([
        float(np.max(target)),
        float(np.mean(target)),
        float(np.std(target)),
        float(np.argmax(target)),
        float(np.sum(target)),
        float(np.percentile(target, 75)),
        float(np.percentile(target, 25)),
    ], dtype=np.float32)

    return np.concatenate([target, stats])


def load_pcm_left_channel(filepath):
    raw = np.fromfile(filepath, dtype=np.int16)
    if raw.size % 2 != 0:
        raw = raw[:-1]
    stereo = raw.reshape(-1, 2)
    return stereo[:, 0].astype(np.float32) / 32768.0


def preprocess_and_segment(filepath):
    signal = load_pcm_left_channel(filepath)
    signal = butter_bandpass_filter(signal)

    chunks = []
    for i in range(0, len(signal) - SAMPLES_PER_CHUNK + 1, HOP_SAMPLES):
        seg = signal[i:i + SAMPLES_PER_CHUNK]
        if len(seg) != SAMPLES_PER_CHUNK:
            continue
        seg = seg[DIRECT_PATH_SAMPLES:]
        seg = np.pad(seg, (0, DIRECT_PATH_SAMPLES))  # keep fixed length
        chunks.append(seg)

    return chunks


# ─────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────
def get_statistical_features(segment):
    zcr = librosa.feature.zero_crossing_rate(segment)[0]
    return np.array([
        np.mean(segment),
        np.std(segment),
        np.max(segment),
        np.min(segment),
        np.mean(segment ** 2),
        np.mean(zcr),
    ], dtype=np.float32)


def get_wavelet_features(segment, wavelet="db4", level=4):
    try:
        coeffs = pywt.wavedec(segment, wavelet, level=level)
    except Exception:
        coeffs = pywt.wavedec(segment, wavelet, level=1)

    feats = []
    for c in coeffs[1:]:
        feats.extend([
            np.mean(c),
            np.std(c),
            np.max(c),
            np.min(c),
            np.mean(c ** 2),
        ])
    return np.array(feats, dtype=np.float32)


def get_combined_flat_features(segment, ref_chirp):
    return np.concatenate([
        get_statistical_features(segment),
        get_wavelet_features(segment),
        tap_profile_features(segment, ref_chirp),
    ])


def get_stft_image(segment, n_fft=2048, fixed_size=(64, 64)):
    try:
        D = np.abs(librosa.stft(segment, n_fft=n_fft))
        S_db = librosa.amplitude_to_db(D, ref=np.max)
        S_db = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)
        resized = cv2.resize(S_db, fixed_size, interpolation=cv2.INTER_CUBIC)
        return resized.astype(np.float32)[..., np.newaxis]
    except Exception:
        return np.zeros((*fixed_size, 1), dtype=np.float32)


def get_mfcc_fixed(segment, sr=SAMPLE_RATE, n_mfcc=40, fixed_frames=64):
    # Baseline only
    hop = max(64, int(np.floor(len(segment) / (fixed_frames - 1))))
    mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=n_mfcc, hop_length=hop)

    if mfcc.shape[1] < fixed_frames:
        mfcc = np.pad(
            mfcc,
            ((0, 0), (0, fixed_frames - mfcc.shape[1])),
            mode="constant",
            constant_values=mfcc.min(),
        )
    else:
        mfcc = mfcc[:, :fixed_frames]

    mfcc = (mfcc - mfcc.min()) / (mfcc.max() - mfcc.min() + 1e-8)
    return mfcc.astype(np.float32)[..., np.newaxis]


# ─────────────────────────────────────────
# DATASET BUILDER
# ─────────────────────────────────────────
def parse_food_code(pcm_path: Path) -> int:
    stem = pcm_path.stem

    if OLD_FMT_RE.match(stem):
        return 10

    m = IRB_RE.match(stem)
    if m:
        return int(m.group(3))

    return -1


CACHE_FILE = SCRIPT_DIR / "feature_cache_multiclass.npz"

def build_dataset():
    if CACHE_FILE.exists():
        print(f"  Loading cached features from {CACHE_FILE.name} ...")
        d = np.load(CACHE_FILE, allow_pickle=False)
        return (d["feat_stat"], d["feat_wave"], d["feat_combined"],
                d["feat_stft"], d["feat_mfcc"], d["labels"], d["participant_ids"])

    print("  No cache — extracting features (will be cached for next run) ...")
    ref_chirp = generate_reference_chirp()
    feat_stat = []
    feat_wave = []
    feat_combined = []
    feat_stft = []
    feat_mfcc = []
    labels = []
    participant_ids = []

    print("\n📦 Loading multiclass data (participants 022–041) ...\n")
    print("   Segmentation: 1.5 s window, 0.5 s hop (overlapping)\n")

    for folder_name in EATING_FOLDERS:
        folder_path = DATA_DIR / folder_name
        if not folder_path.exists():
            print(f"  ⚠ Missing: {folder_path}")
            continue

        participant_int = int(folder_name)

        pcm_files = sorted([
            f for f in folder_path.glob("*.pcm")
            if "_idleTail" not in f.stem and "_meta" not in f.stem
        ])

        loaded = 0
        food_codes_seen = set()

        for pcm in pcm_files:
            food_code = parse_food_code(pcm)
            if food_code not in VALID_FOOD_CODES:
                continue

            food_codes_seen.add(food_code)

            chunks = preprocess_and_segment(str(pcm))
            for seg in chunks:
                feat_stat.append(get_statistical_features(seg))
                feat_wave.append(get_wavelet_features(seg))
                feat_combined.append(get_combined_flat_features(seg, ref_chirp))
                feat_stft.append(get_stft_image(seg))
                feat_mfcc.append(get_mfcc_fixed(seg))
                labels.append(food_code)
                participant_ids.append(participant_int)
                loaded += 1

        print(f"  📂 {folder_name}: {loaded} segments | foods: {sorted(food_codes_seen)}")

    out = (np.array(feat_stat), np.array(feat_wave), np.array(feat_combined),
           np.array(feat_stft), np.array(feat_mfcc),
           np.array(labels), np.array(participant_ids))

    np.savez_compressed(CACHE_FILE,
                        feat_stat=out[0], feat_wave=out[1], feat_combined=out[2],
                        feat_stft=out[3], feat_mfcc=out[4],
                        labels=out[5], participant_ids=out[6])
    print(f"  Features cached to {CACHE_FILE.name}")
    return out


# ─────────────────────────────────────────
# METRICS / VIS
# ─────────────────────────────────────────
def summarize_metrics(metric_lists):
    return {k: (np.mean(v), np.std(v)) for k, v in metric_lists.items()}


def save_confusion_matrix(y_true, y_pred, feature_name):
    unique_labels = sorted(set(y_true))
    display_labels = [FOOD_NAMES[k] for k in unique_labels]

    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_pct,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=display_labels,
        yticklabels=display_labels,
    )
    plt.title(f"[Multiclass] LOPO Confusion Matrix (%): {feature_name}")
    plt.ylabel("True Food")
    plt.xlabel("Predicted Food")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    out = FIG_DIR / f"lopo_multiclass_cm_{feature_name}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"    💾 Confusion matrix saved: {out.name}")


# ─────────────────────────────────────────
# LOPO — CLASSICAL MODELS
# ─────────────────────────────────────────
def lopo_classic(X, y, groups, model_name="RF", feature_name=""):
    logo = LeaveOneGroupOut()
    fold_metrics = defaultdict(list)
    all_y_true, all_y_pred = [], []
    fold_rows = []

    for train_idx, test_idx in logo.split(X, y, groups):
        participant_id = int(groups[test_idx[0]])
        X_te = X[test_idx]
        y_te = y[test_idx]

        # Restrict to classes present in held-out participant
        test_classes = np.unique(y_te)
        train_mask = np.isin(y[train_idx], test_classes)
        X_tr = X[train_idx][train_mask]
        y_tr = y[train_idx][train_mask]

        if len(np.unique(y_tr)) < 2:
            continue

        if model_name == "RF":
            clf = RandomForestClassifier(
                n_estimators=150,
                class_weight="balanced",
                random_state=SEED,
            )
        elif model_name == "SVM":
            clf = SVC(
                kernel="rbf",
                class_weight="balanced",
                random_state=SEED,
            )
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", clf),
        ])
        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te)

        acc = accuracy_score(y_te, y_pred)
        macro_f1 = f1_score(y_te, y_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_te, y_pred, average="weighted", zero_division=0)

        fold_metrics["accuracy"].append(acc)
        fold_metrics["macro_f1"].append(macro_f1)
        fold_metrics["weighted_f1"].append(weighted_f1)

        fold_rows.append({
            "model": model_name,
            "feature": feature_name,
            "participant": participant_id,
            "num_classes": len(test_classes),
            "classes": ",".join(map(str, sorted(test_classes))),
            "accuracy": acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
        })

        all_y_true.extend(y_te)
        all_y_pred.extend(y_pred)

    return summarize_metrics(fold_metrics), np.array(all_y_true), np.array(all_y_pred), pd.DataFrame(fold_rows)


# ─────────────────────────────────────────
# LOPO — CNN
# ─────────────────────────────────────────
def build_cnn_multiclass(input_shape, num_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def lopo_cnn_multiclass(X, y, groups, input_shape, feature_name="", epochs=10, batch_size=32):
    logo = LeaveOneGroupOut()
    fold_metrics = defaultdict(list)
    all_y_true, all_y_pred = [], []
    fold_rows = []

    for train_idx, test_idx in logo.split(X, y, groups):
        participant_id = int(groups[test_idx[0]])
        X_te = X[test_idx].astype(np.float32)
        y_te = y[test_idx]

        # Restrict to classes present in held-out participant
        test_classes = np.unique(y_te)
        train_mask = np.isin(y[train_idx], test_classes)
        X_tr = X[train_idx][train_mask].astype(np.float32)
        y_tr = y[train_idx][train_mask]

        if len(np.unique(y_tr)) < 2:
            continue

        code_to_idx = {c: i for i, c in enumerate(sorted(test_classes))}
        idx_to_code = {i: c for c, i in code_to_idx.items()}

        y_tr_mapped = np.array([code_to_idx[c] for c in y_tr])
        num_classes = len(test_classes)

        mean = X_tr.mean()
        std = X_tr.std() if X_tr.std() > 0 else 1.0
        X_tr = (X_tr - mean) / std
        X_te = (X_te - mean) / std

        class_weight = {}
        for cls in range(num_classes):
            n_cls = np.sum(y_tr_mapped == cls)
            class_weight[cls] = len(y_tr_mapped) / (num_classes * (n_cls + 1e-8))

        model = build_cnn_multiclass(input_shape, num_classes)
        es = EarlyStopping(patience=3, restore_best_weights=True)

        model.fit(
            X_tr,
            y_tr_mapped,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            callbacks=[es],
            class_weight=class_weight,
            verbose=0,
        )

        y_pred_mapped = np.argmax(model.predict(X_te, verbose=0), axis=1)
        y_pred_orig = np.array([idx_to_code[i] for i in y_pred_mapped])

        acc = accuracy_score(y_te, y_pred_orig)
        macro_f1 = f1_score(y_te, y_pred_orig, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_te, y_pred_orig, average="weighted", zero_division=0)

        fold_metrics["accuracy"].append(acc)
        fold_metrics["macro_f1"].append(macro_f1)
        fold_metrics["weighted_f1"].append(weighted_f1)

        fold_rows.append({
            "model": "CNN",
            "feature": feature_name,
            "participant": participant_id,
            "num_classes": len(test_classes),
            "classes": ",".join(map(str, sorted(test_classes))),
            "accuracy": acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
        })

        all_y_true.extend(y_te)
        all_y_pred.extend(y_pred_orig)

        tf.keras.backend.clear_session()

    return summarize_metrics(fold_metrics), np.array(all_y_true), np.array(all_y_pred), pd.DataFrame(fold_rows)


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def main():
    print("=" * 72)
    print("  SensEat — LOPO Multiclass Classification (Participants 022–041)")
    print(f"  Food classes: {list(FOOD_NAMES.values())}")
    print("=" * 72)

    X_stat, X_wave, X_combined, X_stft, X_mfcc, y, groups = build_dataset()

    if len(y) == 0:
        print("❌ No segments found. Check DATA_DIR path.")
        return

    print(f"\n✅ Total segments: {len(y)}")
    print(f"   Participants: {len(np.unique(groups))}")
    for code in sorted(VALID_FOOD_CODES):
        count = int(np.sum(y == code))
        print(f"   {FOOD_NAMES[code]:15s} (code {code:02d}): {count} segments")

    results_rows = []
    foldwise_rows = []

    experiments = [
        ("RF", "statistical", X_stat),
        ("SVM", "statistical", X_stat),
        ("RF", "wavelet_DWT", X_wave),
        ("SVM", "wavelet_DWT", X_wave),
        ("RF", "stat+wavelet_DWT", X_combined),
        ("SVM", "stat+wavelet_DWT", X_combined),
    ]

    for model_name, feature_name, X in experiments:
        print(f"\n🔧 {model_name} + {feature_name}")
        res, y_true, y_pred, fold_df = lopo_classic(X, y, groups, model_name=model_name, feature_name=feature_name)
        print(
            f"   LOPO → Acc={res['accuracy'][0]:.4f}±{res['accuracy'][1]:.4f}  "
            f"MacroF1={res['macro_f1'][0]:.4f}±{res['macro_f1'][1]:.4f}  "
            f"WeightedF1={res['weighted_f1'][0]:.4f}±{res['weighted_f1'][1]:.4f}"
        )
        save_confusion_matrix(y_true, y_pred, f"{model_name}_{feature_name}")
        results_rows.append({
            "model": model_name,
            "feature": feature_name,
            "accuracy": round(res["accuracy"][0], 4),
            "acc_std": round(res["accuracy"][1], 4),
            "macro_f1": round(res["macro_f1"][0], 4),
            "macro_f1_std": round(res["macro_f1"][1], 4),
            "weighted_f1": round(res["weighted_f1"][0], 4),
            "weighted_f1_std": round(res["weighted_f1"][1], 4),
        })
        foldwise_rows.append(fold_df)

    cnn_experiments = [
        ("STFT", X_stft),
        ("MFCC", X_mfcc),
    ]

    for feature_name, X in cnn_experiments:
        print(f"\n🧠 CNN + {feature_name}")
        res, y_true, y_pred, fold_df = lopo_cnn_multiclass(
            X, y, groups, input_shape=X[0].shape, feature_name=feature_name
        )
        print(
            f"   LOPO → Acc={res['accuracy'][0]:.4f}±{res['accuracy'][1]:.4f}  "
            f"MacroF1={res['macro_f1'][0]:.4f}±{res['macro_f1'][1]:.4f}  "
            f"WeightedF1={res['weighted_f1'][0]:.4f}±{res['weighted_f1'][1]:.4f}"
        )
        save_confusion_matrix(y_true, y_pred, f"CNN_{feature_name}")
        results_rows.append({
            "model": "CNN",
            "feature": feature_name,
            "accuracy": round(res["accuracy"][0], 4),
            "acc_std": round(res["accuracy"][1], 4),
            "macro_f1": round(res["macro_f1"][0], 4),
            "macro_f1_std": round(res["macro_f1"][1], 4),
            "weighted_f1": round(res["weighted_f1"][0], 4),
            "weighted_f1_std": round(res["weighted_f1"][1], 4),
        })
        foldwise_rows.append(fold_df)

    df = pd.DataFrame(results_rows).sort_values("macro_f1", ascending=False)
    out_csv = SCRIPT_DIR / "pipeline_lopo_multiclass_results.csv"
    df.to_csv(out_csv, index=False)

    foldwise_df = pd.concat(foldwise_rows, ignore_index=True)
    foldwise_csv = SCRIPT_DIR / "pipeline_lopo_multiclass_foldwise.csv"
    foldwise_df.to_csv(foldwise_csv, index=False)

    print("\n" + "=" * 72)
    print("  RANKED RESULTS — LOPO Multiclass")
    print("=" * 72)
    print(df.to_string(index=False))
    print(f"\n💾 Results saved to {out_csv}")
    print(f"💾 Fold-wise results saved to {foldwise_csv}")
    print("\n✅ Multiclass LOPO pipeline complete.")


if __name__ == "__main__":
    main()