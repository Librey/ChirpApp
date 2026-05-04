"""
SensEat — Robustness Evaluation Pipeline
=========================================
Cross-condition evaluation: train on 41-participant baseline (office, 30cm, 120deg,
primary phone), test on each variation recorded by participant 042.

Conditions tested vs baseline:
  home        : different environment (home vs office)
  dist_15cm   : distance 15 cm  (baseline 30 cm)
  dist_20cm   : distance 20 cm
  dist_50cm   : distance 50 cm
  angle_60    : device angle 60 deg  (baseline 120 deg)
  angle_90    : device angle 90 deg
  angle_180   : device angle 180 deg
  noise_60db  : white noise 60 dB background
  noise_70db  : white noise 70 dB background
  samsung_s23 : different phone (Samsung S23)

Tasks:
  T1  Binary   : eating (carrots + yogurt) vs idle
  T2  3-class  : idle vs carrots vs yogurt

Models:
  RF   on flat features  (best stable model, fast)
  CNN-STFT               (best binary model, slower)

IMPORTANT: Both models are trained ONCE on the 41-participant baseline.
The same trained model is then applied to every condition — fair comparison.

Output:
  pipeline_robustness_results.csv
  figures/robustness/robustness_summary.png
"""

import os, re, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import pywt
import cv2

from scipy.signal import butter, lfilter, hilbert
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score,
                              precision_score, recall_score)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

# ── CONFIG ──────────────────────────────────────────────────────────────────
SAMPLE_RATE         = 44100
LOWCUT              = 17500.0
HIGHCUT             = 20500.0
FILTER_ORDER        = 6
CHIRP_PERIOD_S      = 1.5
SAMPLES_PER_CHUNK   = int(SAMPLE_RATE * CHIRP_PERIOD_S)
HOP_SAMPLES         = int(SAMPLE_RATE * 0.5)
DIRECT_PATH_SAMPLES = int((0.30 / 343.0) * SAMPLE_RATE)
CHIRP_START_HZ      = 18000.0
CHIRP_END_HZ        = 20000.0
CHIRP_DUR_SAMPLES   = int(SAMPLE_RATE * 1.0)
TARGET_TAP_MIN      = int(2 * 0.10 / 343.0 * SAMPLE_RATE)
TARGET_TAP_MAX      = int(2 * 0.50 / 343.0 * SAMPLE_RATE)
SEED                = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)
RNG = np.random.default_rng(SEED)

SCRIPT_DIR  = Path(__file__).resolve().parent
FIG_DIR     = SCRIPT_DIR / "figures" / "robustness"
FIG_DIR.mkdir(parents=True, exist_ok=True)

EVAL_ROOT        = SCRIPT_DIR / "Environment_Evaluation_Data"
BINARY_CACHE     = SCRIPT_DIR / "feature_cache_binary_v4.npz"
MULTICLASS_CACHE = SCRIPT_DIR / "feature_cache_multiclass_balanced_v2.npz"

# Condition folder mapping
CONDITIONS = {
    "home":       EVAL_ROOT / "Diff-environment",
    "angle_60":   EVAL_ROOT / "Diff angle" / "60 degrees",
    "angle_90":   EVAL_ROOT / "Diff angle" / "90 degrees",
    "angle_180":  EVAL_ROOT / "Diff angle" / "180",
    "dist_15cm":  EVAL_ROOT / "Diff distance" / "15cm",
    "dist_20cm":  EVAL_ROOT / "Diff distance" / "20cm",
    "dist_50cm":  EVAL_ROOT / "Diff distance" / "50 cm",
    "noise_60db": EVAL_ROOT / "Diff noise" / "60 db",
    "noise_70db": EVAL_ROOT / "Diff noise" / "70 db",
    "samsung":    EVAL_ROOT / "Diff mobile-s23",
}

FOOD_NAMES = {0: "Idle", 5: "Carrots", 7: "Yogurt"}

# IRB regex -- handles Windows copy suffix "(N)"
IRB_RE = re.compile(r"^(\d+)_(\d{3})_(\d{2})_(\d{2})(?: \(\d+\))?$")


# ── PREPROCESSING ───────────────────────────────────────────────────────────
def butter_bandpass(data):
    nyq  = 0.5 * SAMPLE_RATE
    low  = LOWCUT / nyq;  high = HIGHCUT / nyq
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


def cancel_direct_path(segment, ref):
    s = segment.astype(np.float64);  d = ref.astype(np.float64)
    c = np.dot(s, d) / (np.dot(d, d) + 1e-8)
    return (s - c * d).astype(np.float32)


def compute_ref_direct(idle_files, max_files=5):
    segs = []
    for f in list(idle_files)[:max_files]:
        sig = butter_bandpass(load_pcm(str(f)))
        for i in range(0, len(sig) - SAMPLES_PER_CHUNK + 1, HOP_SAMPLES):
            seg = sig[i: i + SAMPLES_PER_CHUNK]
            if len(seg) == SAMPLES_PER_CHUNK:
                segs.append(seg.astype(np.float64))
    if not segs:
        return None
    return np.mean(segs, axis=0).astype(np.float32)


def preprocess_and_segment(filepath, ref_direct=None):
    signal = butter_bandpass(load_pcm(filepath))
    segs   = []
    for i in range(0, len(signal) - SAMPLES_PER_CHUNK + 1, HOP_SAMPLES):
        seg = signal[i: i + SAMPLES_PER_CHUNK]
        if len(seg) != SAMPLES_PER_CHUNK:
            continue
        seg = seg[DIRECT_PATH_SAMPLES:]
        seg = np.pad(seg, (0, DIRECT_PATH_SAMPLES))
        if ref_direct is not None:
            seg = cancel_direct_path(seg, ref_direct)
        segs.append(seg)
    return segs


# ── FEATURE EXTRACTION ──────────────────────────────────────────────────────
def tap_profile_features(segment, ref_chirp):
    rx  = segment[:CHIRP_DUR_SAMPLES].astype(np.float64)
    ref = ref_chirp.astype(np.float64)
    Nfft = 1 << (len(rx) + len(ref) - 1).bit_length()
    taps = np.abs(np.fft.ifft(
        np.fft.fft(rx, Nfft) * np.conj(np.fft.fft(ref, Nfft))))
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
    env = np.abs(hilbert(x))
    return np.array([
        np.mean(x), np.std(x), np.max(x), np.min(x), np.mean(x**2),
        np.mean(zcr), np.mean(env), np.std(env), np.max(env),
    ], dtype=np.float32)


def spectral_features(x):
    S         = np.abs(librosa.stft(x, n_fft=2048))
    centroid  = librosa.feature.spectral_centroid(S=S, sr=SAMPLE_RATE)[0]
    bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=SAMPLE_RATE)[0]
    rolloff   = librosa.feature.spectral_rolloff(S=S, sr=SAMPLE_RATE)[0]
    flux      = librosa.onset.onset_strength(
                    S=librosa.amplitude_to_db(S), sr=SAMPLE_RATE)
    freqs     = librosa.fft_frequencies(sr=SAMPLE_RATE, n_fft=2048)
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


def stft_image(x, size=(64, 64)):
    try:
        D = np.abs(librosa.stft(x, n_fft=2048))
        D = librosa.amplitude_to_db(D, ref=np.max)
        D = (D - D.min()) / (D.max() - D.min() + 1e-8)
        return cv2.resize(D, size)[..., np.newaxis].astype(np.float32)
    except Exception:
        return np.zeros((*size, 1), dtype=np.float32)


# ── CONDITION DATA LOADER ───────────────────────────────────────────────────
def load_condition(folder: Path, ref_chirp):
    """
    Load all PCM files in a condition folder.
    Returns X_flat, X_stft, y_binary, y_3class.
    food_code: 0=idle, 5=carrots, 7=yogurt
    binary label: 0=idle, 1=eating
    3-class label: 0=idle, 1=carrots, 2=yogurt
    """
    pcm_files = sorted([
        f for f in folder.glob("*.pcm")
        if "idleTail" not in f.name and "meta" not in f.name
    ])

    if not pcm_files:
        print(f"    WARNING: no PCM files in {folder}")
        return None

    # Use idle files from this condition to compute reference direct path
    idle_files = []
    for f in pcm_files:
        m = IRB_RE.match(f.stem)
        if m and int(m.group(3)) == 0:
            idle_files.append(f)

    ref_direct = compute_ref_direct(idle_files) if idle_files else None

    flat_list, stft_list, yb_list, y3_list = [], [], [], []

    for f in pcm_files:
        m = IRB_RE.match(f.stem)
        if not m:
            continue
        food_code = int(m.group(3))
        if food_code not in (0, 5, 7):
            continue

        segs = preprocess_and_segment(str(f), ref_direct)
        for seg in segs:
            flat_list.append(combined_flat(seg, ref_chirp))
            stft_list.append(stft_image(seg))
            yb_list.append(0 if food_code == 0 else 1)
            y3_list.append({0: 0, 5: 1, 7: 2}[food_code])

    if not flat_list:
        return None

    counts = {k: yb_list.count(k) for k in set(yb_list)}
    print(f"    Loaded {len(flat_list)} segments: "
          f"idle={counts.get(0,0)}  eating={counts.get(1,0)}")

    return (np.array(flat_list, dtype=np.float32),
            np.array(stft_list, dtype=np.float32),
            np.array(yb_list,   dtype=np.int32),
            np.array(y3_list,   dtype=np.int32))


# ── CNN BUILDER ─────────────────────────────────────────────────────────────
def build_cnn_stft(stft_shape, task="binary"):
    inp = Input(shape=stft_shape)
    x   = layers.Conv2D(32, (3,3), activation="relu", padding="same")(inp)
    x   = layers.BatchNormalization()(x)
    x   = layers.MaxPooling2D((2,2))(x)
    x   = layers.Conv2D(64, (3,3), activation="relu", padding="same")(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.MaxPooling2D((2,2))(x)
    x   = layers.Conv2D(128, (3,3), activation="relu", padding="same")(x)
    x   = layers.GlobalAveragePooling2D()(x)
    x   = layers.Dense(128, activation="relu")(x)
    x   = layers.Dropout(0.4)(x)
    if task == "binary":
        out = layers.Dense(1, activation="sigmoid")(x)
        m   = models.Model(inp, out)
        m.compile("adam", "binary_crossentropy", metrics=["accuracy"])
    else:
        out = layers.Dense(3, activation="softmax")(x)
        m   = models.Model(inp, out)
        m.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


# ── UNDERSAMPLE ─────────────────────────────────────────────────────────────
def undersample(X_flat, X_stft, y):
    """Balance classes by undersampling majority to minority size."""
    classes, counts = np.unique(y, return_counts=True)
    n_min = counts.min()
    idx   = []
    for c in classes:
        ci = np.where(y == c)[0]
        idx.append(RNG.choice(ci, size=n_min, replace=False))
    idx = np.concatenate(idx)
    RNG.shuffle(idx)
    return X_flat[idx], X_stft[idx], y[idx]


# ── TRAINING DATA ────────────────────────────────────────────────────────────
def load_training_binary():
    """All 41 participants: eating vs idle from binary cache."""
    d      = np.load(BINARY_CACHE, allow_pickle=False)
    X_flat = d["X_flat"].astype(np.float32)
    X_stft = d["X_stft"].astype(np.float32)
    y      = d["y"].astype(np.int32)
    return X_flat, X_stft, y


def load_training_3class():
    """
    Idle from binary cache + carrots + yogurt from multiclass cache.
    Binary cache has 148 flat features; multiclass has 136.
    Truncate to the smaller (136) so all sources align.

    NOTE: yogurt (code 7) may not be present in the multiclass cache
    if the 41-participant study did not include yogurt. We print a
    warning and fall back to idle-vs-carrots (2-class) if yogurt is absent.
    """
    bd = np.load(BINARY_CACHE,     allow_pickle=False)
    md = np.load(MULTICLASS_CACHE, allow_pickle=False)

    idle_mask = bd["y"] == 0
    carr_mask = md["y"] == 5
    yog_mask  = md["y"] == 7

    n_feat = md["X_flat"].shape[1]   # 136

    n_idle = int(idle_mask.sum())
    n_carr = int(carr_mask.sum())
    n_yog  = int(yog_mask.sum())
    print(f"  3-class training pool: idle={n_idle}  carrots={n_carr}  yogurt={n_yog}")

    flat_parts = [bd["X_flat"][idle_mask, :n_feat],
                  md["X_flat"][carr_mask]]
    stft_parts = [bd["X_stft"][idle_mask],
                  md["X_stft"][carr_mask]]
    y_parts    = [np.zeros(n_idle, dtype=np.int32),
                  np.ones(n_carr,  dtype=np.int32)]

    if n_yog > 0:
        flat_parts.append(md["X_flat"][yog_mask])
        stft_parts.append(md["X_stft"][yog_mask])
        y_parts.append(np.full(n_yog, 2, dtype=np.int32))
    else:
        print("  WARNING: yogurt (code 7) not found in multiclass cache. "
              "3-class task uses idle vs carrots only (yogurt test samples "
              "will be treated as unknown).")

    X_flat = np.concatenate(flat_parts)
    X_stft = np.concatenate(stft_parts)
    y      = np.concatenate(y_parts)
    return X_flat, X_stft, y, n_feat


# ── TRAIN MODELS ONCE ────────────────────────────────────────────────────────
def train_rf_binary(X_flat, X_stft, y):
    """Train and return RF + scaler for binary classification."""
    Xf_tr, _, y_b = undersample(X_flat, X_stft, y)
    sc = StandardScaler()
    Xf_n = sc.fit_transform(Xf_tr)
    rf = RandomForestClassifier(200, random_state=SEED, n_jobs=-1,
                                class_weight="balanced")
    rf.fit(Xf_n, y_b)
    classes = sorted(np.unique(y_b).tolist())
    print(f"  RF binary: trained on {len(y_b)} samples  classes={classes}")
    return rf, sc


def train_cnn_binary(X_flat, X_stft, y):
    """Train CNN-STFT once. Returns model + normalization stats."""
    mn = float(X_stft.mean())
    sd = float(X_stft.std()) or 1.0
    Xs_norm = (X_stft - mn) / sd
    _, Xs_u, y_u = undersample(X_flat, Xs_norm, y)
    model = build_cnn_stft(Xs_norm.shape[1:], "binary")
    cb = [EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
          ReduceLROnPlateau(monitor="val_loss", patience=4, factor=0.5, verbose=0)]
    model.fit(Xs_u, y_u, epochs=30, batch_size=32,
              validation_split=0.1, callbacks=cb, verbose=0)
    print(f"  CNN binary: trained on {len(y_u)} samples")
    return model, mn, sd


def train_rf_3class(X_flat, X_stft, y):
    """Train and return RF + scaler for 3-class classification."""
    Xf_tr, _, y_b = undersample(X_flat, X_stft, y)
    sc = StandardScaler()
    Xf_n = sc.fit_transform(Xf_tr)
    rf = RandomForestClassifier(200, random_state=SEED, n_jobs=-1)
    rf.fit(Xf_n, y_b)
    classes = sorted(np.unique(y_b).tolist())
    print(f"  RF 3-class: trained on {len(y_b)} samples  classes={classes}")
    return rf, sc


# ── EVALUATION (pre-trained models, no re-training) ──────────────────────────
def eval_binary_on_cond(rf, sc, cnn_model, cnn_mn, cnn_sd,
                        X_flat_te, X_stft_te, y_te):
    """Apply pre-trained binary RF and CNN to one condition's test data."""
    results = {}

    # RF
    Xf_te_n = sc.transform(X_flat_te)
    yp = rf.predict(Xf_te_n)
    results["RF_acc"]  = round(accuracy_score(y_te, yp), 4)
    results["RF_prec"] = round(precision_score(y_te, yp, zero_division=0), 4)
    results["RF_rec"]  = round(recall_score(y_te, yp,    zero_division=0), 4)
    results["RF_f1"]   = round(f1_score(y_te, yp,        zero_division=0), 4)
    print(f"    RF   -> Acc={results['RF_acc']:.3f}  "
          f"P={results['RF_prec']:.3f}  R={results['RF_rec']:.3f}  "
          f"F1={results['RF_f1']:.3f}")

    # CNN-STFT
    Xs_te_n = (X_stft_te - cnn_mn) / cnn_sd
    probs   = cnn_model.predict(Xs_te_n, verbose=0).ravel()
    yp_cnn  = (probs > 0.5).astype(int)
    results["CNN_acc"]  = round(accuracy_score(y_te, yp_cnn), 4)
    results["CNN_prec"] = round(precision_score(y_te, yp_cnn, zero_division=0), 4)
    results["CNN_rec"]  = round(recall_score(y_te, yp_cnn,    zero_division=0), 4)
    results["CNN_f1"]   = round(f1_score(y_te, yp_cnn,        zero_division=0), 4)
    print(f"    CNN  -> Acc={results['CNN_acc']:.3f}  "
          f"P={results['CNN_prec']:.3f}  R={results['CNN_rec']:.3f}  "
          f"F1={results['CNN_f1']:.3f}")

    return results


def eval_3class_on_cond(rf, sc, n_feat, X_flat_te, y_te):
    """Apply pre-trained 3-class RF to one condition's test data."""
    # Test features were extracted with 148-dim pipeline; truncate to training dim
    X_flat_te = X_flat_te[:, :n_feat]
    Xf_te_n   = sc.transform(X_flat_te)
    yp        = rf.predict(Xf_te_n)

    macro_f1 = round(f1_score(y_te, yp, average="macro",    zero_division=0), 4)
    wtd_f1   = round(f1_score(y_te, yp, average="weighted", zero_division=0), 4)
    acc      = round(accuracy_score(y_te, yp), 4)
    per_cls  = f1_score(y_te, yp, labels=[0, 1, 2], average=None, zero_division=0)
    print(f"    3-cls -> Acc={acc:.3f}  MacroF1={macro_f1:.3f}  "
          f"[idle={per_cls[0]:.2f} carr={per_cls[1]:.2f} yog={per_cls[2]:.2f}]")
    return {"3c_acc": acc, "3c_macro_f1": macro_f1, "3c_wtd_f1": wtd_f1,
            "3c_f1_idle":    round(float(per_cls[0]), 4),
            "3c_f1_carrots": round(float(per_cls[1]), 4),
            "3c_f1_yogurt":  round(float(per_cls[2]), 4)}


# ── PLOTS ───────────────────────────────────────────────────────────────────
def plot_summary(df):
    conditions = df["condition"].tolist()
    x = np.arange(len(conditions))
    w = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("SensEat -- Robustness Evaluation\n"
                 "Train: 41-participant baseline  |  Test: each condition",
                 fontsize=13, fontweight="bold")

    # Binary F1
    ax = axes[0]
    ax.bar(x - w/2, df["RF_f1"],  w, label="RF",      color="#4C72B0")
    ax.bar(x + w/2, df["CNN_f1"], w, label="CNN-STFT", color="#55A868")
    ax.axhline(0.905, color="#4C72B0", linestyle="--", linewidth=1.2,
               alpha=0.6, label="RF baseline (0.905)")
    ax.axhline(0.946, color="#55A868", linestyle="--", linewidth=1.2,
               alpha=0.6, label="CNN baseline (0.946)")
    ax.set_xticks(x); ax.set_xticklabels(conditions, rotation=35, ha="right")
    ax.set_ylim(0, 1.1); ax.set_ylabel("F1"); ax.set_title("Binary: Eating vs Idle")
    ax.legend(fontsize=8); ax.grid(axis="y", linestyle="--", alpha=0.4)

    # 3-class MacroF1
    ax2 = axes[1]
    ax2.bar(x, df["3c_macro_f1"], color="#DD8452")
    ax2.axhline(0.264, color="red", linestyle="--", linewidth=1.2,
                alpha=0.7, label="RF baseline MacroF1 (0.264)")
    ax2.set_xticks(x); ax2.set_xticklabels(conditions, rotation=35, ha="right")
    ax2.set_ylim(0, 1.1); ax2.set_ylabel("Macro F1")
    ax2.set_title("3-Class: Idle vs Carrots vs Yogurt")
    ax2.legend(fontsize=9); ax2.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    out = FIG_DIR / "robustness_summary.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Chart saved: {out.name}")


# ── MAIN ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  SensEat -- Robustness Evaluation Pipeline")
    print("  Train: 41-participant baseline (office, 30cm, 120deg, primary phone)")
    print("  Test : participant 042 across 10 conditions")
    print("  Models trained ONCE; same model applied to all conditions.")
    print("=" * 70)

    ref_chirp = generate_reference_chirp()

    # ── Load caches ────────────────────────────────────────────────────────
    print("\nLoading training data from caches ...")
    Xf_bin, Xs_bin, yb = load_training_binary()
    Xf_3c,  Xs_3c,  y3, n_feat_3c = load_training_3class()
    print(f"  Binary  train: {len(yb)} segments "
          f"(eating={int(np.sum(yb==1))} idle={int(np.sum(yb==0))})")
    print(f"  3-class train: {len(y3)} segments "
          f"(idle={int(np.sum(y3==0))} carrot={int(np.sum(y3==1))} "
          f"yogurt={int(np.sum(y3==2))})")

    # ── Train all models ONCE ──────────────────────────────────────────────
    print("\n" + "-" * 60)
    print("Training models (once on full 41-participant baseline) ...")
    rf_bin,  sc_bin          = train_rf_binary(Xf_bin, Xs_bin, yb)
    cnn_bin, cnn_mn, cnn_sd  = train_cnn_binary(Xf_bin, Xs_bin, yb)
    rf_3c,   sc_3c           = train_rf_3class(Xf_3c, Xs_3c, y3)
    print("Training complete.")

    # ── Evaluate each condition using the same trained models ──────────────
    rows = []
    for cond_name, cond_folder in CONDITIONS.items():
        print(f"\n{'-'*60}")
        print(f"  Condition: {cond_name}  ({cond_folder.name})")

        result = load_condition(cond_folder, ref_chirp)
        if result is None:
            print(f"    SKIP -- no data found")
            continue

        Xf_te, Xs_te, yb_te, y3_te = result

        if len(np.unique(yb_te)) < 2:
            print(f"    SKIP -- only one class in binary test set")
            continue

        print(f"  Binary evaluation ...")
        bin_res = eval_binary_on_cond(rf_bin, sc_bin, cnn_bin, cnn_mn, cnn_sd,
                                      Xf_te, Xs_te, yb_te)

        print(f"  3-class evaluation ...")
        if len(np.unique(y3_te)) >= 2:
            mc_res = eval_3class_on_cond(rf_3c, sc_3c, n_feat_3c, Xf_te, y3_te)
        else:
            mc_res = {"3c_acc": 0.0, "3c_macro_f1": 0.0, "3c_wtd_f1": 0.0,
                      "3c_f1_idle": 0.0, "3c_f1_carrots": 0.0, "3c_f1_yogurt": 0.0}

        row = {"condition": cond_name}
        row.update(bin_res)
        row.update(mc_res)
        rows.append(row)

    if not rows:
        print("No conditions evaluated.")
        return

    df = pd.DataFrame(rows)
    out_csv = SCRIPT_DIR / "pipeline_robustness_results.csv"
    df.to_csv(out_csv, index=False)

    plot_summary(df)

    # ── Summary table ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  ROBUSTNESS SUMMARY")
    print(f"  Baseline (train=41p): RF F1=0.905 | CNN F1=0.946 | 3-cls MacroF1=0.264")
    print("=" * 70)
    cols = ["condition", "RF_f1", "CNN_f1", "3c_macro_f1", "RF_rec", "CNN_rec"]
    print(df[cols].to_string(index=False))

    best_cnn  = df.loc[df["CNN_f1"].idxmax()]
    worst_cnn = df.loc[df["CNN_f1"].idxmin()]
    best_3c   = df.loc[df["3c_macro_f1"].idxmax()]

    print(f"\n  Best  binary condition : {best_cnn['condition']} "
          f"(CNN F1={best_cnn['CNN_f1']:.4f})")
    print(f"  Worst binary condition : {worst_cnn['condition']} "
          f"(CNN F1={worst_cnn['CNN_f1']:.4f})")
    print(f"  Best  3-class condition: {best_3c['condition']} "
          f"(MacroF1={best_3c['3c_macro_f1']:.4f})")
    print(f"\n  Results saved: {out_csv.name}")
    print("  Robustness evaluation complete.")


if __name__ == "__main__":
    main()
