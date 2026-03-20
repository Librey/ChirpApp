"""
SensEat 1.5s Chirp-Level Classification Pipeline
=================================================
Fixed 1.5-second segmentation aligned to the hardware chirp cycle.

Phase 1 (this script): Binary classification — Eating (1) vs Idle (0)
Evaluates ALL feature × model combinations and outputs ranked results.

Usage:  python audio_pipeline_1_5s.py
Output: pipeline_1_5s_results.csv
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

from scipy.signal import butter, lfilter
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

# Suppress TF info logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
SAMPLE_RATE    = 44100
LOWCUT         = 17500.0
HIGHCUT        = 20500.0
FILTER_ORDER   = 6
CHIRP_PERIOD_S = 1.5                          # Fixed chirp cycle
SAMPLES_PER_CHUNK = int(SAMPLE_RATE * CHIRP_PERIOD_S)  # 66 150
DIRECT_PATH_SAMPLES = int((0.30 / 343.0) * SAMPLE_RATE)  # ~38

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
RAW_DATA   = SCRIPT_DIR / "raw_data"

# Participant folders that contain EATING recordings (All 20)
EATING_FOLDERS = [f"{i:03d}" for i in range(1, 21)]
# Folders containing IDLE recordings (Pixel data only)
IDLE_FOLDERS   = ["idle-gp"]

# IRB filename pattern: W_XXX_YY_ZZ
IRB_RE = re.compile(r"^(\d+)_(\d{3})_(\d{2})_(\d{2})$")

# ─────────────────────────────────────────
# AUDIO LOADING & PREPROCESSING
# ─────────────────────────────────────────

def butter_bandpass_filter(data, lowcut, highcut, fs, order=6):
    """Apply Butterworth bandpass filter."""
    nyquist = 0.5 * fs
    low  = lowcut / nyquist
    high = highcut / nyquist
    if low <= 0 or high >= 1 or low >= high:
        return data
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)


def load_pcm_left_channel(filepath):
    """Load stereo PCM file, return left channel as float32 in [-1, 1]."""
    raw = np.fromfile(filepath, dtype=np.int16)
    if raw.size % 2 != 0:
        raw = raw[:-1]
    stereo = raw.reshape(-1, 2)
    left = stereo[:, 0].astype(np.float32) / 32768.0
    return left


def preprocess_and_segment(filepath):
    """Load PCM → left channel → strip direct path → bandpass → 1.5s chunks."""
    signal = load_pcm_left_channel(filepath)

    # Strip direct-path interference
    signal = signal[DIRECT_PATH_SAMPLES:]

    # Apply bandpass filter to isolate chirp band
    signal = butter_bandpass_filter(signal, LOWCUT, HIGHCUT, SAMPLE_RATE, FILTER_ORDER)

    # Chop into fixed 1.5s segments
    chunks = []
    for i in range(0, len(signal), SAMPLES_PER_CHUNK):
        seg = signal[i : i + SAMPLES_PER_CHUNK]
        if len(seg) == SAMPLES_PER_CHUNK:
            chunks.append(seg)
    return chunks


# ─────────────────────────────────────────
# FEATURE EXTRACTION
# (Same methods as audio-filter.py, self-contained)
# ─────────────────────────────────────────

def get_statistical_features(segment):
    """6-dim flat vector: mean, std, max, min, power, ZCR."""
    zcr = librosa.feature.zero_crossing_rate(segment)[0]
    return np.array([
        np.mean(segment),
        np.std(segment),
        np.max(segment),
        np.min(segment),
        np.mean(segment ** 2),
        np.mean(zcr)
    ], dtype=np.float32)


def get_wavelet_features(segment, wavelet='db4', level=4):
    """Flat vector of DWT coefficient statistics."""
    try:
        coeffs = pywt.wavedec(segment, wavelet, level=level)
    except Exception:
        coeffs = pywt.wavedec(segment, wavelet, level=1)
    feats = []
    for c in coeffs[1:]:
        feats.extend([np.mean(c), np.std(c), np.max(c), np.min(c), np.mean(c**2)])
    return np.array(feats, dtype=np.float32)


def get_mel_spectrogram(segment, sr=SAMPLE_RATE, n_mels=64, n_fft=2048, fixed_frames=64):
    """64×64×1 mel-spectrogram image."""
    hop = max(64, int(np.floor(len(segment) / (fixed_frames - 1))))
    S = librosa.feature.melspectrogram(y=segment, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels)
    S_db = librosa.power_to_db(S, ref=np.max)
    if S_db.shape[1] < fixed_frames:
        S_db = np.pad(S_db, ((0, 0), (0, fixed_frames - S_db.shape[1])),
                      mode='constant', constant_values=(S_db.min(),))
    elif S_db.shape[1] > fixed_frames:
        S_db = S_db[:, :fixed_frames]
    return S_db.astype(np.float32)[..., np.newaxis]


def get_mfcc_fixed(segment, sr=SAMPLE_RATE, n_mfcc=40, fixed_frames=64):
    """40×64×1 MFCC image."""
    hop = max(64, int(np.floor(len(segment) / (fixed_frames - 1))))
    mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=n_mfcc, hop_length=hop)
    if mfcc.shape[1] < fixed_frames:
        mfcc = np.pad(mfcc, ((0, 0), (0, fixed_frames - mfcc.shape[1])),
                      mode='constant', constant_values=(mfcc.min(),))
    elif mfcc.shape[1] > fixed_frames:
        mfcc = mfcc[:, :fixed_frames]
    return mfcc.astype(np.float32)[..., np.newaxis]


def get_wavelet_scalogram(segment, wavelet='morl', scales=None, fixed_size=(64, 64)):
    """64×64×1 CWT scalogram image."""
    if scales is None:
        scales = np.arange(1, 65)
    try:
        coefficients, _ = pywt.cwt(segment, scales, wavelet)
        scalo = np.abs(coefficients)
        scalo = (scalo - scalo.min()) / (scalo.max() - scalo.min() + 1e-8)
        resized = cv2.resize(scalo, fixed_size, interpolation=cv2.INTER_CUBIC)
        return resized.astype(np.float32)[..., np.newaxis]
    except Exception:
        return np.zeros((*fixed_size, 1), dtype=np.float32)


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


# ─────────────────────────────────────────
# DATASET BUILDER
# ─────────────────────────────────────────

def build_dataset():
    """Walk raw_data/ and build the full feature dataset for binary classification."""

    feat_stat  = []
    feat_wave  = []
    feat_mel   = []
    feat_mfcc  = []
    feat_scalo = []
    feat_stft  = []
    labels     = []
    meta_info  = []   # (folder, filename, chunk_idx, food_code)

    # --- Eating files (label = 1) ---
    for folder_name in EATING_FOLDERS:
        folder_path = RAW_DATA / folder_name
        if not folder_path.exists():
            print(f"  ⚠ Eating folder not found: {folder_path}")
            continue

        pcm_files = sorted([
            f for f in folder_path.glob("*.pcm")
            if not f.stem.endswith("_idleTail") and not f.stem.endswith("_meta")
        ])
        print(f"  📂 {folder_name}: {len(pcm_files)} eating files")

        for pcm in pcm_files:
            chunks = preprocess_and_segment(str(pcm))
            # Parse food code from filename
            m = IRB_RE.match(pcm.stem)
            food_code = int(m.group(3)) if m else -1

            for ci, seg in enumerate(chunks):
                feat_stat.append(get_statistical_features(seg))
                feat_wave.append(get_wavelet_features(seg))
                feat_mel.append(get_mel_spectrogram(seg))
                feat_mfcc.append(get_mfcc_fixed(seg))
                feat_scalo.append(get_wavelet_scalogram(seg))
                feat_stft.append(get_stft_image(seg))
                labels.append(1)
                meta_info.append((folder_name, pcm.name, ci, food_code))

    # --- Idle files (label = 0) ---
    for idle_folder_name in IDLE_FOLDERS:
        idle_path = RAW_DATA / idle_folder_name
        if not idle_path.exists():
            print(f"  ⚠ Idle folder not found: {idle_path}")
            continue

        idle_files = sorted([
            f for f in idle_path.glob("*.pcm")
            if not f.stem.endswith("_idleTail") and not f.stem.endswith("_meta")
        ])
        print(f"  📂 {idle_folder_name}: {len(idle_files)} idle files")

        for pcm in idle_files:
            chunks = preprocess_and_segment(str(pcm))
            for ci, seg in enumerate(chunks):
                feat_stat.append(get_statistical_features(seg))
                feat_wave.append(get_wavelet_features(seg))
                feat_mel.append(get_mel_spectrogram(seg))
                feat_mfcc.append(get_mfcc_fixed(seg))
                feat_scalo.append(get_wavelet_scalogram(seg))
                feat_stft.append(get_stft_image(seg))
                labels.append(0)
                meta_info.append((idle_folder_name, pcm.name, ci, 0))

    return (
        np.array(feat_stat),
        np.array(feat_wave),
        np.array(feat_mel),
        np.array(feat_mfcc),
        np.array(feat_scalo),
        np.array(feat_stft),
        np.array(labels),
        meta_info
    )


# ─────────────────────────────────────────
# MODEL TRAINING
# ─────────────────────────────────────────

def train_classic_ml(X, y, model_name="SVM", n_splits=5):
    """Train SVM / RF / kNN with cross-validation + 80/20 split."""
    if model_name == "SVM":
        clf = SVC(kernel='rbf', probability=True, random_state=SEED)
    elif model_name == "RF":
        clf = RandomForestClassifier(n_estimators=100, random_state=SEED)
    elif model_name == "kNN":
        clf = KNeighborsClassifier(n_neighbors=5)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    pipe = Pipeline([('scaler', StandardScaler()), ('clf', clf)])

    # Cross-validation
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    scoring = ['accuracy', 'precision', 'recall', 'f1']
    scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring, return_train_score=False)
    cv_results = {m: (scores[f'test_{m}'].mean(), scores[f'test_{m}'].std()) for m in scoring}

    # 80/20 holdout
    split_results = _holdout_split_classic(pipe, X, y)

    return cv_results, split_results


def _holdout_split_classic(pipe, X, y):
    min_class = int(np.min(np.bincount(y)))
    if len(y) >= 10 and min_class >= 2:
        try:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.2, random_state=SEED, stratify=y
            )
            pipe.fit(X_tr, y_tr)
            y_pred = pipe.predict(X_te)
            return {
                'accuracy':  accuracy_score(y_te, y_pred),
                'precision': precision_score(y_te, y_pred, zero_division=0),
                'recall':    recall_score(y_te, y_pred, zero_division=0),
                'f1':        f1_score(y_te, y_pred, zero_division=0),
            }
        except Exception:
            pass
    return None


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


def train_cnn_cv(X, y, input_shape, n_splits=5, epochs=20, batch_size=16):
    """Train CNN with cross-validation + 80/20 holdout."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    cv_metrics = defaultdict(list)

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        mean = X_train.mean()
        std  = X_train.std() if X_train.std() > 0 else 1.0
        X_train = (X_train - mean) / std
        X_test  = (X_test  - mean) / std

        model = build_simple_cnn(input_shape)
        es = EarlyStopping(patience=3, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                  validation_split=0.1, callbacks=[es], verbose=0)

        y_pred = (model.predict(X_test, verbose=0).ravel() > 0.5).astype(int)

        cv_metrics['accuracy'].append(accuracy_score(y_test, y_pred))
        cv_metrics['precision'].append(precision_score(y_test, y_pred, zero_division=0))
        cv_metrics['recall'].append(recall_score(y_test, y_pred, zero_division=0))
        cv_metrics['f1'].append(f1_score(y_test, y_pred, zero_division=0))

        tf.keras.backend.clear_session()

    cv_results = {m: (np.mean(v), np.std(v)) for m, v in cv_metrics.items()}

    # 80/20 holdout
    split_results = _holdout_split_cnn(X, y, input_shape, epochs, batch_size)

    tf.keras.backend.clear_session()
    return cv_results, split_results


def _holdout_split_cnn(X, y, input_shape, epochs=20, batch_size=16):
    min_class = int(np.min(np.bincount(y)))
    if len(y) >= 10 and min_class >= 2:
        try:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.2, random_state=SEED, stratify=y
            )
            mean = X_tr.mean()
            std  = X_tr.std() if X_tr.std() > 0 else 1.0
            X_tr = (X_tr - mean) / std
            X_te = (X_te - mean) / std

            model = build_simple_cnn(input_shape)
            es = EarlyStopping(patience=3, restore_best_weights=True)
            model.fit(X_tr, y_tr, epochs=epochs, batch_size=batch_size,
                      validation_split=0.1, callbacks=[es], verbose=0)

            y_pred = (model.predict(X_te, verbose=0).ravel() > 0.5).astype(int)
            tf.keras.backend.clear_session()
            return {
                'accuracy':  accuracy_score(y_te, y_pred),
                'precision': precision_score(y_te, y_pred, zero_division=0),
                'recall':    recall_score(y_te, y_pred, zero_division=0),
                'f1':        f1_score(y_te, y_pred, zero_division=0),
            }
        except Exception:
            pass
    return None


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

def main():
    print("=" * 70)
    print("  SensEat — 1.5s Chirp-Level Binary Classification Pipeline")
    print("  Eating (1) vs Idle (0)")
    print("=" * 70)

    # ── Build dataset ──
    print("\n📦 Loading and segmenting PCM files (T = 1.5s) ...\n")
    X_stat, X_wave, X_mel, X_mfcc, X_scalo, X_stft, y, meta = build_dataset()

    n_total  = len(y)
    n_eat    = int(np.sum(y == 1))
    n_idle   = int(np.sum(y == 0))
    print(f"\n✅ Dataset built: {n_total} total segments")
    print(f"   Eating: {n_eat}  |  Idle: {n_idle}")

    if n_total == 0 or len(np.unique(y)) < 2:
        print("❌ Need both Eating and Idle segments. Aborting.")
        return

    # Determine effective CV folds
    min_count = int(np.min(np.bincount(y)))
    n_splits = min(5, min_count)
    if n_splits < 2:
        print(f"❌ Not enough samples per class for CV (min={min_count}). Aborting.")
        return
    print(f"   CV folds: {n_splits}")

    # ── Run all combinations ──
    results_rows = []

    # --- Classic ML on flat features ---
    flat_features = [
        ("statistical", X_stat),
        ("wavelet_DWT", X_wave),
    ]
    classic_models = ["SVM", "RF", "kNN"]

    for feat_name, X in flat_features:
        if X.size == 0:
            continue
        X2 = X.reshape((X.shape[0], -1)) if X.ndim > 2 else X
        for model_name in classic_models:
            print(f"\n🔧 {model_name} + {feat_name}  (shape {X2.shape})")
            cv_res, split_res = train_classic_ml(X2, y, model_name, n_splits=n_splits)

            # CV row
            results_rows.append({
                'model': model_name, 'feature': feat_name, 'eval': 'CV',
                'accuracy':  round(cv_res['accuracy'][0],  4),
                'acc_std':   round(cv_res['accuracy'][1],  4),
                'precision': round(cv_res['precision'][0], 4),
                'recall':    round(cv_res['recall'][0],    4),
                'f1':        round(cv_res['f1'][0],        4),
            })
            print(f"   CV  → Acc={cv_res['accuracy'][0]:.4f}±{cv_res['accuracy'][1]:.4f}  "
                  f"F1={cv_res['f1'][0]:.4f}")

            # 80/20 row
            if split_res:
                results_rows.append({
                    'model': model_name, 'feature': feat_name, 'eval': '80/20',
                    'accuracy':  round(split_res['accuracy'],  4),
                    'acc_std':   0.0,
                    'precision': round(split_res['precision'], 4),
                    'recall':    round(split_res['recall'],    4),
                    'f1':        round(split_res['f1'],        4),
                })
                print(f"   80/20 → Acc={split_res['accuracy']:.4f}  F1={split_res['f1']:.4f}")

    # --- CNN on 2D features ---
    cnn_features = [
        ("mel_spectrogram", X_mel),
        ("MFCC",            X_mfcc),
        ("wavelet_CWT",     X_scalo),
        ("STFT",            X_stft),
    ]

    for feat_name, X in cnn_features:
        if X.size == 0:
            continue
        X_float = X.astype(np.float32)
        print(f"\n🧠 CNN + {feat_name}  (shape {X_float[0].shape})")
        cv_res, split_res = train_cnn_cv(X_float, y, input_shape=X_float[0].shape,
                                         n_splits=n_splits)

        results_rows.append({
            'model': 'CNN', 'feature': feat_name, 'eval': 'CV',
            'accuracy':  round(cv_res['accuracy'][0],  4),
            'acc_std':   round(cv_res['accuracy'][1],  4),
            'precision': round(cv_res['precision'][0], 4),
            'recall':    round(cv_res['recall'][0],    4),
            'f1':        round(cv_res['f1'][0],        4),
        })
        print(f"   CV  → Acc={cv_res['accuracy'][0]:.4f}±{cv_res['accuracy'][1]:.4f}  "
              f"F1={cv_res['f1'][0]:.4f}")

        if split_res:
            results_rows.append({
                'model': 'CNN', 'feature': feat_name, 'eval': '80/20',
                'accuracy':  round(split_res['accuracy'],  4),
                'acc_std':   0.0,
                'precision': round(split_res['precision'], 4),
                'recall':    round(split_res['recall'],    4),
                'f1':        round(split_res['f1'],        4),
            })
            print(f"   80/20 → Acc={split_res['accuracy']:.4f}  F1={split_res['f1']:.4f}")

    # ── Save results ──
    if results_rows:
        df = pd.DataFrame(results_rows)
        out_csv = SCRIPT_DIR / "pipeline_1_5s_results.csv"
        df.to_csv(out_csv, index=False)
        print(f"\n💾 Results saved to {out_csv}")

        # Print ranked summary
        print("\n" + "=" * 70)
        print("  RANKED RESULTS (sorted by F1, CV only)")
        print("=" * 70)
        df_cv = df[df['eval'] == 'CV'].sort_values('f1', ascending=False)
        print(df_cv[['model', 'feature', 'accuracy', 'precision', 'recall', 'f1']].to_string(index=False))

        print("\n" + "=" * 70)
        print("  RANKED RESULTS (sorted by F1, 80/20 holdout)")
        print("=" * 70)
        df_split = df[df['eval'] == '80/20'].sort_values('f1', ascending=False)
        if len(df_split) > 0:
            print(df_split[['model', 'feature', 'accuracy', 'precision', 'recall', 'f1']].to_string(index=False))
        else:
            print("  (No holdout results)")

    print("\n✅ Pipeline complete.")


if __name__ == '__main__':
    main()
