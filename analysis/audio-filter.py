import os
import random
import json
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

from pathlib import Path

BASE = Path(__file__).resolve().parents[1]   # points to analysis/
POSITIVE_DIR = str(BASE / "analysis/raw_data")
NEGATIVE_DIR = str(BASE / "analysis/negative_data")

LOAD_POSITIVE = True
LOAD_NEGATIVE = True


SAMPLE_RATE = 44100
LOWCUT = 17500.0
HIGHCUT = 20500.0
FILTER_ORDER = 6
T_VALUES = [1, 2, 4]


SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    if low <= 0 or high >= 1 or low >= high:
        
        print(f"[WARN] Invalid bandpass normalized values (low={low:.3f}, high={high:.3f}). Skipping filter.")
        return data
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)
    return y


def load_and_clean_audio(file_path):
    try:
        if file_path.lower().endswith(".pcm"):
            signal = np.fromfile(file_path, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            signal, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)

        # Apply bandpass filter
        cleaned = butter_bandpass_filter(signal, LOWCUT, HIGHCUT, SAMPLE_RATE, FILTER_ORDER)
        return cleaned

    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None



def segment_signal(signal, period_seconds):
    samples_per_segment = int(SAMPLE_RATE * period_seconds)
    segments = []
    if len(signal) < samples_per_segment:
        return np.array([])
    for i in range(0, len(signal), samples_per_segment):
        segment = signal[i:i + samples_per_segment]
        if len(segment) == samples_per_segment:
            segments.append(segment)
    return np.array(segments)



def get_wavelet_features(segment, wavelet='db4', level=4):
    # truncate to reasonable length if extremely short
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
            np.mean(c**2)
        ])
    return np.array(feats)


def get_wavelet_scalogram(segment, wavelet='morl', scales=None, fixed_size=(64, 64)):
    """Generate a 2D scalogram (continuous wavelet transform) for CNN input."""
    if scales is None:
        scales = np.arange(1, 65)
    
    try:
        # Use Morlet (complex) wavelet for CWT instead of db4
        coefficients, frequencies = pywt.cwt(segment, scales, wavelet)
        scalogram = np.abs(coefficients)
        scalogram = (scalogram - scalogram.min()) / (scalogram.max() - scalogram.min() + 1e-8)
        scalogram_resized = cv2.resize(scalogram, fixed_size, interpolation=cv2.INTER_CUBIC)
        return scalogram_resized.astype(np.float32)[..., np.newaxis]
    except Exception as e:
        print(f"Error in wavelet scalogram: {e}")
        return np.zeros((*fixed_size, 1), dtype=np.float32)


def get_statistical_features(segment):
    zcr = librosa.feature.zero_crossing_rate(segment)[0]
    return np.array([
        np.mean(segment),
        np.std(segment),
        np.max(segment),
        np.min(segment),
        np.mean(segment**2),
        np.mean(zcr)
    ], dtype=np.float32)


def get_mel_spectrogram(segment, sr=SAMPLE_RATE, n_mels=64, n_fft=2048, fixed_frames=64):

    hop_length = max(64, int(np.floor(len(segment) / (fixed_frames - 1))))
    S = librosa.feature.melspectrogram(y=segment, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    S_db = librosa.power_to_db(S, ref=np.max)
    
    if S_db.shape[1] < fixed_frames:
        pad_width = fixed_frames - S_db.shape[1]
        S_db = np.pad(S_db, ((0, 0), (0, pad_width)), mode='constant', constant_values=(S_db.min(),))
    elif S_db.shape[1] > fixed_frames:
        S_db = S_db[:, :fixed_frames]
    return S_db.astype(np.float32)[..., np.newaxis]


def get_mfcc_fixed(segment, sr=SAMPLE_RATE, n_mfcc=40, fixed_frames=64):
    hop_length = max(64, int(np.floor(len(segment) / (fixed_frames - 1))))
    mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    if mfcc.shape[1] < fixed_frames:
        pad_width = fixed_frames - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0,0),(0,pad_width)), mode='constant', constant_values=(mfcc.min(),))
    elif mfcc.shape[1] > fixed_frames:
        mfcc = mfcc[:, :fixed_frames]
    return mfcc.astype(np.float32)[..., np.newaxis]


def get_stft_image(segment, n_fft=2048, fixed_size=(64,64)):
    """Generate STFT (Short-Time Fourier Transform) spectrogram image."""
    try:
        D = np.abs(librosa.stft(segment, n_fft=n_fft))
        S_db = librosa.amplitude_to_db(D, ref=np.max)
        S_db = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)
        resized = cv2.resize(S_db, fixed_size, interpolation=cv2.INTER_CUBIC)
        return resized.astype(np.float32)[..., np.newaxis]
    except Exception as e:
        print(f"Error in STFT: {e}")
        return np.zeros((*fixed_size, 1), dtype=np.float32)


def load_dataset_dynamic(period):
    features_stat = []
    features_wave = []
    features_wavelet_scalo = []
    features_spec = []
    features_mfcc = []
    features_stft = []
    labels = []

    directories = []
    if LOAD_POSITIVE:
        directories.append((POSITIVE_DIR, 1))
    if LOAD_NEGATIVE:
        directories.append((NEGATIVE_DIR, 0))

    if not directories:
        print("Error: Both LOAD_POSITIVE and LOAD_NEGATIVE are False. Nothing to do.")
        return (np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]))

    for dir_path, label_val in directories:
        if not os.path.exists(dir_path):
            print(f"Warning: Folder not found: {dir_path}")
            continue
        files = [f for f in os.listdir(dir_path) if f.lower().endswith(('.wav', '.pcm'))]
        print(f"  Processing {len(files)} files from {dir_path}...")

        for file in files:
            f_path = os.path.join(dir_path, file)
            signal = load_and_clean_audio(f_path)
            if signal is None:
                continue
            segments = segment_signal(signal, period)
            for seg in segments:
                features_stat.append(get_statistical_features(seg))
                features_wave.append(get_wavelet_features(seg))
                features_wavelet_scalo.append(get_wavelet_scalogram(seg))
                features_spec.append(get_mel_spectrogram(seg))
                features_mfcc.append(get_mfcc_fixed(seg))
                features_stft.append(get_stft_image(seg))
                labels.append(label_val)

    if len(labels) == 0:
        return (np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]))

    return (np.array(features_stat), np.array(features_wave), np.array(features_wavelet_scalo), 
            np.array(features_spec), np.array(features_mfcc), np.array(features_stft), np.array(labels))



def train_classic_ml(X, y, model_name="SVM", n_splits=5):
    """Train classic ML model with cross-validation and explicit train/test split."""
    if model_name == "SVM":
        clf = SVC(kernel='rbf', probability=True, random_state=SEED)
    elif model_name == "RF":
        clf = RandomForestClassifier(n_estimators=100, random_state=SEED)
    elif model_name == "kNN":
        clf = KNeighborsClassifier(n_neighbors=5)
    else:
        raise ValueError("Unknown model")

    pipe = Pipeline([('scaler', StandardScaler()), ('clf', clf)])
    
    # Cross-validation results
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    scoring = ['accuracy', 'precision', 'recall', 'f1']
    scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring, return_train_score=False)
    cv_results = {metric: (scores[f'test_{metric}'].mean(), scores[f'test_{metric}'].std()) for metric in scoring}
    
    # Explicit 80/20 train/test split (only if we have enough samples)
    min_class_count = np.min(np.bincount(y))
    if len(y) >= 10 and min_class_count >= 2:  # Need at least 2 per class for stratified split
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            y_pred_proba = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe.named_steps['clf'], 'predict_proba') else pipe.predict(X_test)
            
            split_results = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0)
            }
        except Exception as e:
            # Fallback if split fails
            print(f"    Warning: Could not perform 80/20 split ({e}). Using CV results only.")
            split_results = {k: v[0] for k, v in cv_results.items()}  # Use CV mean as fallback
    else:
        # Not enough samples, use CV results
        split_results = {k: v[0] for k, v in cv_results.items()}
    
    return cv_results, split_results



def build_simple_cnn(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(16, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def train_cnn_cv(X, y, input_shape, n_splits=5, epochs=20, batch_size=16):
    """Train CNN with cross-validation and explicit train/test split."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    cv_metrics = defaultdict(list)

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        mean = X_train.mean()
        std = X_train.std() if X_train.std() > 0 else 1.0
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std

        model = build_simple_cnn(input_shape)
        es = EarlyStopping(patience=3, restore_best_weights=True)
        
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, callbacks=[es], verbose=0)
        y_pred_prob = model.predict(X_test, verbose=0).ravel()
        y_pred = (y_pred_prob > 0.5).astype(int)

        cv_metrics['accuracy'].append(accuracy_score(y_test, y_pred))
        cv_metrics['precision'].append(precision_score(y_test, y_pred, zero_division=0))
        cv_metrics['recall'].append(recall_score(y_test, y_pred, zero_division=0))
        cv_metrics['f1'].append(f1_score(y_test, y_pred, zero_division=0))

        tf.keras.backend.clear_session()

    cv_results = {m: (np.mean(v), np.std(v)) for m, v in cv_metrics.items()}
    
    # Explicit 80/20 train/test split (only if we have enough samples)
    min_class_count = np.min(np.bincount(y))
    if len(y) >= 10 and min_class_count >= 2:
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
            
            mean = X_train.mean()
            std = X_train.std() if X_train.std() > 0 else 1.0
            X_train = (X_train - mean) / std
            X_test = (X_test - mean) / std
            
            model = build_simple_cnn(input_shape)
            es = EarlyStopping(patience=3, restore_best_weights=True)
            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, callbacks=[es], verbose=0)
            y_pred_prob = model.predict(X_test, verbose=0).ravel()
            y_pred = (y_pred_prob > 0.5).astype(int)
            
            split_results = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0)
            }
        except Exception as e:
            print(f"    Warning: Could not perform 80/20 split ({e}). Using CV results only.")
            split_results = {k: v[0] for k, v in cv_results.items()}
    else:
        # Not enough samples, use CV results
        split_results = {k: v[0] for k, v in cv_results.items()}
    
    tf.keras.backend.clear_session()
    
    return cv_results, split_results



def main():
    print(f"Starting pipeline (POS={LOAD_POSITIVE}, NEG={LOAD_NEGATIVE})")
    results_rows = []

    for T in T_VALUES:
        print(f"\n=== Period T = {T}s ===")
        X_stat, X_wave, X_wavelet_scalo, X_spec, X_mfcc, X_stft, y = load_dataset_dynamic(T)
        if len(y) == 0:
            print("No samples found for this T. Skipping.")
            continue
        print(f"Total samples: {len(y)} | Pos:{np.sum(y==1)} Neg:{np.sum(y==0)}")

        # Check classes
        if len(np.unique(y)) < 2:
            print("Need at least two classes (positive & negative). Skipping this T.")
            continue

        # Determine effective number of CV splits based on smallest class count
        try:
            min_count = int(np.min(np.bincount(y)))
        except Exception:
            min_count = 0
        n_splits_eff = min(5, min_count)
        if n_splits_eff < 2:
            print(f"Not enough samples per class for cross-validation (min per class={min_count}). Skipping this T.")
            continue

        # CLASSIC ML on stat + wavelet
        for modality_name, X in [('stat', X_stat), ('wavelet', X_wave)]:
            if X.size == 0:
                continue
            
            X2 = X.reshape((X.shape[0], -1)) if X.ndim > 2 else X
            print(f"Training classic models on {modality_name} features: shape={X2.shape}")
            for model_name in ['SVM','RF','kNN']:
                cv_res, split_res = train_classic_ml(X2, y, model_name, n_splits=n_splits_eff)
                print(f"  {model_name} ({modality_name}): CV acc={cv_res['accuracy'][0]:.3f}±{cv_res['accuracy'][1]:.3f}, "
                      f"Split acc={split_res['accuracy']:.3f}")
                
                # Record CV results
                results_rows.append({
                    'T': T, 'model': model_name, 'modality': modality_name, 'eval_type': 'cross_validation',
                    'accuracy_mean': float(cv_res['accuracy'][0]), 'accuracy_std': float(cv_res['accuracy'][1]),
                    'precision_mean': float(cv_res['precision'][0]), 'recall_mean': float(cv_res['recall'][0]), 'f1_mean': float(cv_res['f1'][0])
                })
                
                # Record 80/20 split results
                results_rows.append({
                    'T': T, 'model': model_name, 'modality': modality_name, 'eval_type': '80_20_split',
                    'accuracy_mean': float(split_res['accuracy']), 'accuracy_std': 0.0,
                    'precision_mean': float(split_res['precision']), 'recall_mean': float(split_res['recall']), 'f1_mean': float(split_res['f1'])
                })

        # CNN for image-based modalities
        for modality_name, X in [('mel_spec', X_spec), ('mfcc', X_mfcc), ('wavelet_scalo', X_wavelet_scalo), ('stft', X_stft)]:
            if X.size == 0:
                continue
            print(f"Training CNN on {modality_name} with input shape {X[0].shape}")
            X_float = X.astype(np.float32)
            cv_res, split_res = train_cnn_cv(X_float, y, input_shape=X_float[0].shape, n_splits=n_splits_eff)
            print(f"  CNN ({modality_name}): CV acc={cv_res['accuracy'][0]:.3f}±{cv_res['accuracy'][1]:.3f}, "
                  f"Split acc={split_res['accuracy']:.3f}")
            
            # Record CV results
            results_rows.append({
                'T': T, 'model': 'CNN', 'modality': modality_name, 'eval_type': 'cross_validation',
                'accuracy_mean': float(cv_res['accuracy'][0]), 'accuracy_std': float(cv_res['accuracy'][1]),
                'precision_mean': float(cv_res['precision'][0]), 'recall_mean': float(cv_res['recall'][0]), 'f1_mean': float(cv_res['f1'][0])
            })
            
            # Record 80/20 split results
            results_rows.append({
                'T': T, 'model': 'CNN', 'modality': modality_name, 'eval_type': '80_20_split',
                'accuracy_mean': float(split_res['accuracy']), 'accuracy_std': 0.0,
                'precision_mean': float(split_res['precision']), 'recall_mean': float(split_res['recall']), 'f1_mean': float(split_res['f1'])
            })

    if len(results_rows) > 0:
        df = pd.DataFrame(results_rows)
        out_csv = 'model_results.csv'
        df.to_csv(out_csv, index=False)
        print(f"\nSaved results to {out_csv}")
        print("\nResults summary:")
        print(df.to_string())
    else:
        print("No results to save.")

if __name__ == '__main__':
    main()
