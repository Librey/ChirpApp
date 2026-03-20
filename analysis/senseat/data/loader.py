"""
senseat/data/loader.py
======================
PCM loading, preprocessing, segmentation.
Tracks participant_id for every segment — required for GroupKFold / LOPO.
Also loads _idleTail.pcm files as idle (label=0) to fix class imbalance.
"""

import re
import numpy as np
from pathlib import Path
from scipy.signal import butter, lfilter

# ─────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────
SAMPLE_RATE          = 44100
LOWCUT               = 17500.0
HIGHCUT              = 20500.0
FILTER_ORDER         = 6
CHIRP_PERIOD_S       = 1.5
SAMPLES_PER_CHUNK    = int(SAMPLE_RATE * CHIRP_PERIOD_S)   # 66150
HOP_SAMPLES          = int(SAMPLE_RATE * 0.5)              # 0.5s hop for overlap
DIRECT_PATH_SAMPLES  = int((0.30 / 343.0) * SAMPLE_RATE)  # ~38 samples

IRB_RE = re.compile(r"^(\d+)_(\d{3})_(\d{2})_(\d{2})$")

FOOD_NAMES = {
    1:  "Water",       2:  "Crackers",     3:  "Potato_Chips",
    4:  "Chocolate",   5:  "Baby_Carrots", 6:  "Apple_Slice",
    7:  "Almonds",     8:  "Cereal",       9:  "Gummy_Bears",
    10: "Pistachios",  11: "Peanuts"
}

# ─────────────────────────────────────────
# LOW-LEVEL HELPERS
# ─────────────────────────────────────────

def butter_bandpass(data, lowcut=LOWCUT, highcut=HIGHCUT, fs=SAMPLE_RATE, order=FILTER_ORDER):
    nyq  = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    if low <= 0 or high >= 1 or low >= high:
        return data
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)


def load_pcm_left(filepath):
    """Load stereo 16-bit PCM, return left channel as float32 in [-1, 1]."""
    raw = np.fromfile(filepath, dtype=np.int16)
    if raw.size % 2 != 0:
        raw = raw[:-1]
    stereo = raw.reshape(-1, 2)
    return stereo[:, 0].astype(np.float32) / 32768.0


def preprocess(signal):
    """Strip direct path + bandpass filter."""
    signal = signal[DIRECT_PATH_SAMPLES:]
    return butter_bandpass(signal)


def segment_signal(signal, overlap=True):
    """
    Chop signal into 1.5s chunks.
    overlap=True  → 0.5s hop (used for multiclass, more segments)
    overlap=False → non-overlapping (used for binary, avoids leakage)

    Normalization is by recording-level max (computed before chunking),
    so amplitude differences between eating and idle are preserved.
    """
    hop     = HOP_SAMPLES if overlap else SAMPLES_PER_CHUNK
    rec_max = np.max(np.abs(signal))
    if rec_max > 0:
        signal = signal / rec_max   # recording-level norm — preserves relative energy
    chunks = []
    for i in range(0, len(signal) - SAMPLES_PER_CHUNK + 1, hop):
        seg = signal[i : i + SAMPLES_PER_CHUNK].copy()
        if len(seg) == SAMPLES_PER_CHUNK:
            chunks.append(seg)
    return chunks


# ─────────────────────────────────────────
# DATASET BUILDERS
# ─────────────────────────────────────────

def build_binary_dataset(raw_data_path):
    """
    Build binary dataset: Eating (1) vs Idle (0).

    Idle sources (fixes class imbalance):
      - raw_data/Idle-gp/       → dedicated idle recordings
      - raw_data/Idle/          → additional idle recordings
      - raw_data/XXX/*_idleTail.pcm → idle tail from every participant

    Returns:
        segments     : list of np.ndarray (66150,)
        labels       : list of int  (0 or 1)
        participant_ids : list of str  (e.g. "001") — for GroupKFold
        meta         : list of dict
    """
    raw_data = Path(raw_data_path)
    segments, labels, participant_ids, meta = [], [], [], []

    eating_folders = [f"{i:03d}" for i in range(1, 21)]

    # ── Eating segments (label=1) ──
    for folder_name in eating_folders:
        folder_path = raw_data / folder_name
        if not folder_path.exists():
            continue

        eating_files = sorted([
            f for f in folder_path.glob("*.pcm")
            if not f.stem.endswith("_idleTail")
        ])

        for pcm in eating_files:
            m = IRB_RE.match(pcm.stem)
            if not m:
                continue
            food_code = int(m.group(3))
            signal    = preprocess(load_pcm_left(pcm))
            chunks    = segment_signal(signal, overlap=False)

            for ci, seg in enumerate(chunks):
                segments.append(seg)
                labels.append(1)
                participant_ids.append(folder_name)
                meta.append({
                    "participant": folder_name,
                    "file": pcm.name,
                    "chunk": ci,
                    "food_code": food_code,
                    "label": 1,
                    "source": "eating"
                })

    # ── Idle segments from _idleTail files (label=0) ──
    for folder_name in eating_folders:
        folder_path = raw_data / folder_name
        if not folder_path.exists():
            continue

        idle_tail_files = sorted(folder_path.glob("*_idleTail.pcm"))
        for pcm in idle_tail_files:
            signal = preprocess(load_pcm_left(pcm))
            chunks = segment_signal(signal, overlap=False)
            for ci, seg in enumerate(chunks):
                segments.append(seg)
                labels.append(0)
                participant_ids.append(folder_name)
                meta.append({
                    "participant": folder_name,
                    "file": pcm.name,
                    "chunk": ci,
                    "food_code": 0,
                    "label": 0,
                    "source": "idleTail"
                })

    # ── Idle segments from dedicated idle folders (label=0) ──
    for idle_folder in ["Idle-gp", "Idle"]:
        idle_path = raw_data / idle_folder
        if not idle_path.exists():
            continue
        for pcm in sorted(idle_path.glob("*.pcm")):
            if pcm.stem.endswith("_idleTail"):
                continue
            signal = preprocess(load_pcm_left(pcm))
            chunks = segment_signal(signal, overlap=False)
            for ci, seg in enumerate(chunks):
                segments.append(seg)
                labels.append(0)
                participant_ids.append("idle")
                meta.append({
                    "participant": "idle",
                    "file": pcm.name,
                    "chunk": ci,
                    "food_code": 0,
                    "label": 0,
                    "source": idle_folder
                })

    return (
        np.array(segments, dtype=np.float32),
        np.array(labels,   dtype=np.int32),
        np.array(participant_ids),
        meta
    )


def build_multiclass_dataset(raw_data_path):
    """
    Build multiclass dataset: food type classification.
    Uses non-overlapping segments to prevent data leakage.
    Only includes food codes present in FOOD_NAMES.

    Returns:
        segments        : np.ndarray (N, 66150)
        labels          : np.ndarray (N,)  — food codes
        participant_ids : np.ndarray (N,)  — for GroupKFold
        meta            : list of dict
        label_map       : dict {food_code: class_idx}
        idx_map         : dict {class_idx: food_code}
    """
    raw_data = Path(raw_data_path)
    segments, labels, participant_ids, meta = [], [], [], []

    eating_folders = [f"{i:03d}" for i in range(1, 21)]

    for folder_name in eating_folders:
        folder_path = raw_data / folder_name
        if not folder_path.exists():
            continue

        pcm_files = sorted([
            f for f in folder_path.glob("*.pcm")
            if not f.stem.endswith("_idleTail")
        ])

        for pcm in pcm_files:
            m = IRB_RE.match(pcm.stem)
            if not m:
                continue
            food_code = int(m.group(3))
            if food_code not in FOOD_NAMES:
                continue

            signal = preprocess(load_pcm_left(pcm))
            chunks = segment_signal(signal, overlap=False)

            for ci, seg in enumerate(chunks):
                segments.append(seg)
                labels.append(food_code)
                participant_ids.append(folder_name)
                meta.append({
                    "participant": folder_name,
                    "file": pcm.name,
                    "chunk": ci,
                    "food_code": food_code,
                    "food_name": FOOD_NAMES[food_code]
                })

    labels_arr = np.array(labels, dtype=np.int32)
    unique_codes = sorted(set(labels))
    label_map = {code: idx for idx, code in enumerate(unique_codes)}
    idx_map   = {idx: code for idx, code in enumerate(unique_codes)}

    return (
        np.array(segments, dtype=np.float32),
        labels_arr,
        np.array(participant_ids),
        meta,
        label_map,
        idx_map
    )
