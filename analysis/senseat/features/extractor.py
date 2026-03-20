"""
senseat/features/extractor.py
==============================
Feature extraction:
  - STFT spectrogram  (64x64)
  - MFCC              (40x64)
  - SAR Range-Doppler map (64x64)  ← new, MobiSys 2018 inspired
  - SpecAugment augmentation
"""

import numpy as np
import librosa
import cv2
from scipy.signal import butter, lfilter, spectrogram as scipy_spectrogram
from spafe.features.gfcc import gfcc as compute_gfcc

# ─────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────
SAMPLE_RATE    = 44100
CHIRP_START_HZ = 18000.0
CHIRP_END_HZ   = 20000.0
CHIRP_DUR_S    = 1.0
GAP_DUR_S      = 0.5
CHIRP_PERIOD_S = 1.5
SAMPLES_PER_CHUNK   = int(SAMPLE_RATE * CHIRP_PERIOD_S)
CHIRP_ONLY_SAMPLES  = int(SAMPLE_RATE * CHIRP_DUR_S)
IMG_SIZE       = (64, 64)

# ─────────────────────────────────────────
# STFT SPECTROGRAM
# ─────────────────────────────────────────

def get_stft_image(segment, n_fft=2048, fixed_size=IMG_SIZE):
    """64×64×1 normalized STFT spectrogram."""
    try:
        D     = np.abs(librosa.stft(segment, n_fft=n_fft))
        S_db  = librosa.amplitude_to_db(D, ref=np.max)
        S_db  = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)
        img   = cv2.resize(S_db, fixed_size, interpolation=cv2.INTER_CUBIC)
        return img.astype(np.float32)[..., np.newaxis]
    except Exception:
        return np.zeros((*fixed_size, 1), dtype=np.float32)


# ─────────────────────────────────────────
# MFCC
# ─────────────────────────────────────────

def get_mfcc_image(segment, sr=SAMPLE_RATE, n_mfcc=40, fixed_frames=64):
    """n_mfcc×64×1 normalized MFCC image."""
    try:
        hop  = max(64, int(np.floor(len(segment) / (fixed_frames - 1))))
        mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=n_mfcc, hop_length=hop)
        if mfcc.shape[1] < fixed_frames:
            mfcc = np.pad(mfcc, ((0, 0), (0, fixed_frames - mfcc.shape[1])),
                          mode='constant', constant_values=mfcc.min())
        else:
            mfcc = mfcc[:, :fixed_frames]
        mfcc = (mfcc - mfcc.min()) / (mfcc.max() - mfcc.min() + 1e-8)
        return mfcc.astype(np.float32)[..., np.newaxis]
    except Exception:
        return np.zeros((n_mfcc, fixed_frames, 1), dtype=np.float32)


# ─────────────────────────────────────────
# GFCC
# ─────────────────────────────────────────

def get_gfcc_image(segment, sr=SAMPLE_RATE, n_gfcc=40, fixed_frames=64):
    """
    n_gfcc×64×1 normalized GFCC image.
    Gammatone filter bank — models cochlear frequency response.
    Better than MFCC for distinguishing food textures (crunchy vs soft).
    """
    try:
        # spafe expects int16-range floats or normalized; works with float32 [-1,1]
        feats = compute_gfcc(segment, fs=sr, num_ceps=n_gfcc,
                             pre_emph=True, normalize="mvn")
        # feats shape: (frames, n_gfcc) → transpose to (n_gfcc, frames)
        feats = feats.T
        if feats.shape[1] < fixed_frames:
            feats = np.pad(feats, ((0, 0), (0, fixed_frames - feats.shape[1])),
                           mode='constant', constant_values=feats.min())
        else:
            feats = feats[:, :fixed_frames]
        feats = (feats - feats.min()) / (feats.max() - feats.min() + 1e-8)
        return feats.astype(np.float32)[..., np.newaxis]
    except Exception:
        return np.zeros((n_gfcc, fixed_frames, 1), dtype=np.float32)


# ─────────────────────────────────────────
# SAR — REFERENCE CHIRP
# ─────────────────────────────────────────

def generate_reference_chirp():
    """
    Correct linear chirp (matched filter for FMCW dechirping).
    Verified in verify_chirp_formula.py — produces cleaner beat freq than app formula.
    phase = 2*pi*(f0*t + (f1-f0)*t^2 / (2*T))
    """
    t     = np.linspace(0, CHIRP_DUR_S, CHIRP_ONLY_SAMPLES, endpoint=False)
    phase = 2.0 * np.pi * (
        CHIRP_START_HZ * t +
        (CHIRP_END_HZ - CHIRP_START_HZ) * t**2 / (2.0 * CHIRP_DUR_S)
    )
    chirp = np.sin(phase)
    gap   = np.zeros(int(SAMPLE_RATE * GAP_DUR_S))
    return np.concatenate([chirp, gap])


# ─────────────────────────────────────────
# SAR — TIMING ALIGNMENT
# ─────────────────────────────────────────

def align_reference(received, reference):
    """
    Cross-correlate received signal with reference chirp to find timing offset.
    Shifts reference to align with the actual echo arrival.
    Returns (aligned_reference, delay_samples).
    """
    chirp_only = reference[:CHIRP_ONLY_SAMPLES]
    search_len = min(len(received), SAMPLE_RATE * 2)
    corr       = np.correlate(received[:search_len], chirp_only, mode='full')
    delay      = int(np.argmax(np.abs(corr))) - len(chirp_only) + 1
    delay      = max(0, min(delay, len(reference) - 1))

    if delay > 0:
        aligned = np.concatenate([np.zeros(delay), reference[:-delay]])
    else:
        aligned = reference.copy()

    return aligned, delay


# ─────────────────────────────────────────
# SAR — DECHIRPING
# ─────────────────────────────────────────

def dechirp(segment, ref):
    """
    FMCW dechirp: multiply received × reference → IF signal.
    Lowpass  < 2kHz  : keeps beat frequencies (jaw distance range)
    Highpass > 10Hz  : removes DC + static face reflection
    """
    n     = min(len(segment), len(ref))
    mixed = segment[:n] * ref[:n]

    nyq   = 0.5 * SAMPLE_RATE
    b, a  = butter(4, 2000.0 / nyq, btype='low')
    mixed = lfilter(b, a, mixed)

    b2, a2 = butter(4, 10.0 / nyq, btype='high')
    return lfilter(b2, a2, mixed)


# ─────────────────────────────────────────
# SAR — RANGE-DOPPLER MAP  (MobiSys 2018)
# ─────────────────────────────────────────

def get_sar_range_doppler(segment, fixed_size=IMG_SIZE):
    """
    Extract SAR Range-Doppler map from a 1.5s chirp segment.

    Steps (MobiSys 2018 inspired):
      1. Align reference chirp to received signal via cross-correlation
      2. Dechirp: received × reference → IF signal
      3. Static clutter removal: subtract mean of IF signal
      4. FFT of IF signal → range profile (distance to jaw)
      5. STFT of IF signal → Range-Time map (jaw distance over time)
      6. Resize to 64×64×1

    The Range-Time map shows:
      - X axis: time within the 1.5s window
      - Y axis: beat frequency (∝ jaw distance)
      - Intensity: how strongly jaw reflects at that distance/time
    """
    try:
        ref          = generate_reference_chirp()
        ref_aligned, _ = align_reference(segment, ref)
        if_signal    = dechirp(segment, ref_aligned)

        # Static clutter removal — subtract mean (removes static face reflection)
        if_signal    = if_signal - np.mean(if_signal)

        # STFT of IF signal → Range-Time map
        # nperseg controls range resolution; noverlap controls time resolution
        f, t, Sxx = scipy_spectrogram(
            if_signal,
            fs=SAMPLE_RATE,
            nperseg=2048,
            noverlap=1792,
            window='hann'
        )

        # Keep only beat frequencies 10–500 Hz (jaw distance range)
        freq_mask = (f >= 10) & (f <= 500)
        rdm       = Sxx[freq_mask, :]

        # Log scale + normalize
        rdm = 10 * np.log10(rdm + 1e-12)
        rdm = (rdm - rdm.min()) / (rdm.max() - rdm.min() + 1e-8)

        # Resize to fixed_size
        img = cv2.resize(rdm.astype(np.float32), fixed_size, interpolation=cv2.INTER_CUBIC)
        return img[..., np.newaxis]

    except Exception:
        return np.zeros((*fixed_size, 1), dtype=np.float32)


# ─────────────────────────────────────────
# BATCH FEATURE EXTRACTION
# ─────────────────────────────────────────

def extract_features(segments, feature_type="stft", verbose=True):
    """
    Extract features for all segments.
    feature_type: "stft" | "mfcc" | "sar"
    Returns np.ndarray of shape (N, H, W, 1)
    """
    fn_map = {
        "stft": get_stft_image,
        "mfcc": get_mfcc_image,
        "gfcc": get_gfcc_image,
        "sar":  get_sar_range_doppler,
    }
    if feature_type not in fn_map:
        raise ValueError(f"Unknown feature_type: {feature_type}. Choose from {list(fn_map)}")

    fn      = fn_map[feature_type]
    results = []
    total   = len(segments)

    for i, seg in enumerate(segments):
        results.append(fn(seg))
        if verbose and (i + 1) % 200 == 0:
            print(f"  [{feature_type}] {i+1}/{total} segments processed")

    return np.array(results, dtype=np.float32)


# ─────────────────────────────────────────
# SPECAUGMENT
# ─────────────────────────────────────────

def spec_augment(image, num_time_masks=2, num_freq_masks=2,
                 time_mask_max=10, freq_mask_max=8):
    """
    SpecAugment: randomly mask time and frequency bands.
    Applied during training only — forces model to learn robust features.

    image: np.ndarray (H, W, 1)
    Returns augmented image of same shape.
    """
    img = image.copy()
    H, W, _ = img.shape

    # Frequency masking — mask horizontal bands
    for _ in range(num_freq_masks):
        f_width = np.random.randint(1, freq_mask_max + 1)
        f_start = np.random.randint(0, max(1, H - f_width))
        img[f_start : f_start + f_width, :, 0] = 0.0

    # Time masking — mask vertical bands
    for _ in range(num_time_masks):
        t_width = np.random.randint(1, time_mask_max + 1)
        t_start = np.random.randint(0, max(1, W - t_width))
        img[:, t_start : t_start + t_width, 0] = 0.0

    return img


def augment_batch(X, apply_prob=0.5, **kwargs):
    """
    Apply SpecAugment to a batch with probability apply_prob.
    X: np.ndarray (N, H, W, 1)
    """
    X_aug = X.copy()
    for i in range(len(X_aug)):
        if np.random.random() < apply_prob:
            X_aug[i] = spec_augment(X_aug[i], **kwargs)
    return X_aug
