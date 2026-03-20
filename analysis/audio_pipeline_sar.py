"""
SensEat: SAR Dechirp Pipeline
==============================
Based on the approach from MobiSys '18: "AIM: Acoustic Imaging on a Mobile"

HOW IT WORKS:
  1. The Android app generates a chirp signal and saves it as a reference PCM file
     (chirp_reference_set2.pcm) before playing it through the speaker.
  2. The microphone records reflections and saves them as a stereo PCM file.
  3. Here in Python: load both PCM files and multiply them.
     That's the dechirp — no filters needed.
  4. FFT of the multiplied signal → beat frequencies → these map to distances.

The reference PCM is: MONO, 16-bit little-endian, 44100 Hz, 1.5s (one full period).
The recorded PCM is: STEREO, 16-bit little-endian, 44100 Hz, 30s.

IMPORTANT:
  The reference chirp must be the ACTUAL file saved from the Android app
  (chirp_reference_set2.pcm), NOT a mathematically generated signal.
  The Android app uses:  amp = sin(2π · freq(t) · t)
  which is different from the standard linear chirp formula and must match exactly.

Android app chirp formula (generateChirpSamplesWithGaps, setting 2):
  chirpDurationMs = 1000ms,  gapDurationMs = 500ms
  freq(t) = 18000 + (20000 - 18000) * (t / 1.0)
  amp(i)  = sin(2π · freq(t) · t)   where t = i / 44100
"""

import os
import warnings
from pathlib import Path

import numpy as np
from scipy.signal import spectrogram
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# CONFIG  (must match Android app exactly)
# ─────────────────────────────────────────
SAMPLE_RATE       = 44100
CHIRP_START_HZ    = 18_000.0
CHIRP_END_HZ      = 20_000.0
CHIRP_DUR_MS      = 1000                             # setting 2
GAP_DUR_MS        = 500                              # setting 2
CHIRP_SAMPLES     = (CHIRP_DUR_MS * SAMPLE_RATE) // 1000   # 44100
GAP_SAMPLES       = (GAP_DUR_MS  * SAMPLE_RATE) // 1000   # 22050
PERIOD_SAMPLES    = CHIRP_SAMPLES + GAP_SAMPLES             # 66150
PERIOD_S          = PERIOD_SAMPLES / SAMPLE_RATE            # 1.5 s

SCRIPT_DIR = Path(__file__).resolve().parent
RAW_DATA   = SCRIPT_DIR / "raw_data"
FIG_DIR    = SCRIPT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

# Reference chirp PCM saved from the Android app
# Using the file uploaded to raw_data/transmitted_chirp/
REFERENCE_PCM = SCRIPT_DIR / "raw_data" / "transmitted_chirp" / "chirp_reference_set2 (2).pcm"


# ─────────────────────────────────────────
# LOAD FUNCTIONS
# ─────────────────────────────────────────

def load_reference_chirp() -> np.ndarray:
    """
    Load the reference chirp from the PCM file saved by the Android app.

    The Android app saves chirpSamples (MONO, 16-bit LE, 44100 Hz) before
    playing it through the speaker. This is the exact signal that was
    transmitted — use it directly as the dechirp reference.

    Falls back to replicating the Android formula if the file is not found.
    """
    if REFERENCE_PCM.exists():
        print(f"  Loading reference chirp from: {REFERENCE_PCM.name}")
        raw = np.fromfile(REFERENCE_PCM, dtype=np.int16)
        return raw.astype(np.float64) / 32768.0   # normalise to [-1, 1]

    # ── Fallback: replicate the EXACT Android formula ──────────────────────
    # This matches generateChirpSamplesWithGaps(setting=2) in MainActivity.kt:
    #   freq = chirpStartHz + (chirpEndHz - chirpStartHz) * (t / (chirpDurationMs/1000))
    #   amp  = sin(2π · freq · t)
    # Note: this is NOT the standard linear chirp formula. It differs from
    # sin(2π·(f0·t + (f1-f0)·t²/(2T))). Using this fallback is approximate.
    # use the actual saved file, not a generated one.
    print(f"  WARNING: {REFERENCE_PCM.name} not found.")
    print(f"  Using Android formula fallback — save the real file from the app!")

    ref = np.zeros(PERIOD_SAMPLES, dtype=np.float64)
    for i in range(CHIRP_SAMPLES):
        t = i / SAMPLE_RATE
        freq = CHIRP_START_HZ + (CHIRP_END_HZ - CHIRP_START_HZ) * (t / (CHIRP_DUR_MS / 1000.0))
        ref[i] = np.sin(2 * np.pi * freq * t)
    # gap remains zero
    return ref


def load_recorded_pcm_left(filepath) -> np.ndarray:
    """
    Load the recorded stereo PCM from the microphone.
    Extract the LEFT channel and normalise to [-1, 1].
    """
    raw = np.fromfile(filepath, dtype=np.int16)
    if raw.size % 2:
        raw = raw[:-1]
    stereo = raw.reshape(-1, 2)
    return stereo[:, 0].astype(np.float64) / 32768.0


# ─────────────────────────────────────────
# DECHIRP: just multiply
# ─────────────────────────────────────────

def dechirp(rx_period: np.ndarray, ref_chirp: np.ndarray) -> np.ndarray:
    """
    FMCW dechirp as described in the AIM paper :
      IF_signal = received_signal  ×  reference_chirp

    No filters. No DC removal. Just multiply.

    The chirp portion of rx_period (first CHIRP_SAMPLES samples) is multiplied
    by the reference chirp. The result contains beat frequencies that map
    directly to the distance of the reflector:
        f_beat [Hz]  →  R [m]  =  f_beat * c * T / (2 * BW)
    """
    # Only use the chirp portion (not the gap)
    rx   = rx_period[:CHIRP_SAMPLES]
    ref  = ref_chirp[:CHIRP_SAMPLES]
    n    = min(len(rx), len(ref))
    return rx[:n] * ref[:n]


# ─────────────────────────────────────────
# BUILD 2D IF MATRIX  s(n, k)
# ─────────────────────────────────────────

def build_if_matrix(signal: np.ndarray, ref_chirp: np.ndarray) -> np.ndarray:
    """
    Stack dechirped signals from every chirp period into a 2D matrix s(n, k).

      n = chirp (azimuth / time) index   → rows
      k = IF sample (range) index        → columns

    This is the core of the SAR approach from the AIM paper:
    the 2D matrix captures both range (k) and time/Doppler (n) information.
    """
    n_periods = len(signal) // PERIOD_SAMPLES
    if n_periods == 0:
        raise ValueError("Recording too short for even one chirp period.")

    S = np.zeros((n_periods, CHIRP_SAMPLES), dtype=np.float64)

    for n in range(n_periods):
        start    = n * PERIOD_SAMPLES
        rx_chunk = signal[start : start + PERIOD_SAMPLES]
        S[n]     = dechirp(rx_chunk, ref_chirp)

    return S   # shape: (N_periods, CHIRP_SAMPLES)


# ─────────────────────────────────────────
# RANGE PROFILE: FFT of each IF row
# ─────────────────────────────────────────

def range_profile_matrix(S: np.ndarray) -> tuple:
    """
    FFT each row of the IF matrix to get the range profile per chirp period.

    The FFT of the dechirped IF signal gives frequency bins. Each bin k
    corresponds to a beat frequency f_beat = k / T_chirp Hz, which maps to:
        R = f_beat * c * T / (2 * BW)
          = k * (343.0 * 1.0) / (2 * 2000)
          = k * 0.08575 m

    Returns:
        RT       : Range-Time matrix   (N_periods × N_freq_bins)
        range_m  : range axis in metres
    """
    # rfft: real input → positive frequencies only
    RT      = np.abs(np.fft.rfft(S, axis=1))          # (N, CHIRP_SAMPLES//2+1)
    n_bins  = RT.shape[1]

    # Frequency of bin k = k * (fs / N) = k * (44100 / 44100) = k Hz
    # (since the IF signal spans T_chirp = 1.0 s → frequency resolution = 1 Hz)
    freq_hz = np.arange(n_bins, dtype=np.float64)      # Hz
    beat_to_range = SOUND_SPEED * (CHIRP_DUR_MS / 1000.0) / (2.0 * (CHIRP_END_HZ - CHIRP_START_HZ))
    range_m = freq_hz * beat_to_range                  # metres

    return RT, range_m


SOUND_SPEED = 343.0   # m/s


# ─────────────────────────────────────────
# 2D FFT → SAR IMAGE
# ─────────────────────────────────────────

def sar_image(S: np.ndarray, range_m: np.ndarray) -> tuple:
    """
    Apply 2D FFT to the IF matrix to get the SAR image, as done in AIM.

    AIM Section 2: the IF signal for a point reflector is a 2D sinusoid.
    2D FFT maps each reflector to a spike in frequency space → the image.

      Range  axis  (columns): FFT of k → beat frequency → distance
      Azimuth axis (rows)   : FFT of n → Doppler → jaw velocity / cross-range

    Returns:
        image      : 2D magnitude (N_periods × N_freq_bins)
        doppler_hz : Doppler frequency axis (centred at 0)
    """
    RT = np.abs(np.fft.rfft(S, axis=1))   # range FFT first

    # Azimuth FFT (across chirp periods), fftshift to centre zero-Doppler
    SAR = np.fft.fftshift(np.fft.fft(RT, axis=0), axes=0)
    image = np.abs(SAR)

    PRF        = 1.0 / PERIOD_S                              # pulse repetition freq
    N_periods  = S.shape[0]
    doppler_hz = np.fft.fftshift(np.fft.fftfreq(N_periods, d=1.0 / PRF))

    return image, doppler_hz


# ─────────────────────────────────────────
# FULL PIPELINE
# ─────────────────────────────────────────

def run_pipeline(pcm_path, ref_chirp: np.ndarray, label: str) -> dict:
    """Run the full SAR pipeline on one recording."""
    print(f"\n[SAR] {label}: {Path(pcm_path).name}")

    signal    = load_recorded_pcm_left(pcm_path)
    S         = build_if_matrix(signal, ref_chirp)
    RT, range_m = range_profile_matrix(S)
    img, dop  = sar_image(S, range_m)

    print(f"  Chirp periods: {S.shape[0]}   Range bins: {S.shape[1]}")
    return dict(S=S, RT=RT, img=img, range_m=range_m, dop_hz=dop)


# ─────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────

def visualize():
    eat_pcm  = RAW_DATA / "002" / "1_002_02_01.pcm"
    idle_pcm = RAW_DATA / "Idle-gp" / "1_000_00_01.pcm"

    for p in [eat_pcm, idle_pcm]:
        if not p.exists():
            print(f"File not found: {p}")
            return

    ref = load_reference_chirp()

    eat  = run_pipeline(eat_pcm,  ref, "EATING")
    idle = run_pipeline(idle_pcm, ref, "IDLE")

    # Zoom to jaw region: 0–35 cm
    # Beat freq for 35 cm: f = 0.35 / 0.08575 ≈ 4.1 Hz → bin index ≈ 4
    # We'll zoom to first 20 bins to show the relevant range
    N_ZOOM = 20   # first 20 range bins = 0 to ~1.7 m (more than enough)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        "SAR Dechirp Pipeline  (AIM method)\n"
        "Received PCM  ×  Reference Chirp PCM  →  FFT  →  Beat Frequency = Distance",
        fontsize=11, fontweight='bold'
    )

    r_zoom = eat['range_m'][:N_ZOOM] * 100   # cm

    # ── [0,0] Range-Time — Eating ──────────────────────────────────
    ax = axes[0, 0]
    t_eat = np.arange(eat['RT'].shape[0]) * PERIOD_S
    im = ax.pcolormesh(
        r_zoom, t_eat,
        20 * np.log10(eat['RT'][:, :N_ZOOM] + 1e-12),
        shading='gouraud', cmap='plasma'
    )
    plt.colorbar(im, ax=ax, label='dB')
    ax.set_title("Range–Time Map  (Eating)")
    ax.set_xlabel("Range (cm)  ≡  Beat Frequency × 8.575 cm/Hz")
    ax.set_ylabel("Time (s)")

    # ── [0,1] Range-Time — Idle ────────────────────────────────────
    ax = axes[0, 1]
    t_idle = np.arange(idle['RT'].shape[0]) * PERIOD_S
    im = ax.pcolormesh(
        idle['range_m'][:N_ZOOM] * 100, t_idle,
        20 * np.log10(idle['RT'][:, :N_ZOOM] + 1e-12),
        shading='gouraud', cmap='plasma'
    )
    plt.colorbar(im, ax=ax, label='dB')
    ax.set_title("Range–Time Map  (Idle)")
    ax.set_xlabel("Range (cm)")
    ax.set_ylabel("Time (s)")

    # ── [1,0] SAR Image (2D FFT) — Eating ─────────────────────────
    ax = axes[1, 0]
    im = ax.pcolormesh(
        r_zoom, eat['dop_hz'],
        20 * np.log10(eat['img'][:, :N_ZOOM] + 1e-12),
        shading='gouraud', cmap='viridis'
    )
    plt.colorbar(im, ax=ax, label='dB')
    ax.set_title("SAR Image  (Eating)\n2D FFT of IF matrix — Doppler vs Range")
    ax.set_xlabel("Range (cm)")
    ax.set_ylabel("Doppler (Hz)")
    ax.axhline(0, color='white', linewidth=0.5, linestyle='--')

    # ── [1,1] SAR Image (2D FFT) — Idle ───────────────────────────
    ax = axes[1, 1]
    im = ax.pcolormesh(
        idle['range_m'][:N_ZOOM] * 100, idle['dop_hz'],
        20 * np.log10(idle['img'][:, :N_ZOOM] + 1e-12),
        shading='gouraud', cmap='viridis'
    )
    plt.colorbar(im, ax=ax, label='dB')
    ax.set_title("SAR Image  (Idle)\n2D FFT of IF matrix — Doppler vs Range")
    ax.set_xlabel("Range (cm)")
    ax.set_ylabel("Doppler (Hz)")
    ax.axhline(0, color='white', linewidth=0.5, linestyle='--')

    plt.tight_layout()
    out = FIG_DIR / "sar_aim_pipeline.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSAR image saved -> {out}")


if __name__ == '__main__':
    visualize()
