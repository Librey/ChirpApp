"""
Time-Frequency Domain Visualizer
=================================
Generates spectrogram figures for each food recording.
A spectrogram shows:
  - X-axis : Time (seconds)
  - Y-axis : Frequency (Hz)
  - Color  : Signal strength (brighter = stronger signal at that frequency/time)

Usage:
  1. Put your new recordings inside:
        analysis/raw_data/Participant-1-02232026/
  2. Run:
        py plot_time_freq.py
  3. Figures are saved to:
        analysis/figures/time_freq/
"""

import numpy as np
import re
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────

SAMPLE_RATE  = 44100          # samples recorded per second by the app
DATA_FOLDER  = Path(__file__).resolve().parent / "old_data"
OUTPUT_DIR   = Path(__file__).resolve().parent / "figures" / "time_freq"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# We zoom the frequency axis to our chirp band (18–20 kHz)
# A little wider (17–21 kHz) so we can see the edges clearly
FREQ_MIN_HZ  = 17_000
FREQ_MAX_HZ  = 21_000

# IRB food code → food name mapping
FOOD_NAMES = {
    0:  "Idle",
    1:  "Tortilla",
    2:  "Mandarin",
    3:  "Chicken_Breast",
    4:  "Cheeze_It",
    5:  "Carrots",
    6:  "Chocolate",
    7:  "Yogurt",
    8:  "Noodles",
    9:  "Water",
    10: "Coke"
}

# Matches IRB format: W_XXX_YY_ZZ  (e.g. 1_021_04_01)
IRB_RE = re.compile(r"^(\d+)_(\d{3})_(\d{2})_(\d{2})$")


# ─────────────────────────────────────────────────────────────
# STEP 1 — LOAD PCM FILE
# ─────────────────────────────────────────────────────────────

def load_pcm(filepath):
    """
    Read a raw PCM file recorded by the Android app.

    The app saves audio as:
      - 16-bit integers  (each sample is a number from -32768 to +32767)
      - Stereo           (2 channels: LEFT, RIGHT interleaved)
      - 44100 Hz         (44100 samples per second per channel)

    We:
      1. Read all bytes as 16-bit integers
      2. Reshape into pairs [LEFT, RIGHT]
      3. Take only LEFT channel
      4. Divide by 32768 to normalise to range [-1.0, +1.0]
    """
    raw    = np.fromfile(filepath, dtype=np.int16)   # read raw bytes as 16-bit ints
    if raw.size % 2 != 0:
        raw = raw[:-1]                               # drop last byte if odd (safety)
    stereo = raw.reshape(-1, 2)                      # reshape: each row = [LEFT, RIGHT]
    left   = stereo[:, 0].astype(np.float32)         # take LEFT channel only
    left   = left / 32768.0                          # normalise: -32768..32767 → -1..+1
    return left


# ─────────────────────────────────────────────────────────────
# STEP 2 — COMPUTE SPECTROGRAM
# ─────────────────────────────────────────────────────────────

def compute_spectrogram(signal):
    """
    Compute the time-frequency spectrogram of the signal.

    How it works:
      - Split the signal into short overlapping windows (each 2048 samples = ~46ms)
      - For each window: run FFT → get which frequencies are present
      - Stack all windows side by side → 2D image: rows=frequency, cols=time

    Parameters used:
      nperseg=2048   : window length (2048 samples ÷ 44100 Hz ≈ 46ms per window)
      noverlap=1024  : windows overlap by 50% → smooth time axis
      window='hann'  : Hann window reduces spectral leakage (standard practice)

    Returns:
      freqs  : array of frequency values (Hz)  — the Y-axis
      times  : array of time values (seconds)  — the X-axis
      Sxx    : 2D power array (frequency × time) — the color values
    """
    freqs, times, Sxx = spectrogram(
        signal,
        fs       = SAMPLE_RATE,
        nperseg  = 2048,          # window size: 2048 samples
        noverlap = 1024,          # overlap: 50%
        window   = 'hann',        # smooth window to reduce FFT artifacts
        scaling  = 'spectrum'     # return power spectrum (not density)
    )
    return freqs, times, Sxx


# ─────────────────────────────────────────────────────────────
# STEP 3 — PLOT AND SAVE ONE SPECTROGRAM
# ─────────────────────────────────────────────────────────────

def plot_spectrogram(signal, title, out_path):
    """
    Plot the time-frequency spectrogram for one recording and save it.

    The color scale uses dBFS (decibels relative to full scale):
      dBFS = 10 * log10(power / max_power)
    This means the loudest point in the recording = 0 dB.
    Everything else is negative (e.g. -60 dB = much quieter than the peak).
    vmin=-120, vmax=0 keeps the color scale fixed across all food plots
    so you can directly compare brightness between recordings.
    """
    freqs, times, Sxx = compute_spectrogram(signal)

    # Convert power to dBFS (relative to max power in this recording)
    # Dividing by Sxx.max() makes the peak always 0 dB
    # Add 1e-12 to avoid log(0) which would give -infinity
    Sxx_dB = 10 * np.log10(Sxx / Sxx.max() + 1e-12)

    # Keep only the rows (frequencies) between FREQ_MIN_HZ and FREQ_MAX_HZ
    freq_mask = (freqs >= FREQ_MIN_HZ) & (freqs <= FREQ_MAX_HZ)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle(title, fontsize=13, fontweight='bold')

    # ── TOP PLOT: Time Domain ──────────────────────────────────
    # Shows the raw amplitude of the signal over time
    # X-axis = time in seconds, Y-axis = amplitude (-1 to +1)
    t_axis = np.arange(len(signal)) / SAMPLE_RATE   # convert sample index to seconds
    axes[0].plot(t_axis, signal, linewidth=0.4, color='steelblue')
    axes[0].set_title("Time Domain  (raw signal amplitude)")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(alpha=0.3)

    # ── BOTTOM PLOT: Spectrogram (Time-Frequency) ──────────────
    # Shows how signal energy is distributed across frequencies over time
    # X-axis = time, Y-axis = frequency, Color = power in dB
    im = axes[1].pcolormesh(
        times,                    # X-axis: time in seconds
        freqs[freq_mask] / 1000,  # Y-axis: frequency in kHz (divide by 1000)
        Sxx_dB[freq_mask, :],     # Color: power in dBFS, zoomed to our band
        shading = 'gouraud',      # smooth interpolation between pixels
        cmap    = 'inferno',      # color map: black=quiet, yellow=loud
        vmin    = -120,           # fixed min: anything quieter than -120 dB = black
        vmax    = 0               # fixed max: 0 dBFS = peak of the recording = brightest
    )
    plt.colorbar(im, ax=axes[1], label='Power (dBFS)')

    # Draw reference lines at chirp start (18 kHz) and end (20 kHz)
    axes[1].axhline(18, color='cyan',  linewidth=1, linestyle='--', label='Chirp start 18 kHz')
    axes[1].axhline(20, color='lime',  linewidth=1, linestyle='--', label='Chirp end   20 kHz')
    axes[1].legend(fontsize=8, loc='upper right')

    axes[1].set_title("Time-Frequency Spectrogram  (zoomed to chirp band)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Frequency (kHz)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path.name}")


# ─────────────────────────────────────────────────────────────
# STEP 4 — COMPARISON PLOT: all foods on one figure
# ─────────────────────────────────────────────────────────────

def plot_all_comparison(food_signals):
    """
    One big figure with one spectrogram per food item, side by side.
    Makes it easy to compare how different foods look in the chirp band.

    food_signals: dict  {food_name: signal_array}
    """
    n      = len(food_signals)
    ncols  = 3
    nrows  = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    fig.suptitle(
        "Time-Frequency Spectrograms — All Food Items\n"
        "(Y-axis: frequency kHz | X-axis: time s | Color: signal power dB)",
        fontsize=13, fontweight='bold'
    )

    # Flatten axes array for easy iteration even if only one row
    axes_flat = np.array(axes).flatten()

    for idx, (food_name, signal) in enumerate(sorted(food_signals.items())):
        ax = axes_flat[idx]

        freqs, times, Sxx = compute_spectrogram(signal)
        Sxx_dB    = 10 * np.log10(Sxx / Sxx.max() + 1e-12)
        freq_mask = (freqs >= FREQ_MIN_HZ) & (freqs <= FREQ_MAX_HZ)

        im = ax.pcolormesh(
            times,
            freqs[freq_mask] / 1000,
            Sxx_dB[freq_mask, :],
            shading = 'gouraud',
            cmap    = 'inferno',
            vmin    = -120,
            vmax    = 0
        )
        ax.axhline(18, color='cyan', linewidth=0.8, linestyle='--')
        ax.axhline(20, color='lime', linewidth=0.8, linestyle='--')
        ax.set_title(food_name, fontweight='bold')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Freq (kHz)")
        plt.colorbar(im, ax=ax, label='dB')

    # Hide any unused subplot panels
    for idx in range(len(food_signals), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.tight_layout()
    out = OUTPUT_DIR / "all_foods_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved comparison: {out.name}")


# ─────────────────────────────────────────────────────────────
# MAIN — find files, load, plot
# ─────────────────────────────────────────────────────────────

def main():
    print(f"Looking for PCM files in: {DATA_FOLDER}")

    if not DATA_FOLDER.exists():
        print(f"ERROR: Folder not found: {DATA_FOLDER}")
        print("Please upload your recordings to that folder and run again.")
        return

    # Find all PCM files (skip idleTail files)
    pcm_files = sorted([
        f for f in DATA_FOLDER.glob("*.pcm")
        if "idleTail" not in f.name
    ])

    if not pcm_files:
        print("No PCM files found. Upload your recordings and try again.")
        return

    print(f"Found {len(pcm_files)} files\n")

    # Group files by food label
    # Supports IRB format:   W_XXX_YY_ZZ  (e.g. 1_021_04_01)
    # Supports pilot format: chirp_<food>_30cm_... (e.g. chirp_coke_30cm_...)
    food_groups = {}
    for f in pcm_files:
        m = IRB_RE.match(f.stem)
        if m:
            # IRB format — extract food code from position 3 (YY)
            food_code = int(m.group(3))
            food_name = FOOD_NAMES.get(food_code, f"Code_{food_code:02d}")
        else:
            # Pilot format — extract food name from position 1
            parts     = f.stem.split('_')
            food_name = parts[1] if len(parts) > 1 else "unknown"

        if food_name not in food_groups:
            food_groups[food_name] = []
        food_groups[food_name].append(f)

    print("Foods found:", list(food_groups.keys()), "\n")

    # For each food: pick the first file, load it, plot individual spectrogram
    food_signals = {}
    for food_name, files in sorted(food_groups.items()):
        filepath = files[0]                   # take first recording of this food
        print(f"Processing: {food_name}  →  {filepath.name}")

        signal = load_pcm(filepath)           # load and normalise

        # Individual spectrogram for this food
        out_path = OUTPUT_DIR / f"spectrogram_{food_name}.png"
        plot_spectrogram(
            signal   = signal,
            title    = f"Food: {food_name}  |  File: {filepath.name}",
            out_path = out_path
        )

        food_signals[food_name] = signal      # store for comparison plot

    # One big comparison figure with all foods
    print("\nGenerating all-foods comparison figure...")
    plot_all_comparison(food_signals)

    print(f"\nDone. All figures saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
