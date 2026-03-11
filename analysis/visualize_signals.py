"""
SensEat — Signal-Level Visualizations
======================================
Generates detailed signal-analysis figures for meeting presentation.
Shows what the pipeline "sees" at each stage: raw signal, FFT, and
all 6 feature representations side-by-side for Eating vs Idle.

Usage:  python visualize_signals.py
Output: Saves PNGs in analysis/figures/
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import librosa
import librosa.display
import pywt
import cv2
from scipy.signal import butter, lfilter, stft as scipy_stft
from pathlib import Path

plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 200, 'font.size': 11,
    'axes.titlesize': 13, 'axes.labelsize': 11, 'figure.facecolor': 'white',
})

SCRIPT_DIR = Path(__file__).resolve().parent
FIG_DIR = SCRIPT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

SR = 44100
DIRECT_PATH = int((0.30 / 343.0) * SR)
CHUNK = int(1.5 * SR)  # 66150

# ─── Helpers ───

def load_chunk(filepath, chunk_idx=10):
    """Load a specific 1.5s chunk from a PCM file (chunk_idx=10 → middle-ish)."""
    raw = np.fromfile(filepath, dtype=np.int16)
    if raw.size % 2 != 0:
        raw = raw[:-1]
    left = raw.reshape(-1, 2)[:, 0].astype(np.float32) / 32768.0
    left = left[DIRECT_PATH:]
    # Bandpass
    nyq = 0.5 * SR
    b, a = butter(6, [17500/nyq, 20500/nyq], btype='band')
    filtered = lfilter(b, a, left)
    # Extract specific chunk
    start = chunk_idx * CHUNK
    if start + CHUNK > len(filtered):
        start = len(filtered) // 2
    return filtered[start : start + CHUNK]


def find_files():
    """Find one eating and one idle file."""
    raw = SCRIPT_DIR / "raw_data"
    eating_file = idle_file = None
    for f in sorted((raw / "001").glob("*.pcm")):
        if not f.stem.endswith("_idleTail"):
            eating_file = f
            break
    for f in sorted((raw / "Idle").glob("*.pcm")):
        if not f.stem.endswith("_idleTail"):
            idle_file = f
            break
    return eating_file, idle_file


# ═══════════════════════════════════════════════════════════
# FIGURE 7: FFT Comparison (Eating vs Idle)
# ═══════════════════════════════════════════════════════════

def plot_fft_comparison(eat_chunk, idle_chunk, eat_name, idle_name):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for col, (chunk, label, name, color) in enumerate([
        (eat_chunk, "EATING", eat_name, '#1565C0'),
        (idle_chunk, "IDLE", idle_name, '#E65100'),
    ]):
        fft_vals = np.abs(np.fft.rfft(chunk))
        freqs = np.fft.rfftfreq(len(chunk), 1/SR)

        # Focus on chirp band
        mask = (freqs >= 16000) & (freqs <= 22000)
        axes[col].plot(freqs[mask], fft_vals[mask], linewidth=0.8, color=color)
        axes[col].fill_between(freqs[mask], fft_vals[mask], alpha=0.3, color=color)
        axes[col].axvspan(18000, 20000, alpha=0.1, color='green', label='Chirp Band (18-20 kHz)')
        axes[col].set_title(f'{label} — FFT\n({name})', fontweight='bold')
        axes[col].set_xlabel('Frequency (Hz)')
        axes[col].set_ylabel('Magnitude')
        axes[col].legend(fontsize=9)
        axes[col].grid(alpha=0.3)

    plt.suptitle('FFT Frequency Spectrum: Eating vs Idle (1.5s chunk)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    out = FIG_DIR / "fig7_fft_comparison.png"
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {out}")


# ═══════════════════════════════════════════════════════════
# FIGURE 8: All 6 Feature Representations Side-by-Side
# ═══════════════════════════════════════════════════════════

def plot_all_features_sidebyside(eat_chunk, idle_chunk, eat_name, idle_name):
    """6 rows (one per feature type) × 2 columns (eating vs idle)."""
    fig, axes = plt.subplots(6, 2, figsize=(16, 24))

    for col, (chunk, label, name) in enumerate([
        (eat_chunk, "EATING", eat_name),
        (idle_chunk, "IDLE", idle_name),
    ]):
        cmap_eat = 'inferno' if col == 0 else 'YlOrRd'

        # Row 0: Time-domain waveform
        t = np.arange(len(chunk)) / SR
        axes[0, col].plot(t, chunk, linewidth=0.3, color='#1565C0' if col == 0 else '#E65100')
        axes[0, col].set_title(f'{label}: Time Domain Waveform', fontweight='bold')
        axes[0, col].set_xlabel('Time (s)')
        axes[0, col].set_ylabel('Amplitude')
        axes[0, col].grid(alpha=0.3)

        # Row 1: STFT Spectrogram (what CNN+STFT sees)
        D = np.abs(librosa.stft(chunk, n_fft=2048))
        S_db = librosa.amplitude_to_db(D, ref=np.max)
        img = librosa.display.specshow(S_db, sr=SR, x_axis='time', y_axis='hz',
                                       ax=axes[1, col], cmap=cmap_eat)
        axes[1, col].set_ylim(16000, 22000)
        axes[1, col].set_title(f'{label}: STFT Spectrogram (CNN input)', fontweight='bold')
        plt.colorbar(img, ax=axes[1, col], label='dB', format='%+2.0f')

        # Row 2: Mel Spectrogram
        S_mel = librosa.feature.melspectrogram(y=chunk, sr=SR, n_mels=64, n_fft=2048)
        S_mel_db = librosa.power_to_db(S_mel, ref=np.max)
        img = librosa.display.specshow(S_mel_db, sr=SR, x_axis='time', y_axis='mel',
                                       ax=axes[2, col], cmap=cmap_eat)
        axes[2, col].set_title(f'{label}: Mel Spectrogram (CNN input)', fontweight='bold')
        plt.colorbar(img, ax=axes[2, col], label='dB', format='%+2.0f')

        # Row 3: MFCC
        mfcc = librosa.feature.mfcc(y=chunk, sr=SR, n_mfcc=40)
        img = librosa.display.specshow(mfcc, sr=SR, x_axis='time',
                                       ax=axes[3, col], cmap='coolwarm')
        axes[3, col].set_title(f'{label}: MFCC (40 coefficients, CNN input)', fontweight='bold')
        axes[3, col].set_ylabel('MFCC Coefficient')
        plt.colorbar(img, ax=axes[3, col])

        # Row 4: CWT Scalogram
        scales = np.arange(1, 65)
        coeffs, _ = pywt.cwt(chunk, scales, 'morl')
        scalo = np.abs(coeffs)
        axes[4, col].imshow(scalo, aspect='auto', cmap=cmap_eat,
                           extent=[0, 1.5, scales[-1], scales[0]])
        axes[4, col].set_title(f'{label}: CWT Scalogram (Morlet, CNN input)', fontweight='bold')
        axes[4, col].set_xlabel('Time (s)')
        axes[4, col].set_ylabel('Scale')

        # Row 5: Statistical features as a mini bar chart
        zcr = librosa.feature.zero_crossing_rate(chunk)[0]
        stat_names = ['Mean', 'Std', 'Max', 'Min', 'Power', 'ZCR']
        stat_vals = [np.mean(chunk), np.std(chunk), np.max(chunk),
                     np.min(chunk), np.mean(chunk**2), np.mean(zcr)]
        bar_colors = ['#2196F3' if col == 0 else '#FF9800'] * 6
        axes[5, col].barh(stat_names, stat_vals, color=bar_colors, edgecolor='white')
        axes[5, col].set_title(f'{label}: Statistical Features (SVM/RF/kNN input)', fontweight='bold')
        axes[5, col].set_xlabel('Value')
        axes[5, col].grid(axis='x', alpha=0.3)
        # Add value labels
        for i, v in enumerate(stat_vals):
            axes[5, col].text(v, i, f' {v:.6f}', va='center', fontsize=8)

    plt.suptitle('All 6 Feature Representations: Eating vs Idle\n(Single 1.5s chunk, after bandpass filtering)',
                fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    out = FIG_DIR / "fig8_all_features_eating_vs_idle.png"
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {out}")


# ═══════════════════════════════════════════════════════════
# FIGURE 9: Multiple Chunks Overlay (shows consistency)
# ═══════════════════════════════════════════════════════════

def plot_multiple_chunks_overlay(eating_file, idle_file):
    """Show 5 different 1.5s chunks from eating and idle, overlaid on same axes."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    for col, (filepath, label, color) in enumerate([
        (eating_file, "EATING", '#1565C0'),
        (idle_file, "IDLE", '#E65100'),
    ]):
        # Load multiple chunks
        raw = np.fromfile(filepath, dtype=np.int16)
        if raw.size % 2 != 0:
            raw = raw[:-1]
        left = raw.reshape(-1, 2)[:, 0].astype(np.float32) / 32768.0
        left = left[DIRECT_PATH:]
        nyq = 0.5 * SR
        b, a = butter(6, [17500/nyq, 20500/nyq], btype='band')
        filtered = lfilter(b, a, left)

        chunk_indices = [0, 5, 10, 15, 20]
        t = np.arange(CHUNK) / SR

        # Row 0: Time domain overlay
        for ci in chunk_indices:
            start = ci * CHUNK
            if start + CHUNK <= len(filtered):
                seg = filtered[start : start + CHUNK]
                axes[0, col].plot(t, seg, linewidth=0.3, alpha=0.6, label=f'Chunk {ci}')
        axes[0, col].set_title(f'{label} — 5 Different 1.5s Chunks (Time Domain)', fontweight='bold')
        axes[0, col].set_xlabel('Time (s)')
        axes[0, col].set_ylabel('Amplitude')
        axes[0, col].legend(fontsize=8, loc='upper right')
        axes[0, col].grid(alpha=0.3)

        # Row 1: FFT overlay
        for ci in chunk_indices:
            start = ci * CHUNK
            if start + CHUNK <= len(filtered):
                seg = filtered[start : start + CHUNK]
                fft_vals = np.abs(np.fft.rfft(seg))
                freqs = np.fft.rfftfreq(len(seg), 1/SR)
                mask = (freqs >= 17000) & (freqs <= 21000)
                axes[1, col].plot(freqs[mask], fft_vals[mask], linewidth=0.6, alpha=0.6, label=f'Chunk {ci}')
        axes[1, col].axvspan(18000, 20000, alpha=0.1, color='green')
        axes[1, col].set_title(f'{label} — FFT of 5 Chunks (Chirp Band)', fontweight='bold')
        axes[1, col].set_xlabel('Frequency (Hz)')
        axes[1, col].set_ylabel('Magnitude')
        axes[1, col].legend(fontsize=8, loc='upper right')
        axes[1, col].grid(alpha=0.3)

    plt.suptitle('Signal Consistency: Multiple 1.5s Chunks from Same Recording',
                fontsize=15, fontweight='bold')
    plt.tight_layout()
    out = FIG_DIR / "fig9_multiple_chunks_overlay.png"
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {out}")


# ═══════════════════════════════════════════════════════════
# FIGURE 10: Segmentation Diagram (first 10 chunks of a file)
# ═══════════════════════════════════════════════════════════

def plot_segmentation_diagram(eating_file):
    """Show how the raw signal gets chopped into 1.5s chunks."""
    raw = np.fromfile(eating_file, dtype=np.int16)
    if raw.size % 2 != 0:
        raw = raw[:-1]
    left = raw.reshape(-1, 2)[:, 0].astype(np.float32) / 32768.0
    left = left[DIRECT_PATH:]
    nyq = 0.5 * SR
    b, a = butter(6, [17500/nyq, 20500/nyq], btype='band')
    filtered = lfilter(b, a, left)

    # Show first 15 seconds (10 chunks)
    show_samples = min(10 * CHUNK, len(filtered))
    t = np.arange(show_samples) / SR
    signal = filtered[:show_samples]

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(t, signal, linewidth=0.3, color='#1565C0')

    # Draw chunk boundaries
    colors = plt.cm.Set3(np.linspace(0, 1, 10))
    for i in range(10):
        start_s = i * 1.5
        end_s = (i + 1) * 1.5
        if end_s * SR <= show_samples:
            ax.axvspan(start_s, end_s, alpha=0.15, color=colors[i])
            ax.axvline(start_s, color='red', linewidth=0.8, alpha=0.5, linestyle='--')
            ax.text(start_s + 0.75, ax.get_ylim()[1] * 0.85, f'Chunk {i}',
                   ha='center', fontsize=8, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor=colors[i], alpha=0.8))

    ax.set_title(f'Segmentation: First 15 Seconds of {eating_file.stem}\n'
                 f'Each colored block = one 1.5s chunk (66,150 samples) → one feature vector → one prediction',
                fontsize=13, fontweight='bold')
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Amplitude (bandpass filtered)', fontsize=12)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out = FIG_DIR / "fig10_segmentation_diagram.png"
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {out}")


# ═══════════════════════════════════════════════════════════
# RUN ALL
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("Generating signal-level figures...\n")

    eating_file, idle_file = find_files()
    if eating_file is None or idle_file is None:
        print("❌ Could not find eating/idle files!")
        exit(1)

    print(f"Eating file: {eating_file.name}")
    print(f"Idle file:   {idle_file.name}\n")

    # Load representative chunks
    eat_chunk = load_chunk(str(eating_file), chunk_idx=10)
    idle_chunk = load_chunk(str(idle_file), chunk_idx=10)

    plot_fft_comparison(eat_chunk, idle_chunk, eating_file.stem, idle_file.stem)
    plot_all_features_sidebyside(eat_chunk, idle_chunk, eating_file.stem, idle_file.stem)
    plot_multiple_chunks_overlay(eating_file, idle_file)
    plot_segmentation_diagram(eating_file)

    print(f"\n✅ All signal figures saved to: {FIG_DIR}/")
