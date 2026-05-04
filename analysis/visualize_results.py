"""
SensEat — Visualization Script for Binary Classification Results
================================================================
Generates publication-quality figures for meeting presentation.

Usage:  python visualize_results.py
Output: Saves PNG files in analysis/figures/
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

# Use non-interactive backend
matplotlib.use('Agg')

# ─── Style ───
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.facecolor': 'white',
})

SCRIPT_DIR = Path(__file__).resolve().parent
FIG_DIR = SCRIPT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ─── Load results ───
df = pd.read_csv(SCRIPT_DIR / "pipeline_1_5s_results.csv")

# ═══════════════════════════════════════════════════════════
# FIGURE 1: Bar chart comparing all 10 combinations (F1)
# ═══════════════════════════════════════════════════════════

def plot_f1_comparison():
    """Side-by-side bar chart of F1 scores: CV vs 80/20 for all combos."""
    df_cv = df[df['eval'] == 'CV'].sort_values('f1', ascending=True).reset_index(drop=True)
    df_split = df[df['eval'] == '80/20'].sort_values('f1', ascending=True).reset_index(drop=True)

    # Merge on model+feature to align
    df_cv['combo'] = df_cv['model'] + ' + ' + df_cv['feature']
    df_split['combo'] = df_split['model'] + ' + ' + df_split['feature']
    merged = df_cv[['combo', 'f1']].merge(df_split[['combo', 'f1']], on='combo', suffixes=('_cv', '_holdout'))
    merged = merged.sort_values('f1_cv', ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, 7))
    y = np.arange(len(merged))
    bar_h = 0.35

    bars1 = ax.barh(y - bar_h/2, merged['f1_cv'],      bar_h, label='5-Fold CV', color='#2196F3', edgecolor='white')
    bars2 = ax.barh(y + bar_h/2, merged['f1_holdout'],  bar_h, label='80/20 Holdout', color='#FF9800', edgecolor='white')

    ax.set_yticks(y)
    ax.set_yticklabels(merged['combo'], fontsize=11)
    ax.set_xlabel('F1 Score', fontsize=13)
    ax.set_title('Binary Classification: Eating vs Idle — All Model × Feature Combinations\n(T = 1.5s chirp-level segmentation)', fontsize=14, fontweight='bold')
    ax.set_xlim(0.8, 1.02)
    ax.legend(fontsize=12, loc='lower right')
    ax.axvline(x=0.9, color='gray', linestyle='--', alpha=0.4, label='F1 = 0.90 baseline')
    ax.grid(axis='x', alpha=0.3)

    # Add value labels
    for bar in bars1:
        w = bar.get_width()
        ax.text(w + 0.003, bar.get_y() + bar.get_height()/2, f'{w:.3f}', va='center', fontsize=9, color='#1565C0')
    for bar in bars2:
        w = bar.get_width()
        ax.text(w + 0.003, bar.get_y() + bar.get_height()/2, f'{w:.3f}', va='center', fontsize=9, color='#E65100')

    plt.tight_layout()
    out = FIG_DIR / "fig1_f1_comparison_all_combos.png"
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {out}")


# ═══════════════════════════════════════════════════════════
# FIGURE 2: Accuracy bar chart with error bars (CV)
# ═══════════════════════════════════════════════════════════

def plot_accuracy_with_errorbars():
    """CV accuracy with ±std error bars for all combinations."""
    df_cv = df[df['eval'] == 'CV'].copy()
    df_cv['combo'] = df_cv['model'] + ' + ' + df_cv['feature']
    df_cv = df_cv.sort_values('accuracy', ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, 7))

    colors = []
    for _, row in df_cv.iterrows():
        if row['model'] == 'CNN':
            colors.append('#E91E63')
        elif row['model'] == 'RF':
            colors.append('#4CAF50')
        elif row['model'] == 'SVM':
            colors.append('#2196F3')
        else:
            colors.append('#FF9800')

    bars = ax.barh(df_cv['combo'], df_cv['accuracy'], xerr=df_cv['acc_std'],
                   color=colors, edgecolor='white', capsize=4, error_kw={'linewidth': 1.5})

    ax.set_xlabel('Accuracy (5-Fold CV)', fontsize=13)
    ax.set_title('Binary Classification Accuracy with Standard Deviation\n(T = 1.5s chirp-level segmentation)', fontsize=14, fontweight='bold')
    ax.set_xlim(0.75, 1.02)
    ax.grid(axis='x', alpha=0.3)

    # Add value labels
    for bar, (_, row) in zip(bars, df_cv.iterrows()):
        w = bar.get_width()
        ax.text(w + row['acc_std'] + 0.005, bar.get_y() + bar.get_height()/2,
                f'{w:.3f}±{row["acc_std"]:.3f}', va='center', fontsize=9)

    # Legend for model types
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#E91E63', label='CNN'),
        Patch(facecolor='#4CAF50', label='Random Forest'),
        Patch(facecolor='#2196F3', label='SVM'),
        Patch(facecolor='#FF9800', label='kNN'),
    ]
    ax.legend(handles=legend_elements, fontsize=11, loc='lower right')

    plt.tight_layout()
    out = FIG_DIR / "fig2_accuracy_cv_errorbars.png"
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {out}")


# ═══════════════════════════════════════════════════════════
# FIGURE 3: Heatmap of metrics (Precision, Recall, F1)
# ═══════════════════════════════════════════════════════════

def plot_metrics_heatmap():
    """Heatmap showing all metrics for all combinations (80/20 holdout)."""
    df_split = df[df['eval'] == '80/20'].copy()
    df_split['combo'] = df_split['model'] + ' + ' + df_split['feature']
    df_split = df_split.sort_values('f1', ascending=False).reset_index(drop=True)

    data = df_split[['accuracy', 'precision', 'recall', 'f1']].values
    combos = df_split['combo'].values

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0.8, vmax=1.0)

    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1'], fontsize=12)
    ax.set_yticks(range(len(combos)))
    ax.set_yticklabels(combos, fontsize=11)

    # Annotate cells
    for i in range(len(combos)):
        for j in range(4):
            val = data[i, j]
            color = 'white' if val > 0.95 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=10, fontweight='bold', color=color)

    ax.set_title('Performance Metrics Heatmap — 80/20 Holdout\n(T = 1.5s chirp-level segmentation)', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Score', shrink=0.8)
    plt.tight_layout()
    out = FIG_DIR / "fig3_metrics_heatmap.png"
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {out}")


# ═══════════════════════════════════════════════════════════
# FIGURE 4: Dataset composition pie/bar chart
# ═══════════════════════════════════════════════════════════

def plot_dataset_composition():
    """Show dataset composition: segments per class and per participant."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Pie: Eating vs Idle
    sizes = [855, 190]
    labels = [f'Eating\n({sizes[0]} segments)', f'Idle\n({sizes[1]} segments)']
    colors_pie = ['#2196F3', '#FF9800']
    explode = (0.04, 0.04)
    ax1.pie(sizes, explode=explode, labels=labels, colors=colors_pie, autopct='%1.1f%%',
            shadow=False, startangle=90, textprops={'fontsize': 12})
    ax1.set_title('Class Distribution', fontsize=14, fontweight='bold')

    # Bar: Files per source folder
    folders = ['P001\n(14 files)', 'P009\n(14 files)', 'P020\n(17 files)', 'Idle\n(10 files)']
    # Approximate segment counts per folder (39 segments per 30s file)
    seg_counts = [14*39, 14*39, 17*39, 10*19]  # idle files may differ
    folder_colors = ['#2196F3', '#42A5F5', '#64B5F6', '#FF9800']
    bars = ax2.bar(folders, seg_counts, color=folder_colors, edgecolor='white', linewidth=1.5)
    ax2.set_ylabel('Approximate Segments', fontsize=12)
    ax2.set_title('Segments by Source Folder', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    # Value labels
    for bar in bars:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 5, str(int(h)), ha='center', fontsize=11, fontweight='bold')

    plt.suptitle('SensEat Dataset — 1.5s Chirp-Level Segmentation (T = 1.5s, SR = 44100 Hz)',
                 fontsize=14, fontweight='bold', y=1.03)
    plt.tight_layout()
    out = FIG_DIR / "fig4_dataset_composition.png"
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {out}")


# ═══════════════════════════════════════════════════════════
# FIGURE 5: Pipeline diagram (text-based)
# ═══════════════════════════════════════════════════════════

def plot_pipeline_diagram():
    """Visual overview of the processing pipeline."""
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.axis('off')

    # Pipeline boxes
    boxes = [
        ("Raw PCM Files\n(stereo, 44100 Hz\n30s recordings)", "#E3F2FD", 0.02),
        ("Left Channel\nExtraction\n+ Normalize", "#BBDEFB", 0.17),
        ("Direct Path\nRemoval\n(first 38 samples)", "#90CAF9", 0.32),
        ("Bandpass Filter\n17.5 – 20.5 kHz\n(Butterworth, 6th)", "#64B5F6", 0.47),
        ("1.5s Segmentation\n66,150 samples\nper chunk", "#42A5F5", 0.62),
        ("Feature\nExtraction\n(6 methods)", "#1E88E5", 0.77),
    ]

    for text, color, x in boxes:
        rect = plt.Rectangle((x, 0.35), 0.13, 0.35, linewidth=2, edgecolor='#1565C0',
                             facecolor=color, zorder=2, transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(x + 0.065, 0.525, text, ha='center', va='center', fontsize=9,
               transform=ax.transAxes, fontweight='bold', zorder=3)

    # Arrows
    for i in range(len(boxes) - 1):
        x_start = boxes[i][2] + 0.13
        x_end = boxes[i+1][2]
        ax.annotate('', xy=(x_end, 0.525), xytext=(x_start, 0.525),
                    arrowprops=dict(arrowstyle='->', color='#1565C0', lw=2),
                    transform=ax.transAxes)

    # Feature extraction branches
    features_top = ["Statistical\n(6 features)", "Wavelet DWT\n(~20 features)"]
    features_bot = ["Mel Spectrogram\n(64×64)", "MFCC\n(40×64)", "CWT Scalogram\n(64×64)", "STFT\n(64×64)"]

    # Top row (flat features → classic ML)
    for i, feat in enumerate(features_top):
        x = 0.15 + i * 0.18
        rect = plt.Rectangle((x, 0.78), 0.14, 0.15, linewidth=1.5, edgecolor='#4CAF50',
                             facecolor='#E8F5E9', zorder=2, transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(x + 0.07, 0.855, feat, ha='center', va='center', fontsize=8,
               transform=ax.transAxes, zorder=3)

    # Classic ML box
    rect = plt.Rectangle((0.55, 0.78), 0.14, 0.15, linewidth=2, edgecolor='#4CAF50',
                         facecolor='#C8E6C9', zorder=2, transform=ax.transAxes)
    ax.add_patch(rect)
    ax.text(0.62, 0.855, "SVM / RF / kNN", ha='center', va='center', fontsize=9,
           transform=ax.transAxes, fontweight='bold', zorder=3)

    # Bottom row (2D features → CNN)
    for i, feat in enumerate(features_bot):
        x = 0.02 + i * 0.16
        rect = plt.Rectangle((x, 0.05), 0.13, 0.15, linewidth=1.5, edgecolor='#E91E63',
                             facecolor='#FCE4EC', zorder=2, transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(x + 0.065, 0.125, feat, ha='center', va='center', fontsize=8,
               transform=ax.transAxes, zorder=3)

    # CNN box
    rect = plt.Rectangle((0.68, 0.05), 0.12, 0.15, linewidth=2, edgecolor='#E91E63',
                         facecolor='#F8BBD0', zorder=2, transform=ax.transAxes)
    ax.add_patch(rect)
    ax.text(0.74, 0.125, "CNN\n(2-layer)", ha='center', va='center', fontsize=9,
           transform=ax.transAxes, fontweight='bold', zorder=3)

    # Results box
    rect = plt.Rectangle((0.84, 0.35), 0.14, 0.35, linewidth=2, edgecolor='#F57F17',
                         facecolor='#FFF9C4', zorder=2, transform=ax.transAxes)
    ax.add_patch(rect)
    ax.text(0.91, 0.525, "Evaluation\n5-Fold CV\n+ 80/20 Split\n\nAccuracy\nPrecision\nRecall\nF1",
           ha='center', va='center', fontsize=8, transform=ax.transAxes, fontweight='bold', zorder=3)

    # Arrows from classic ML and CNN to Results
    ax.annotate('', xy=(0.84, 0.525), xytext=(0.69, 0.855),
                arrowprops=dict(arrowstyle='->', color='#4CAF50', lw=1.5),
                transform=ax.transAxes)
    ax.annotate('', xy=(0.84, 0.525), xytext=(0.80, 0.125),
                arrowprops=dict(arrowstyle='->', color='#E91E63', lw=1.5),
                transform=ax.transAxes)

    ax.set_title('SensEat 1.5s Chirp-Level Classification Pipeline Overview',
                fontsize=15, fontweight='bold', pad=20)

    plt.tight_layout()
    out = FIG_DIR / "fig5_pipeline_overview.png"
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {out}")


# ═══════════════════════════════════════════════════════════
# FIGURE 6: Example spectrogram comparison (Eating vs Idle)
# ═══════════════════════════════════════════════════════════

def plot_example_spectrograms():
    """Show example 1.5s spectrograms: one eating, one idle, side by side."""
    from scipy.signal import butter, lfilter, stft as scipy_stft

    SAMPLE_RATE = 44100
    DIRECT_PATH = int((0.30 / 343.0) * SAMPLE_RATE)
    CHUNK = int(1.5 * SAMPLE_RATE)

    def load_one_chunk(filepath):
        raw = np.fromfile(filepath, dtype=np.int16)
        if raw.size % 2 != 0:
            raw = raw[:-1]
        left = raw.reshape(-1, 2)[:, 0].astype(np.float32) / 32768.0
        left = left[DIRECT_PATH:]
        # Bandpass
        nyq = 0.5 * SAMPLE_RATE
        b, a = butter(6, [17500/nyq, 20500/nyq], btype='band')
        left = lfilter(b, a, left)
        # Take middle chunk (more likely to have activity)
        mid = len(left) // 2
        return left[mid : mid + CHUNK]

    eating_file = None
    idle_file = None

    # Find first eating and idle files
    raw = SCRIPT_DIR / "raw_data"
    for f in sorted((raw / "001").glob("*.pcm")):
        if not f.stem.endswith("_idleTail"):
            eating_file = f
            break
    for f in sorted((raw / "Idle").glob("*.pcm")):
        if not f.stem.endswith("_idleTail"):
            idle_file = f
            break

    if eating_file is None or idle_file is None:
        print("⚠ Could not find example files for spectrogram plot")
        return

    eat_chunk = load_one_chunk(str(eating_file))
    idle_chunk = load_one_chunk(str(idle_file))

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    for col, (chunk, label, fname) in enumerate([
        (eat_chunk, "EATING", eating_file.stem),
        (idle_chunk, "IDLE", idle_file.stem)
    ]):
        # Time domain
        t = np.arange(len(chunk)) / SAMPLE_RATE
        axes[0, col].plot(t, chunk, linewidth=0.3, color='#1565C0' if col == 0 else '#E65100')
        axes[0, col].set_title(f'{label} — Time Domain\n({fname})', fontsize=12, fontweight='bold')
        axes[0, col].set_xlabel('Time (s)')
        axes[0, col].set_ylabel('Amplitude')
        axes[0, col].set_ylim(-0.10, 0.10)
        axes[0, col].grid(alpha=0.3)

        # STFT spectrogram
        f, t_stft, Zxx = scipy_stft(chunk, fs=SAMPLE_RATE, nperseg=512, noverlap=384)
        mag_db = 20 * np.log10(np.abs(Zxx) + 1e-10)
        im = axes[1, col].pcolormesh(t_stft, f, mag_db, shading='gouraud', cmap='inferno')
        axes[1, col].set_title(f'{label} — STFT Spectrogram', fontsize=12, fontweight='bold')
        axes[1, col].set_xlabel('Time (s)')
        axes[1, col].set_ylabel('Frequency (Hz)')
        axes[1, col].set_ylim(17000, 21000)
        plt.colorbar(im, ax=axes[1, col], label='dB')

    plt.suptitle('Example 1.5s Chunks: Eating vs Idle (Chirp Band 17.5–20.5 kHz)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    out = FIG_DIR / "fig6_eating_vs_idle_spectrograms.png"
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {out}")


# ═══════════════════════════════════════════════════════════
# RUN ALL
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("Generating all figures...\n")
    plot_f1_comparison()
    plot_accuracy_with_errorbars()
    plot_metrics_heatmap()
    plot_dataset_composition()
    plot_pipeline_diagram()
    plot_example_spectrograms()
    print(f"\n✅ All figures saved to: {FIG_DIR}/")
