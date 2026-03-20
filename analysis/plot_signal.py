import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
# Using the path suggested by the user
PCM_FILE = SCRIPT_DIR / "raw_data/001/1_001_02_01.pcm"

SAMPLE_RATE = 44100
DURATION_S = 1.5
SAMPLES_TO_LOAD = int(SAMPLE_RATE * DURATION_S)

def load_pcm_left_channel(filepath, samples_to_read=None):
    """Load stereo PCM file, return left channel as float32 in [-1, 1]."""
    if samples_to_read:
        # 16-bit PCM, 2 channels = 4 bytes per frame
        # count is number of items (int16), so for N frames we need 2*N items
        raw = np.fromfile(filepath, dtype=np.int16, count=samples_to_read * 2)
    else:
        raw = np.fromfile(filepath, dtype=np.int16)
        
    if raw.size % 2 != 0:
        raw = raw[:-1]
    stereo = raw.reshape(-1, 2)
    left = stereo[:, 0].astype(np.float32) / 32768.0
    return left

def main():
    if not PCM_FILE.exists():
        print(f"Error: File not found at {PCM_FILE}")
        # Try a more conservative path if the above fails
        # The user mentioned RAW_DATA = SCRIPT_DIR / "raw_data/001/1_001_02_01"
        # which might mean they want 1_001_02_01.pcm inside that?
        # Let's list the directory if it fails to be sure.
        return

    print(f"Loading {DURATION_S}s from {PCM_FILE.name}...")
    signal = load_pcm_left_channel(PCM_FILE, samples_to_read=SAMPLES_TO_LOAD)
    
    # Time axis
    time = np.linspace(0, len(signal) / SAMPLE_RATE, num=len(signal))

    plt.figure(figsize=(15, 5), dpi=100)
    plt.plot(time, signal, color='#1f77b4', linewidth=0.7, alpha=0.8)
    plt.title(f"Raw PCM Signal - {PCM_FILE.name} (First {DURATION_S}s)", fontsize=14, pad=15)
    plt.xlabel("Time (seconds)", fontsize=12)
    plt.ylabel("Amplitude (normalized)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axhline(0, color='black', linewidth=0.8, alpha=0.5)
    
    # Beautify
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    
    output_plot = SCRIPT_DIR / "raw_signal_plot.png"
    plt.savefig(output_plot)
    print(f"✅ Success! Plot saved as: {output_plot}")
    
    # Attempt to show the plot (might not work in all environments, but good to have)
    # plt.show()

if __name__ == "__main__":
    main()
