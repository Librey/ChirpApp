import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
RAW_DATA = SCRIPT_DIR / "raw_data"

def load_raw_pcm_left(filepath):
    """Load stereo PCM file, return left channel as float32 in [-1, 1]."""
    raw = np.fromfile(filepath, dtype=np.int16)
    if raw.size % 2 != 0:
        raw = raw[:-1]
    stereo = raw.reshape(-1, 2)
    left = stereo[:, 0].astype(np.float32) / 32768.0
    return left

def analyze_magnitude():
    # Pick one eating file and one idle file from the SDSU dataset
    eating_file = RAW_DATA / "Participant-1-02232026" / "chirp_cracker_30cm_120deg_s01_set1_1771918978420.pcm"
    idle_file = RAW_DATA / "Participant-1-02232026" / "chirp_idle_30cm_120deg_s01_set1_1771919886424.pcm"

    if not eating_file.exists() or not idle_file.exists():
        print("Error: Could not find the specified PCM files.")
        return

    print("Loading raw PCM files...")
    eat_sig = load_raw_pcm_left(eating_file)
    idle_sig = load_raw_pcm_left(idle_file)

    # Calculate overall stats
    print("\n--- Raw Amplitude Stats ---")
    print(f"Eating (SDSU - Cracker): Max={np.max(np.abs(eat_sig)):.5f}, Mean={np.mean(np.abs(eat_sig)):.5f}, RMS={np.sqrt(np.mean(eat_sig**2)):.5f}")
    print(f"Idle   (SDSU)          : Max={np.max(np.abs(idle_sig)):.5f}, Mean={np.mean(np.abs(idle_sig)):.5f}, RMS={np.sqrt(np.mean(idle_sig**2)):.5f}")

    # Plot the first 2 seconds (to see the overall envelope)
    sr = 44100
    sec = 2
    t = np.linspace(0, sec, sr * sec)

    plt.figure(figsize=(15, 6))

    plt.subplot(2, 1, 1)
    plt.plot(t, eat_sig[:sr*sec], color='blue', alpha=0.7)
    plt.title(f"SDSU Eating (Cracker) - First {sec}s")
    plt.ylabel("Amplitude")
    plt.ylim(-0.3, 0.3)
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 1, 2)
    plt.plot(t, idle_sig[:sr*sec], color='orange', alpha=0.7)
    plt.title(f"SDSU Idle - First {sec}s")
    plt.ylabel("Amplitude")
    plt.xlabel("Time (s)")
    plt.ylim(-0.3, 0.3)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = SCRIPT_DIR / "figures" / "sdsu_magnitude_investigation.png"
    
    # Ensure dir exists
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    print(f"\nSaved magnitude plot to: {out_path}")

if __name__ == "__main__":
    analyze_magnitude()
