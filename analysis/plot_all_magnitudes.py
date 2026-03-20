import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_raw_pcm_left(filepath):
    raw = np.fromfile(filepath, dtype=np.int16)
    if raw.size % 2 != 0:
        raw = raw[:-1]
    stereo = raw.reshape(-1, 2)
    return stereo[:, 0].astype(np.float32) / 32768.0

def generate_multi_plot():
    sdsu_dir = Path("raw_data/Participant-1-02232026")
    
    # We will pick one representative file from each class
    files_to_plot = {
        "Coke (SDSU)": "chirp_coke_30cm_120deg_s01_set1_1771919629041.pcm",
        "Cracker (SDSU)": "chirp_cracker_30cm_120deg_s01_set1_1771918978420.pcm",
        "Orange (SDSU)": "chirp_orange_30cm_120deg_s01_set1_1771919279905.pcm",
        "Tortillas (SDSU)": "chirp_tortillas_30cm_120deg_s01_set1_1771918405171.pcm",
        "Water (SDSU)": "chirp_water_30cm_120deg_s01_set1_1771918656367.pcm",
        "Idle (SDSU)": "chirp_idle_30cm_120deg_s01_set1_1771919886424.pcm"
    }

    sr = 44100
    sec = 2 # Plot first 2 seconds only
    t = np.linspace(0, sec, sr * sec)

    plt.figure(figsize=(16, 18))
    
    for i, (label, fname) in enumerate(files_to_plot.items(), 1):
        fpath = sdsu_dir / fname
        if not fpath.exists():
            print(f"Skipping {label}, file not found")
            continue
            
        sig = load_raw_pcm_left(fpath)
        
        plt.subplot(6, 1, i)
        # Use orange for idle, blue for foods
        color = 'orange' if 'Idle' in label else 'blue'
        plt.plot(t, sig[:sr*sec], color=color, alpha=0.7)
        plt.title(f"Raw Magnitude Profile - {label}", fontsize=14)
        plt.ylabel("Amplitude", fontsize=12)
        plt.ylim(-0.3, 0.3)
        plt.grid(True, alpha=0.3)
        
        if i == 6:
            plt.xlabel("Time (s)", fontsize=12)

    plt.tight_layout(pad=2.0)
    plt.subplots_adjust(bottom=0.08, hspace=0.3)
    
    out_path = Path("figures") / "sdsu_all_classes_magnitude.png"
    out_path.parent.mkdir(exist_ok=True)
    plt.savefig(out_path, dpi=300)
    print(f"Saved massive multi-plot to {out_path}")

if __name__ == "__main__":
    generate_multi_plot()
