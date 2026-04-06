import os
import re
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram as scipy_spectrogram

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

# ───────────────── CONFIG ─────────────────
SAMPLE_RATE = 44100
CHIRP_DUR_MS = 1000
GAP_DUR_MS = 500
SOUND_SPEED = 343.0

CHIRP_SAMPLES = (CHIRP_DUR_MS * SAMPLE_RATE) // 1000
GAP_SAMPLES = (GAP_DUR_MS * SAMPLE_RATE) // 1000
PERIOD_SAMPLES = CHIRP_SAMPLES + GAP_SAMPLES
PERIOD_SEC = PERIOD_SAMPLES / SAMPLE_RATE

TARGET_DIST_M = (0.10, 0.50)
TARGET_TAP_MIN = int(2 * TARGET_DIST_M[0] / SOUND_SPEED * SAMPLE_RATE)
TARGET_TAP_MAX = int(2 * TARGET_DIST_M[1] / SOUND_SPEED * SAMPLE_RATE)

SAR_WINDOW = 15

SCRIPT_DIR = Path(__file__).resolve().parent
RAW_DATA = SCRIPT_DIR / "sar_raw_data"
FIG_DIR = SCRIPT_DIR / "figures"
PER_CHIRP_DIR = FIG_DIR / "per_chirp"
SAR_DIR = FIG_DIR / "sar"
PER_CHIRP_DIR.mkdir(parents=True, exist_ok=True)
SAR_DIR.mkdir(parents=True, exist_ok=True)

REFERENCE_PCM = SCRIPT_DIR / "raw_data" / "transmitted_chirp" / "chirp_reference_set2 (2).pcm"

# ───────────── IRB PARSING ─────────────
IRB_RE = re.compile(r"^(\d+)_(\d{3})_(\d{2})_(\d{2})")

FOOD_NAMES = {
    0: "Idle",
    1: "Tortilla", 2: "Mandarin", 3: "Chicken_Breast",
    4: "Cheeze_It", 5: "Carrots", 6: "Chocolate",
    7: "Yogurt", 8: "Noodles", 9: "Water", 10: "Coke",
}

def parse_irb_filename(path):
    m = IRB_RE.match(path.name)
    if not m:
        return None
    food_code = int(m.group(3))
    return {
        "path": path,
        "food_code": food_code,
        "food_name": FOOD_NAMES.get(food_code, "Unknown"),
        "is_idle": food_code == 0,
        "is_eating": food_code != 0,
    }

def discover_files(folder):
    return [parse_irb_filename(f) for f in folder.glob("*.pcm") if parse_irb_filename(f)]

# ───────────── LOAD ─────────────
def load_reference_chirp():
    raw = np.fromfile(REFERENCE_PCM, dtype=np.int16).astype(np.float64) / 32768.0
    return raw[:CHIRP_SAMPLES]

def load_pcm_left(path):
    raw = np.fromfile(path, dtype=np.int16)
    raw = raw[:len(raw)//2*2]
    return raw.reshape(-1,2)[:,0].astype(np.float64) / 32768.0

# ───────────── SEGMENT ─────────────
def segment_chirps(signal):
    n = len(signal) // PERIOD_SAMPLES
    return [signal[i*PERIOD_SAMPLES : i*PERIOD_SAMPLES + CHIRP_SAMPLES] for i in range(n)]

# ───────────── LEAST SQUARES (AIM CORE) ─────────────
def estimate_channel_ls(y, x, L=200):
    N = len(y)
    X = np.zeros((N, L))
    for i in range(L):
        X[i:, i] = x[:N-i]
    h, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return h

def reconstruct_target_signal(h, x):
    y_clean = np.zeros_like(x)
    for i in range(len(h)):
        if TARGET_TAP_MIN <= i <= TARGET_TAP_MAX:
            y_clean += h[i] * np.roll(x, i)
    return y_clean

# ───────────── TAP PROFILE ─────────────
def compute_tap_profile(rx, ref):
    Nfft = 1 << (2*len(rx)-1).bit_length()
    RX = np.fft.fft(rx, n=Nfft)
    REF = np.fft.fft(ref, n=Nfft)
    taps = np.fft.ifft(RX * np.conj(REF))
    return taps[:CHIRP_SAMPLES]

def taps_to_dist(n):
    return np.arange(n) / SAMPLE_RATE * SOUND_SPEED / 2

# ───────────── PLOT PER CHIRP ─────────────
def plot_chirp(i, chirp, ref, label):
    taps = compute_tap_profile(chirp, ref)
    dist = taps_to_dist(len(taps))

    f,t,S = scipy_spectrogram(chirp, fs=SAMPLE_RATE, nperseg=512, noverlap=256)

    fig,ax = plt.subplots(1,3,figsize=(15,4))
    fig.suptitle(f"{label} Chirp {i}")

    ax[0].plot(chirp)
    ax[0].set_title("Time")

    ax[1].pcolormesh(t,f,10*np.log10(S+1e-12))
    ax[1].set_title("Spectrogram")

    ax[2].plot(dist,np.abs(taps))
    ax[2].set_title("Tap Profile")

    plt.savefig(PER_CHIRP_DIR/f"{label}_{i}.png")
    plt.close()

# ───────────── BUILD S(n,k) ─────────────
def build_matrix(all_taps):
    z0,z1 = TARGET_TAP_MIN,TARGET_TAP_MAX
    return np.array([t[z0:z1] for t in all_taps])

# ───────────── SAR PROCESS ─────────────
def process_sar(S):
    results=[]
    for i in range(len(S)-SAR_WINDOW):
        win = S[i:i+SAR_WINDOW]

        win = win - np.mean(win,axis=0,keepdims=True)

        fft = np.fft.fft(win,axis=0)

        template = np.mean(fft,axis=1,keepdims=True)
        template /= (np.abs(template)+1e-12)

        filtered = fft * np.conj(template)

        results.append((i,filtered))
    return results

# ───────────── STACK PLOT ─────────────
def plot_stack(S,label):
    plt.imshow(np.abs(S),aspect='auto',origin='lower')
    plt.title(label)
    plt.savefig(SAR_DIR/f"stack_{label}.png")
    plt.close()

# ───────────── MAIN ─────────────
def main():
    files = discover_files(RAW_DATA)

    idle = [f for f in files if f["is_idle"]][0]
    eat = [f for f in files if f["is_eating"]][0]

    ref = load_reference_chirp()

    for info,label in [(eat,"EATING"),(idle,"IDLE")]:
        signal = load_pcm_left(info["path"])
        chirps = segment_chirps(signal)

        all_taps=[]

        for i,c in enumerate(chirps):
            # 🔥 LEAST SQUARES CLEANING
            h = estimate_channel_ls(c,ref)
            clean = reconstruct_target_signal(h,ref)

            taps = compute_tap_profile(clean,ref)
            all_taps.append(taps)

            plot_chirp(i,clean,ref,label)

        S = build_matrix(all_taps)

        plot_stack(S,label)

        sar = process_sar(S)

        print(f"{label} done with {len(chirps)} chirps")

if __name__ == "__main__":
    main()