import re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import stft
import matplotlib.pyplot as plt

SR = 44100
CHIRP_FMIN, CHIRP_FMAX = 18000, 20000
DIRECT_PATH_SAMPLES = int((0.30 / 343.0) * SR)  # ~38
NPERSEG, NOVERLAP = 1024, 512

# Serving-level params
SERVING_WIN_S = 4.0
MIN_GAP_S = 0.7
MIN_SERVING_S = 1.5

IRB_RE = re.compile(r"^(?P<W>\d+)_(?P<XXX>\d{3})_(?P<Y>\d{2})_(?P<ZZ>\d{2})")

def parse_irb(stem: str):
    is_idle_tail = stem.endswith("_idleTail")
    base = stem.replace("_idleTail", "") if is_idle_tail else stem
    m = IRB_RE.match(base)
    if not m:
        return None
    d = m.groupdict()
    d["Y"] = int(d["Y"])
    d["ZZ"] = int(d["ZZ"])
    d["is_idleTail"] = is_idle_tail
    return d, base

def load_pcm_stereo(path: Path, channel="L"):
    raw = np.fromfile(path, dtype=np.int16)
    if raw.size % 2 != 0:
        raw = raw[:-1]
    stereo = raw.reshape(-1, 2).astype(np.float32) / 32768.0
    if channel == "L":
        return stereo[:, 0]
    elif channel == "R":
        return stereo[:, 1]
    return stereo.mean(axis=1)

def preprocess(x):
    return x[DIRECT_PATH_SAMPLES:] if x.size > DIRECT_PATH_SAMPLES else x

def chirp_band_activity(x):
    f, t, Z = stft(x, fs=SR, nperseg=NPERSEG, noverlap=NOVERLAP, boundary=None)
    mag2 = (np.abs(Z) ** 2)
    band = (f >= CHIRP_FMIN) & (f <= CHIRP_FMAX)
    A = mag2[band, :].sum(axis=0)
    return t, np.log(A + 1e-12)

def rolling_mean(a, win_frames):
    """Smooth A(t) — high during eating, low during genuine idle gaps."""
    pad = win_frames // 2
    ap = np.pad(a, (pad, pad), mode="edge")
    out = np.empty_like(a)
    for i in range(len(a)):
        out[i] = ap[i:i+win_frames].mean()
    return out

# ✅ FIX 1: Added 'name' parameter instead of using pcm.name directly
def segment_servings(t, A, name=""):
    dt = float(np.median(np.diff(t))) if len(t) > 2 else 0.01
    win_frames = max(5, int(SERVING_WIN_S / dt))
    if win_frames % 2 == 0:
        win_frames += 1

    B = rolling_mean(A, win_frames)      # smoothed log-energy
    thr = float(np.percentile(B, 40))   # 40th percentile: below = inactive

    # ✅ FIX 2: Use 'name' instead of 'pcm.name'
    print(name, "B min/median/max:",
          float(B.min()), float(np.median(B)), float(B.max()),
          "thr:", thr)

    active = B > thr

    min_gap_frames = max(1, int(MIN_GAP_S / dt))
    min_serv_frames = max(1, int(MIN_SERVING_S / dt))

    segs = []
    start = None
    i = 0
    while i < len(active):
        if active[i] and start is None:
            start = i
            i += 1
            continue
        if start is not None and (not active[i]):
            j = i
            while j < len(active) and (not active[j]):
                j += 1
            if (j - i) >= min_gap_frames:
                end = i
                if (end - start) >= min_serv_frames:
                    segs.append((start, end))
                start = None
            i = j
            continue
        i += 1

    if start is not None:
        end = len(active)
        if (end - start) >= min_serv_frames:
            segs.append((start, end))

    return segs, B, thr


def save_plot(out_png, t, A, B, thr, segs, title):
    plt.figure(figsize=(14, 6))
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(t, A, linewidth=1.0)
    ax1.set_title(f"Chirp-band log-energy A(t): {title}")
    ax1.set_xlabel("Time (s)"); ax1.set_ylabel("log-energy")

    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(t, B, linewidth=1.2, label="serving-activity B(t)=rolling std")
    ax2.axhline(thr, linestyle="--", color="k", label="threshold")
    for s, e in segs:
        ax2.axvspan(t[s], t[min(e - 1, len(t) - 1)], alpha=0.22)
    ax2.set_title("Serving-level segmentation — smoothed mean log-energy")
    ax2.set_xlabel("Time (s)"); ax2.set_ylabel("B(t)")
    ax2.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def run(folder, out_dir="serving_seg_out"):
    pf = Path(folder)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    rows = []
    for pcm in sorted(pf.glob("*.pcm")):
        meta = parse_irb(pcm.stem)
        if meta is None:
            continue
        (d, base) = meta
        if d["is_idleTail"]:
            continue

        x = preprocess(load_pcm_stereo(pcm, "L"))
        t, A = chirp_band_activity(x)

        # ✅ FIX 3: Pass pcm.name into segment_servings
        segs, B, thr = segment_servings(t, A, name=pcm.name)

        plot_path = out / f"{pcm.stem}_SERVING.png"
        save_plot(plot_path, t, A, B, thr, segs, pcm.name)

        for si, (sf, ef) in enumerate(segs):
            rows.append({
                "file": pcm.name,
                "participant": d["XXX"],
                "food_code_Y": d["Y"],
                "trial_ZZ": d["ZZ"],
                "serving_idx": si,
                "start_time_s": float(t[sf]),
                "end_time_s": float(t[min(ef - 1, len(t) - 1)]),
                "start_frame": int(sf),
                "end_frame": int(ef),
                "thr_B": float(thr),
                "plot": plot_path.name
            })

    df = pd.DataFrame(rows)
    out_csv = Path(out_dir) / "segments_serving_level.csv"
    df.to_csv(out_csv, index=False)
    print("Saved:", out_csv, "rows:", len(df))
    return out_csv, len(df)


if __name__ == "__main__":
    import sys
    folder = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "serving_seg_out"
    csv_path, n = run(folder, out_dir)