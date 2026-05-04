# segmentation_servings.py
# Serving-level segmentation using chirp-band (18–20 kHz) activity from STFT.
# - Uses *_idleTail.pcm (if present) to calibrate an "idle" threshold automatically.
# - Otherwise falls back to robust percentile-based thresholding.
# - Outputs: segments.csv + per-file debug plots.

import re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import stft
import matplotlib.pyplot as plt

SR = 44100
CHIRP_FMIN = 18000
CHIRP_FMAX = 20000
DIRECT_PATH_SAMPLES = int((0.30 / 343.0) * SR)  # ~38 at 30 cm

# STFT config (keep consistent with your spectrogram pipeline)
NPERSEG = 1024
NOVERLAP = 512

# Segmentation config
SMOOTH_WIN_FRAMES = 7          # smooth activity curve (odd number recommended)
MIN_INACTIVE_GAP_S = 0.35      # serving boundary requires >= this long inactivity
MIN_ACTIVE_SEG_S = 0.40        # discard tiny active blips shorter than this


IRB_RE = re.compile(r"^(?P<W>\d+)_(?P<XXX>\d{3})_(?P<Y>\d{2})_(?P<ZZ>\d{2})")

def parse_irb(stem: str):
    # stem: filename without extension
    is_idle_tail = stem.endswith("_idleTail")
    base = stem.replace("_idleTail", "") if is_idle_tail else stem
    m = IRB_RE.match(base)
    if not m:
        return None
    d = m.groupdict()
    d["Y"] = int(d["Y"])
    d["ZZ"] = int(d["ZZ"])
    d["is_idleTail"] = is_idle_tail
    d["base_stem"] = base
    return d

def load_pcm_stereo(path: Path, channel="L"):
    raw = np.fromfile(path, dtype=np.int16)
    if raw.size % 2 != 0:
        raw = raw[:-1]
    stereo = raw.reshape(-1, 2).astype(np.float32) / 32768.0
    if channel == "L":
        x = stereo[:, 0]
    elif channel == "R":
        x = stereo[:, 1]
    else:
        x = stereo.mean(axis=1)
    return x

def preprocess(x: np.ndarray):
    # direct-path removal
    if x.size > DIRECT_PATH_SAMPLES:
        x = x[DIRECT_PATH_SAMPLES:]
    return x

def moving_average(x, win):
    if win <= 1:
        return x
    win = int(win)
    if win % 2 == 0:
        win += 1
    kernel = np.ones(win, dtype=np.float32) / win
    return np.convolve(x, kernel, mode="same")

def chirp_band_activity(x: np.ndarray):
    f, t, Z = stft(x, fs=SR, nperseg=NPERSEG, noverlap=NOVERLAP, boundary=None)
    mag2 = (np.abs(Z) ** 2)
    band = (f >= CHIRP_FMIN) & (f <= CHIRP_FMAX)
    A = mag2[band, :].sum(axis=0)
    A = np.log(A + 1e-12)  # stabilize
    return t, A

def estimate_idle_threshold(idle_tail_files):
    """
    Build an idle threshold using idleTail recordings only.
    Returns threshold in log-energy units.
    """
    if not idle_tail_files:
        return None

    all_idle = []
    for p in idle_tail_files:
        x = preprocess(load_pcm_stereo(p, channel="L"))
        if x.size < SR * 0.5:
            continue
        t, A = chirp_band_activity(x)
        A = moving_average(A, SMOOTH_WIN_FRAMES)
        all_idle.append(A)

    if not all_idle:
        return None

    idle_concat = np.concatenate(all_idle)
    # conservative threshold: idle mean + 3*std (tunable)
    thr = float(idle_concat.mean() + 3.0 * idle_concat.std())
    return thr

def segment_servings_from_activity(t, A, thr):
    """
    Returns serving segments as (start_frame, end_frame) frame indices.
    Active when A > thr, inactive when A <= thr.
    Serving boundaries are long inactive gaps (>= MIN_INACTIVE_GAP_S).
    """
    A_s = moving_average(A, SMOOTH_WIN_FRAMES)
    active = A_s > thr

    if len(t) < 2:
        return [], A_s

    dt = float(np.median(np.diff(t)))
    min_gap_frames = max(1, int(MIN_INACTIVE_GAP_S / dt))
    min_active_frames = max(1, int(MIN_ACTIVE_SEG_S / dt))

    segs = []
    i = 0
    start = None

    while i < len(active):
        if active[i] and start is None:
            start = i
            i += 1
            continue

        if start is not None and (not active[i]):
            # count consecutive inactive frames
            j = i
            while j < len(active) and (not active[j]):
                j += 1
            gap_len = j - i

            if gap_len >= min_gap_frames:
                end = i
                if (end - start) >= min_active_frames:
                    segs.append((start, end))
                start = None
            i = j
            continue

        i += 1

    if start is not None:
        end = len(active)
        if (end - start) >= min_active_frames:
            segs.append((start, end))

    return segs, A_s

def frames_to_samples(t, frame_idx):
    return int(t[frame_idx] * SR)

def save_debug_plot(out_png: Path, x, t, A_raw, A_s, thr, segs, title):
    tt = np.arange(len(x)) / SR
    plt.figure(figsize=(14, 7))

    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(tt, x, linewidth=0.35)
    ax1.set_title(f"Waveform: {title}")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")

    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(t, A_s, label="activity (smoothed)", linewidth=1.2)
    ax2.plot(t, A_raw, label="activity (raw)", linewidth=0.6, alpha=0.5)
    ax2.axhline(thr, linestyle="--", color="k", linewidth=1.0, label="threshold")

    for (sf, ef) in segs:
        ax2.axvspan(t[sf], t[min(ef-1, len(t)-1)], alpha=0.22)

    ax2.set_title("Chirp-band activity (18–20 kHz) + detected serving segments")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("log-energy")
    ax2.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def run_participant_folder(participant_folder: str, out_dir: str = "segmentation_out"):
    pf = Path(participant_folder)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    pcm_files = sorted(pf.glob("*.pcm"))
    if not pcm_files:
        raise RuntimeError(f"No .pcm files found in: {pf}")

    # split into main vs idleTail
    idle_tail_files = []
    main_files = []

    for p in pcm_files:
        meta = parse_irb(p.stem)
        if meta is None:
            continue
        if meta["is_idleTail"]:
            idle_tail_files.append(p)
        else:
            main_files.append(p)

    # threshold calibration from idle tails (preferred)
    thr_idle = estimate_idle_threshold(idle_tail_files)

    rows = []

    for p in main_files:
        meta = parse_irb(p.stem)
        x = preprocess(load_pcm_stereo(p, channel="L"))

        t, A = chirp_band_activity(x)

        # If no idleTail calibration exists, use robust per-file threshold
        if thr_idle is None:
            A_s_tmp = moving_average(A, SMOOTH_WIN_FRAMES)
            # threshold = median + k*MAD (robust)
            med = np.median(A_s_tmp)
            mad = np.median(np.abs(A_s_tmp - med)) + 1e-12
            thr = float(med + 4.0 * mad)
        else:
            thr = thr_idle

        segs, A_s = segment_servings_from_activity(t, A, thr)

        # save debug plot per file
        debug_png = out / f"{p.stem}_serving_segments.png"
        save_debug_plot(debug_png, x, t, A, A_s, thr, segs, title=p.name)

        # store segments
        for si, (sf, ef) in enumerate(segs):
            s_samp = frames_to_samples(t, sf)
            e_samp = frames_to_samples(t, min(ef, len(t)-1))

            rows.append({
                "file": p.name,
                "W": meta["W"],
                "participant": meta["XXX"],
                "food_code_Y": meta["Y"],
                "trial_ZZ": meta["ZZ"],
                "serving_segment_idx": si,
                "start_frame": sf,
                "end_frame": ef,
                "start_time_s": float(t[sf]),
                "end_time_s": float(t[min(ef-1, len(t)-1)]),
                "start_sample": int(s_samp),
                "end_sample": int(e_samp),
                "threshold_used": float(thr),
                "used_idleTail_calibration": bool(thr_idle is not None),
                "debug_plot": str(debug_png.name)
            })

    df = pd.DataFrame(rows)
    out_csv = out / "segments_servings.csv"
    df.to_csv(out_csv, index=False)
    return df, out_csv

if __name__ == "__main__":
    # Example:
    # python segmentation_servings.py "/path/to/participant_001_folder"
    import sys
    if len(sys.argv) < 2:
        print("Usage: python segmentation_servings.py <participant_folder> [out_dir]")
        sys.exit(1)

    participant_folder = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "segmentation_out"
    df, csv_path = run_participant_folder(participant_folder, out_dir=out_dir)
    print("Saved:", csv_path)
    print("Segments rows:", len(df))