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

# Target mouth-distance window
TARGET_DIST_M = (0.10, 0.50)
TARGET_TAP_MIN = int(2 * TARGET_DIST_M[0] / SOUND_SPEED * SAMPLE_RATE)
TARGET_TAP_MAX = int(2 * TARGET_DIST_M[1] / SOUND_SPEED * SAMPLE_RATE)

# Multi-chirp SAR window
SAR_WINDOW = 10
SAR_STRIDE = 1

SCRIPT_DIR = Path(__file__).resolve().parent
RAW_DATA = SCRIPT_DIR / "sar_raw_data"
FIG_DIR = SCRIPT_DIR / "figures/sar"
PER_CHIRP_DIR = FIG_DIR / "per_chirp"
SAR_DIR = FIG_DIR / "sar_windows"
PER_CHIRP_DIR.mkdir(parents=True, exist_ok=True)
SAR_DIR.mkdir(parents=True, exist_ok=True)

REFERENCE_PCM = SCRIPT_DIR / "raw_data" / "transmitted_chirp" / "chirp_reference_set2 (2).pcm"

# ───────────── IRB PARSING ─────────────
IRB_RE = re.compile(r"^(\d+)_(\d{3})_(\d{2})_(\d{2})(?:\s*\(\d+\))?\.pcm$")

FOOD_NAMES = {
    0: "Idle",
    1: "Tortilla",
    2: "Mandarin",
    3: "Chicken_Breast",
    4: "Cheeze_It",
    5: "Carrots",
    6: "Chocolate",
    7: "Yogurt",
    8: "Noodles",
    9: "Water",
    10: "Coke",
}


def parse_irb_filename(path: Path):
    m = IRB_RE.match(path.name)
    if not m:
        return None
    food_code = int(m.group(3))
    return {
        "path": path,
        "food_code": food_code,
        "food_name": FOOD_NAMES.get(food_code, f"Unknown_{food_code}"),
        "is_idle": food_code == 0,
        "is_eating": food_code != 0,
    }


def discover_files(folder: Path):
    parsed = []
    for f in sorted(folder.glob("*.pcm")):
        info = parse_irb_filename(f)
        if info:
            parsed.append(info)
    return parsed


# ───────────── LOAD ─────────────
def load_reference_chirp() -> np.ndarray:
    if not REFERENCE_PCM.exists():
        raise FileNotFoundError(f"Reference chirp not found: {REFERENCE_PCM}")
    raw = np.fromfile(REFERENCE_PCM, dtype=np.int16).astype(np.float64) / 32768.0
    if len(raw) < CHIRP_SAMPLES:
        raise ValueError(
            f"Reference chirp too short: expected at least {CHIRP_SAMPLES}, got {len(raw)}"
        )
    return raw[:CHIRP_SAMPLES]


def load_pcm_left(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.int16)
    raw = raw[: len(raw) // 2 * 2]
    stereo = raw.reshape(-1, 2)
    return stereo[:, 0].astype(np.float64) / 32768.0


# ───────────── SEGMENTATION ─────────────
def segment_chirps(signal: np.ndarray):
    """
    Split recording into chirp-only sections.
    Each cycle is [1.0 s chirp][0.5 s gap].
    """
    n_periods = len(signal) // PERIOD_SAMPLES
    chirps = []
    for i in range(n_periods):
        start = i * PERIOD_SAMPLES
        chirp = signal[start : start + CHIRP_SAMPLES]
        if len(chirp) == CHIRP_SAMPLES:
            chirps.append(chirp)
    return chirps


# ───────────── AIM STAGE-1 STYLE DIRECT-PATH REMOVAL ─────────────
def remove_direct_path_ls(rx_chirp: np.ndarray, ref_chirp: np.ndarray) -> np.ndarray:
    """
    Least-squares scaling:
        c = argmin || rx - c * ref ||
    Then subtract c * ref from rx.
    """
    ref = ref_chirp[: len(rx_chirp)]
    denom = np.dot(ref, ref) + 1e-12
    c = np.dot(rx_chirp, ref) / denom
    return rx_chirp - c * ref


# ───────────── TAP / RANGE PROFILE ─────────────
def compute_tap_profile(rx_chirp: np.ndarray, ref_chirp: np.ndarray) -> np.ndarray:
    """
    FFT-based cross-correlation in delay domain.
    Positive lag corresponds to round-trip delay.
    """
    nfft = 1 << (2 * len(rx_chirp) - 1).bit_length()
    RX = np.fft.fft(rx_chirp, n=nfft)
    REF = np.fft.fft(ref_chirp, n=nfft)
    taps = np.fft.ifft(RX * np.conj(REF))
    return taps[:CHIRP_SAMPLES]


def taps_to_distance_m(num_taps: int) -> np.ndarray:
    tap_idx = np.arange(num_taps)
    return tap_idx / SAMPLE_RATE * SOUND_SPEED / 2.0


# ───────────── PER-CHIRP VISUALIZATION ─────────────
def plot_and_save_chirp(
    chirp_idx: int,
    rx_chirp: np.ndarray,
    taps: np.ndarray,
    label: str,
    save_dir: Path,
):
    tap_mag = np.abs(taps)
    tap_phase = np.angle(taps)
    dist_m = taps_to_distance_m(len(taps))
    t_ms = np.arange(len(rx_chirp)) / SAMPLE_RATE * 1000.0

    f_s, t_s, Sxx = scipy_spectrogram(
        rx_chirp,
        fs=SAMPLE_RATE,
        nperseg=512,
        noverlap=256,
        mode="magnitude",
    )

    z0, z1 = TARGET_TAP_MIN, min(TARGET_TAP_MAX, len(taps))

    fig, axes = plt.subplots(1, 5, figsize=(24, 4))
    fig.suptitle(f"{label} | Chirp {chirp_idx:03d}", fontsize=11, fontweight="bold")

    axes[0].plot(t_ms, rx_chirp, linewidth=0.5)
    axes[0].set_title("Time Domain")
    axes[0].set_xlabel("Time (ms)")
    axes[0].set_ylabel("Amplitude")

    axes[1].pcolormesh(
        t_s * 1000,
        f_s / 1000,
        20 * np.log10(Sxx + 1e-12),
        shading="gouraud",
        cmap="inferno",
    )
    axes[1].set_title("Spectrogram")
    axes[1].set_xlabel("Time (ms)")
    axes[1].set_ylabel("Frequency (kHz)")

    axes[2].plot(dist_m, tap_mag, linewidth=0.7)
    axes[2].axvspan(
        TARGET_DIST_M[0],
        TARGET_DIST_M[1],
        alpha=0.2,
        color="green",
        label="Target window",
    )
    axes[2].set_title("Tap Profile")
    axes[2].set_xlabel("Distance (m)")
    axes[2].set_ylabel("|Amplitude|")
    axes[2].legend(fontsize=7)

    axes[3].plot(dist_m[z0:z1], tap_mag[z0:z1], linewidth=0.9)
    axes[3].set_title("Target Window (mag)")
    axes[3].set_xlabel("Distance (m)")
    axes[3].set_ylabel("|Amplitude|")

    axes[4].plot(dist_m[z0:z1], tap_phase[z0:z1], linewidth=0.9)
    axes[4].set_title("Target Window (phase)")
    axes[4].set_xlabel("Distance (m)")
    axes[4].set_ylabel("Phase (rad)")
    axes[4].set_ylim(-np.pi, np.pi)

    plt.tight_layout()
    out = save_dir / f"{label}_chirp_{chirp_idx:03d}.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ───────────── BUILD S(n,k) ─────────────
def build_sar_matrix_from_taps(all_taps: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    z0 = TARGET_TAP_MIN
    z1 = min(TARGET_TAP_MAX, len(all_taps[0]))
    S = np.array([t[z0:z1] for t in all_taps], dtype=np.complex128)
    dist_m = taps_to_distance_m(z1)[z0:z1]
    return S, dist_m


# ───────────── SAR CORE ─────────────
def process_sar_windows(S: np.ndarray):
    """
    Apply SAR on multi-chirp windows.
    """
    if S.shape[0] < SAR_WINDOW:
        raise ValueError(f"Need at least {SAR_WINDOW} chirps, got {S.shape[0]}")

    results = []
    for start in range(0, S.shape[0] - SAR_WINDOW + 1, SAR_STRIDE):
        end = start + SAR_WINDOW
        S_win = S[start:end].copy()

        # remove residual static/common component inside the window
        S_centered = S_win - np.mean(S_win, axis=0, keepdims=True)

        # FFT across chirp dimension n for each range bin k
        col_fft = np.fft.fft(S_centered, axis=0)

        # simple matched filter
        template = np.mean(col_fft, axis=1, keepdims=True)
        template = template / (np.abs(template) + 1e-12)
        filtered = col_fft * np.conj(template)

        results.append({
            "start": start,
            "end": end,
            "S_centered": S_centered,
            "col_fft": col_fft,
            "filtered": filtered,
        })
    return results


# ───────────── PLOTS ─────────────
def plot_chirp_stack(S: np.ndarray, dist_m: np.ndarray, label: str, save_dir: Path):
    chirp_times = np.arange(S.shape[0]) * PERIOD_SEC
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.pcolormesh(
        dist_m,
        chirp_times,
        np.abs(S),
        shading="gouraud",
        cmap="inferno",
    )
    plt.colorbar(im, ax=ax, label="|Amplitude|")
    ax.set_title(f"{label} — Multi-Chirp Stack S(n,k)")
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Time (s)")
    plt.tight_layout()
    plt.savefig(save_dir / f"stack_{label}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_sar_window_result(result: dict, dist_m: np.ndarray, label: str, save_dir: Path):
    start = result["start"]
    end = result["end"]

    S_mag = np.abs(result["S_centered"])
    fft_mag = np.abs(result["col_fft"])
    filt_mag = np.abs(result["filtered"])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"{label} — SAR window chirps {start}..{end-1}", fontsize=11, fontweight="bold")

    im0 = axes[0].imshow(
        20 * np.log10(S_mag + 1e-12),
        aspect="auto",
        origin="lower",
        cmap="plasma",
        extent=[dist_m[0], dist_m[-1], start, end - 1],
    )
    axes[0].set_title("Windowed S(n,k)")
    axes[0].set_xlabel("Distance (m)")
    axes[0].set_ylabel("Chirp index")
    plt.colorbar(im0, ax=axes[0], label="dB")

    im1 = axes[1].imshow(
        20 * np.log10(fft_mag + 1e-12),
        aspect="auto",
        origin="lower",
        cmap="viridis",
        extent=[dist_m[0], dist_m[-1], 0, fft_mag.shape[0] - 1],
    )
    axes[1].set_title("Column FFT")
    axes[1].set_xlabel("Distance (m)")
    axes[1].set_ylabel("FFT bin across chirps")
    plt.colorbar(im1, ax=axes[1], label="dB")

    im2 = axes[2].imshow(
        20 * np.log10(filt_mag + 1e-12),
        aspect="auto",
        origin="lower",
        cmap="inferno",
        extent=[dist_m[0], dist_m[-1], 0, filt_mag.shape[0] - 1],
    )
    axes[2].set_title("Matched Filter Output")
    axes[2].set_xlabel("Distance (m)")
    axes[2].set_ylabel("FFT bin across chirps")
    plt.colorbar(im2, ax=axes[2], label="dB")

    plt.tight_layout()
    out = save_dir / f"{label}_sar_window_{start:03d}_{end-1:03d}.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ───────────── PREPARE CHIRP TAPS ─────────────
def prepare_clean_taps(signal: np.ndarray, ref: np.ndarray, label: str):
    chirps = segment_chirps(signal)
    all_taps = []

    print(f"\n{label}: {len(chirps)} chirps found")
    for i, chirp in enumerate(chirps):
        chirp_clean = remove_direct_path_ls(chirp, ref)
        taps = compute_tap_profile(chirp_clean, ref)
        all_taps.append(taps)
        plot_and_save_chirp(i, chirp_clean, taps, label, PER_CHIRP_DIR)

        if (i + 1) % 5 == 0 or (i + 1) == len(chirps):
            print(f"  saved {i+1}/{len(chirps)} per-chirp figures")

    return all_taps


# ───────────── MAIN ─────────────
def main():
    files = discover_files(RAW_DATA)
    idle_files = [f for f in files if f["is_idle"]]
    eating_files = [f for f in files if f["is_eating"]]

    if not idle_files or not eating_files:
        raise ValueError("Need at least one idle and one eating file in sar_raw_data/")

    idle_file = idle_files[0]
    eat_file = eating_files[0]

    print(f"Eating : {eat_file['path'].name} ({eat_file['food_name']})")
    print(f"Idle   : {idle_file['path'].name}")
    print(
        f"Target distance window: {TARGET_DIST_M[0]}–{TARGET_DIST_M[1]} m "
        f"(taps {TARGET_TAP_MIN}–{TARGET_TAP_MAX})"
    )

    ref = load_reference_chirp()

    # 1. Prepare idle taps first (baseline)
    idle_signal = load_pcm_left(idle_file["path"])
    idle_taps = prepare_clean_taps(idle_signal, ref, "IDLE")

    # 2. Prepare eating taps
    eat_label = f"EATING_{eat_file['food_name']}"
    eat_signal = load_pcm_left(eat_file["path"])
    eat_taps = prepare_clean_taps(eat_signal, ref, eat_label)

    # 3. Build S(n,k)
    S_idle, dist_m = build_sar_matrix_from_taps(idle_taps)
    S_eat, _ = build_sar_matrix_from_taps(eat_taps)

    # 4. AIM-style baseline subtraction:
    #    use idle as static background estimate
    background = np.mean(S_idle, axis=0, keepdims=True)
    S_idle_clean = S_idle - background
    S_eat_clean = S_eat - background

    # 5. Plot stacks
    plot_chirp_stack(S_idle_clean, dist_m, "IDLE", SAR_DIR)
    plot_chirp_stack(S_eat_clean, dist_m, eat_label, SAR_DIR)
    print(f"  saved stack_IDLE.png")
    print(f"  saved stack_{eat_label}.png")

    # 6. SAR windows
    idle_results = process_sar_windows(S_idle_clean)
    eat_results = process_sar_windows(S_eat_clean)

    for res in idle_results:
        plot_sar_window_result(res, dist_m, "IDLE", SAR_DIR)
    for res in eat_results:
        plot_sar_window_result(res, dist_m, eat_label, SAR_DIR)

    print(f"  saved {len(idle_results)} IDLE SAR window figures")
    print(f"  saved {len(eat_results)} {eat_label} SAR window figures")

    print(f"\nPer-chirp figures saved in:\n  {PER_CHIRP_DIR}")
    print(f"SAR window figures saved in:\n  {SAR_DIR}")


if __name__ == "__main__":
    main()