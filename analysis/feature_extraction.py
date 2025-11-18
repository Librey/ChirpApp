import numpy as np
import scipy.stats as stats
from scipy.signal import stft
import librosa
import librosa.display
import pywt
import matplotlib.pyplot as plt

# ---------------------------
# Load PCM helper
# ---------------------------
def load_pcm(filepath, samplerate=48000, dtype=np.int16):
    """Load raw PCM file as numpy array."""
    data = np.fromfile(filepath, dtype=dtype)
    # Normalize to float (-1 to 1)
    data = data.astype(np.float32) / np.iinfo(dtype).max
    return data, samplerate

# ---------------------------
# Statistical Features
# ---------------------------

def extract_time_domain_features(signal):
    """Extract statistical features from time-domain signal s_t."""
    features = {}
    
    features["mean"] = np.mean(signal)
    features["std"] = np.std(signal)
    features["var"] = np.var(signal)
    features["rms"] = np.sqrt(np.mean(signal**2))
    features["max_amp"] = np.max(signal)
    features["min_amp"] = np.min(signal)
    features["peak_to_peak"] = np.ptp(signal)
    features["skewness"] = stats.skew(signal)
    features["kurtosis"] = stats.kurtosis(signal)
    features["zcr"] = ((signal[:-1] * signal[1:]) < 0).sum() / len(signal)
    
    # Shannon entropy
    hist, _ = np.histogram(signal, bins=50, density=True)
    hist = hist[hist > 0]
    features["entropy"] = -np.sum(hist * np.log2(hist))

    return features


def extract_frequency_domain_features(signal, samplerate):
    """Extract statistical features from frequency-domain signal s_f."""
    
    # Compute FFT
    fft_vals = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), 1/samplerate)

    features = {}

    # Basic FFT stats
    features["fft_mean"] = np.mean(fft_vals)
    features["fft_std"] = np.std(fft_vals)
    features["fft_max"] = np.max(fft_vals)
    
    # Spectral centroid
    features["centroid"] = np.sum(freqs * fft_vals) / np.sum(fft_vals)

    # Spectral bandwidth
    features["bandwidth"] = np.sqrt(np.sum(((freqs - features["centroid"])**2) * fft_vals) / np.sum(fft_vals))

    # Spectral roll-off (85%)
    cumulative = np.cumsum(fft_vals) / np.sum(fft_vals)
    rolloff_idx = np.where(cumulative >= 0.85)[0][0]
    features["rolloff"] = freqs[rolloff_idx]

    # Spectral flatness
    gmean = stats.gmean(fft_vals + 1e-10)
    amean = np.mean(fft_vals)
    features["flatness"] = gmean / amean

    return features

# ---------------------------
# MFCC Features
# ---------------------------

def extract_mfcc(signal, samplerate, n_mfcc=13):
    """
    Extract MFCC coefficients from time-domain signal s_t.
    Returns the mean of each MFCC coefficient across all frames.
    """
    mfccs = librosa.feature.mfcc(
        y=signal,
        sr=samplerate,
        n_mfcc=n_mfcc
    )
    
    # Take the mean across time frames
    mfcc_means = mfccs.mean(axis=1)
    
    # Convert to dictionary
    return {f"mfcc_{i+1}": mfcc_means[i] for i in range(n_mfcc)}

# ---------------------------
# Continuous Wavelet Transform (CWT)
# ---------------------------

def extract_cwt(signal, samplerate, wavelet='morl'):
    """
    Compute CWT of the time-domain signal s_t.
    We return the mean energy across all scales to form usable features.
    """
    scales = np.arange(1, 128)

    coefficients, frequencies = pywt.cwt(
        signal,
        scales,
        wavelet,
        sampling_period=1/samplerate
    )

    energy_per_scale = np.mean(np.abs(coefficients)**2, axis=1)

    return {f"cwt_scale_{i+1}": energy_per_scale[i] for i in range(len(scales))}

# ---------------------------
# STFT Features
# ---------------------------

def extract_stft_features(signal, samplerate, nperseg=1024):
    f, t, Zxx = stft(signal, fs=samplerate, nperseg=nperseg)
    magnitude = np.abs(Zxx)

    features = {}
    features["stft_mean"] = np.mean(magnitude)
    features["stft_max"] = np.max(magnitude)

    max_bin = np.argmax(np.sum(magnitude, axis=1))
    features["stft_peak_freq"] = f[max_bin]

    energy_norm = np.sum(magnitude, axis=1)
    energy_norm = energy_norm / np.sum(energy_norm)
    features["stft_bandwidth"] = np.sqrt(np.sum(((f - f[max_bin])**2) * energy_norm))

    return features

# ---------------------------
# Visualization
# ---------------------------

def plot_fft(signal, samplerate):
    fft_vals = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), 1/samplerate)

    plt.figure(figsize=(10,4))
    plt.plot(freqs, fft_vals)
    plt.title("FFT (Frequency Domain)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid(True)
    plt.show()


def plot_stft(signal, samplerate):
    f, t, Zxx = stft(signal, fs=samplerate, nperseg=1024)
    plt.figure(figsize=(10,4))
    plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
    plt.title("STFT Spectrogram")
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (s)")
    plt.colorbar(label="Intensity")
    plt.show()


def plot_mfcc(signal, samplerate):
    mfccs = librosa.feature.mfcc(y=signal, sr=samplerate, n_mfcc=13)
    plt.figure(figsize=(10,4))
    librosa.display.specshow(mfccs, x_axis='time')
    plt.title("MFCC Heatmap")
    plt.colorbar()
    plt.show()


def plot_cwt(signal, samplerate):
    scales = np.arange(1, 128)
    coefficients, frequencies = pywt.cwt(
        signal, scales, 'morl', sampling_period=1/samplerate
    )

    plt.figure(figsize=(10,6))
    plt.imshow(np.abs(coefficients), aspect='auto', cmap='viridis',
               extent=[0, len(signal)/samplerate, scales[-1], scales[0]])
    plt.title("CWT Scalogram")
    plt.xlabel("Time (s)")
    plt.ylabel("Scale")
    plt.colorbar(label="Magnitude")
    plt.show()

# ---------------------------
# MASTER FEATURE EXTRACTION
# ---------------------------

def extract_all_features(filepath, samplerate=48000):
    signal, sr = load_pcm(filepath, samplerate=samplerate)

    td_features = extract_time_domain_features(signal)
    fd_features = extract_frequency_domain_features(signal, sr)
    mfcc_features = extract_mfcc(signal, sr)
    cwt_features = extract_cwt(signal, sr)
    stft_features = extract_stft_features(signal, sr)

    all_features = {}
    all_features.update(td_features)
    all_features.update(fd_features)
    all_features.update(mfcc_features)
    all_features.update(cwt_features)
    all_features.update(stft_features)

    return all_features

# ---------------------------
# MAIN EXECUTION
# ---------------------------
if __name__ == "__main__":
    pcm_file = "test2.pcm"

    signal, sr = load_pcm(pcm_file)

    # Show graphs
    plot_fft(signal, sr)
    plot_stft(signal, sr)
    plot_mfcc(signal, sr)
    plot_cwt(signal, sr)

    # Extract features
    features = extract_all_features(pcm_file)
    print("\nExtracted Features:")
    for k, v in features.items():
        print(f"{k}: {v}")
