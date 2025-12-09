import numpy as np
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt

# --- Configuration ---
# UPDATE THIS with the path to your PCM file
file_path = 'chirp_reflection_xxxxxxxx.pcm'
# Audio properties from the Android app
sample_rate = 44100  # Hz
dtype = np.int16     # 16-bit signed integer

# Filter properties for cleaning the signal
lowcut = 17500.0   # Lower bound of our chirp frequency (a bit of margin)
highcut = 20500.0  # Upper bound of our chirp frequency (a bit of margin)
order = 6          # Filter order - higher is sharper but can be less stable

# --- 1. Load the Raw Signal ---
try:
    raw_signal = np.fromfile(file_path, dtype=dtype)
    print(f"Successfully loaded {len(raw_signal)} samples from '{file_path}'.")
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found. Please check the file name and path.")
    exit()

# Generate a time axis for plotting
time_axis = np.arange(len(raw_signal)) / float(sample_rate)

# --- 2. Clean the Signal with a Band-Pass Filter ---
def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    """Applies a Butterworth band-pass filter to the data."""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)
    return y

# Apply the filter
print(f"Applying a band-pass filter from {lowcut/1000}kHz to {highcut/1000}kHz...")
cleaned_signal = butter_bandpass_filter(raw_signal, lowcut, highcut, sample_rate, order)

# --- 3. Visualize the Results ---
fig, axs = plt.subplots(3, 1, figsize=(12, 18))
plt.style.use('seaborn-v0_8-whitegrid')

# Plot 1: Time Domain
axs[0].plot(time_axis, raw_signal, label='Raw Signal', color='silver', alpha=0.8)
axs[0].plot(time_axis, cleaned_signal, label='Cleaned Signal (17.5-20.5 kHz)', color='cornflowerblue')
axs[0].set_title('Signal in the Time Domain', fontsize=16)
axs[0].set_xlabel('Time (seconds)', fontsize=12)
axs[0].set_ylabel('Amplitude', fontsize=12)
axs[0].legend()
axs[0].grid(True)

# Plot 2: Frequency Domain (FFT)
fft_raw = np.fft.fft(raw_signal)
fft_cleaned = np.fft.fft(cleaned_signal)
freqs = np.fft.fftfreq(len(raw_signal), 1/sample_rate)
# Keep only the positive frequencies
positive_freqs = freqs[:len(freqs)//2]
fft_raw_magnitude = np.abs(fft_raw)[:len(freqs)//2]
fft_cleaned_magnitude = np.abs(fft_cleaned)[:len(freqs)//2]

axs[1].plot(positive_freqs / 1000, fft_raw_magnitude, label='Raw Signal FFT', color='silver', alpha=0.8)
axs[1].plot(positive_freqs / 1000, fft_cleaned_magnitude, label='Cleaned Signal FFT', color='indianred')
axs[1].set_title('Signal in the Frequency Domain', fontsize=16)
axs[1].set_xlabel('Frequency (kHz)', fontsize=12)
axs[1].set_ylabel('Magnitude', fontsize=12)
axs[1].set_xlim(0, sample_rate / 2000) # Show up to the Nyquist frequency
axs[1].axvspan(lowcut/1000, highcut/1000, color='gray', alpha=0.2, label='Target Frequency Band')
axs[1].legend()
axs[1].grid(True)


# Plot 3: Spectrogram of the Cleaned Signal
Pxx, freqs_spec, bins, im = axs[2].specgram(
    cleaned_signal,
    NFFT=1024,
    Fs=sample_rate,
    noverlap=512,
    cmap='viridis'
)
axs[2].set_title('Spectrogram of the Cleaned Signal', fontsize=16)
axs[2].set_xlabel('Time (seconds)', fontsize=12)
axs[2].set_ylabel('Frequency (Hz)', fontsize=12)
axs[2].set_ylim(lowcut - 1000, highcut + 1000) # Zoom into our frequency band
cbar = fig.colorbar(im, ax=axs[2])
cbar.set_label('Intensity (dB)')


# Final adjustments and display
plt.tight_layout(pad=3.0)
plt.suptitle('Chirp Signal Analysis', fontsize=20, y=1.02)
plt.show()
