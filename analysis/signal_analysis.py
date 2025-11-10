import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, spectrogram
import os

# === 1. Load your PCM file ===
filename = "test2.pcm"  # update with your actual file name
fs = 44100  # sample rate used in the Android app
dtype = np.int16  # 16-bit PCM

# Load the raw PCM data
data = np.fromfile(filename, dtype=dtype)

# === 2. Plot Time Domain Signal ===
plt.figure()
plt.plot(np.arange(len(data)) / fs, data)
plt.title("Time Domain Signal")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid()
plt.savefig("time_domain.png")
plt.close()

# === 3. Perform FFT ===
fft_data = np.fft.fft(data)
freqs = np.fft.fftfreq(len(fft_data), 1/fs)

half = len(freqs) // 2
plt.figure()
plt.plot(freqs[:half], np.abs(fft_data[:half]))
plt.title("Frequency Domain (FFT)")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Amplitude")
plt.grid()
plt.savefig("fft_signal.png")
plt.close()

# === 4. Apply Band-Pass Filter (BPF) ===
lowcut, highcut = 18000, 20000  # ultrasonic range
b, a = butter(4, [lowcut / (fs / 2), highcut / (fs / 2)], btype="band")
filtered = filtfilt(b, a, data)

# === 5. Inverse FFT (IFFT) after filtering ===
ifft_data = np.fft.ifft(np.fft.fft(filtered)).real

plt.figure()
plt.plot(np.arange(len(ifft_data)) / fs, ifft_data)
plt.title("Filtered Signal (After BPF + IFFT)")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid()
plt.savefig("filtered_signal.png")
plt.close()

# === 6. Spectrogram (STFT) ===
f, t, Sxx = spectrogram(filtered, fs, nperseg=1024)
plt.figure()
plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
plt.title("Spectrogram (STFT)")
plt.ylabel("Frequency [Hz]")
plt.xlabel("Time [s]")
plt.colorbar(label='Power [dB]')
plt.ylim(0, 25000)
plt.savefig("spectrogram.png")
plt.close()

print("✅ All graphs saved successfully!")
print(f"Saved files in: {os.getcwd()}")
