import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt, stft

# ----- CONFIG -----
samplerate = 44100
filename = "test2.pcm"   # change to your PCM file name

# ----- 1. LOAD PCM -----
data = np.fromfile(filename, dtype=np.int16).astype(np.float32) / 32768.0

# ----- 2. FFT -----
fft_data = np.fft.rfft(data)
freqs = np.fft.rfftfreq(len(data), 1 / samplerate)

# Plot FFT
plt.figure(figsize=(10,4))
plt.plot(freqs, np.abs(fft_data))
plt.title("FFT (Frequency Domain)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.xlim(0, 22000)
plt.grid()
plt.show()

# ----- 3. BAND-PASS FILTER (18–20 kHz chirp band) -----
low = 18000
high = 20000
sos = butter(10, [low, high], btype='bandpass', fs=samplerate, output='sos')
filtered = sosfilt(sos, data)

# ----- 4. Filtered Time-domain signal -----
plt.figure(figsize=(10,4))
plt.plot(filtered[:2000])
plt.title("Filtered Signal (Time Domain)")
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.grid()
plt.show()

# ----- 5. STFT / SPECTROGRAM -----
f, t, Zxx = stft(filtered, fs=samplerate, nperseg=1024)

plt.figure(figsize=(10,6))
plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
plt.title("Spectrogram (Filtered 18kHz–20kHz Band)")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.ylim(15000, 22000)
plt.colorbar(label='Intensity')
plt.show()
