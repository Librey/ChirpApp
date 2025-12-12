import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt, stft
import os
from pathlib import Path

# ----- CONFIG -----
samplerate = 44100

# Get the directory where this script is located
script_dir = Path(__file__).parent
raw_data_folder = script_dir / "raw_data"
analyzed_data_folder = script_dir / "analyzed_data"

# Create folders if they don't exist
os.makedirs(raw_data_folder, exist_ok=True)
os.makedirs(analyzed_data_folder, exist_ok=True)

# Get all .pcm files from raw_data folder
pcm_files = list(Path(raw_data_folder).glob("*.pcm"))

if not pcm_files:
    print(f"No .pcm files found in '{raw_data_folder}' folder.")
    print("Please add your PCM files to the 'raw_data' folder and run again.")
    exit()

print(f"Found {len(pcm_files)} PCM file(s) to process.")

# Process each PCM file
for pcm_file in pcm_files:
    print(f"\n{'='*60}")
    print(f"Processing: {pcm_file.name}")
    print(f"{'='*60}")
    
    # Create output subfolder for this file
    file_basename = pcm_file.stem  # filename without extension
    output_folder = Path(analyzed_data_folder) / file_basename
    os.makedirs(output_folder, exist_ok=True)
    
    # ----- 1. LOAD PCM -----
    try:
        data = np.fromfile(pcm_file, dtype=np.int16).astype(np.float32) / 32768.0
        print(f"Loaded {len(data)} samples from {pcm_file.name}")
    except Exception as e:
        print(f"Error loading {pcm_file.name}: {e}")
        continue
    
    # ----- 2. FFT -----
    fft_data = np.fft.rfft(data)
    freqs = np.fft.rfftfreq(len(data), 1 / samplerate)
    
    # Plot FFT
    plt.figure(figsize=(10,4))
    plt.plot(freqs, np.abs(fft_data))
    plt.title(f"FFT (Frequency Domain) - {file_basename}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.xlim(0, 22000)
    plt.grid()
    plt.savefig(output_folder / "fft.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_folder / 'fft.png'}")
    
    # ----- 3. BAND-PASS FILTER (18–20 kHz chirp band) -----
    low = 18000
    high = 20000
    sos = butter(10, [low, high], btype='bandpass', fs=samplerate, output='sos')
    filtered = sosfilt(sos, data)
    
    # ----- 4. Filtered Time-domain signal -----
    plt.figure(figsize=(10,4))
    plt.plot(filtered[:2000])
    plt.title(f"Filtered Signal (Time Domain) - {file_basename}")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.savefig(output_folder / "filtered_time_domain.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_folder / 'filtered_time_domain.png'}")
    
    # ----- 5. STFT / SPECTROGRAM -----
    f, t, Zxx = stft(filtered, fs=samplerate, nperseg=1024)
    
    plt.figure(figsize=(10,6))
    plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
    plt.title(f"Spectrogram (Filtered 18kHz–20kHz Band) - {file_basename}")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.ylim(15000, 22000)
    plt.colorbar(label='Intensity')
    plt.savefig(output_folder / "spectrogram.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_folder / 'spectrogram.png'}")

print(f"\n{'='*60}")
print(f"Processing complete! All results saved to '{analyzed_data_folder}' folder.")
print(f"{'='*60}")
