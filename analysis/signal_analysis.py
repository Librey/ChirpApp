import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft, find_peaks
import os
from pathlib import Path

# ----- CONFIG -----
samplerate = 44100

script_dir = Path(__file__).parent
raw_data_folder = script_dir / "raw_data"
analyzed_data_folder = script_dir / "analyzed_data"

os.makedirs(raw_data_folder, exist_ok=True)
os.makedirs(analyzed_data_folder, exist_ok=True)

# ----- INTERFERENCE CHANNEL ESTIMATION -----
def estimate_interference_channel(baseline_files, samplerate):
    """
    Estimate interference channel from baseline recordings.
    Returns averaged baseline spectrum in log domain (dB).
    """
    print("\n" + "="*60)
    print("ESTIMATING INTERFERENCE CHANNEL FROM BASELINES")
    print("="*60)
    
    baseline_spectra = []
    min_time_bins = float('inf')
    
    for baseline_file in baseline_files:
        try:
            # Load stereo baseline
            data_raw = np.fromfile(baseline_file, dtype=np.int16)
            data_stereo = data_raw.reshape(-1, 2)
            data = data_stereo[:, 0].astype(np.float32) / 32768.0
            
            # Discard direct path
            reflection_samples = int((0.30 / 343) * samplerate)
            data = data[reflection_samples:]
            
            # Simple segmentation
            abs_data = np.abs(data)
            threshold = np.mean(abs_data) * 0.1
            
            if np.max(abs_data) > threshold:
                start_idx = np.where(abs_data > threshold)[0][0]
                data = data[start_idx:]
            
            # STFT
            f, t, Zxx = stft(data, fs=samplerate, nperseg=1024, noverlap=512)
            
            # Convert to dB
            Zxx_db = 20 * np.log10(np.abs(Zxx) + 1e-10)
            baseline_spectra.append(Zxx_db)
            
            # Track minimum time dimension
            min_time_bins = min(min_time_bins, Zxx_db.shape[1])
            
            print(f"✓ Loaded: {baseline_file.name} - Shape: {Zxx_db.shape}")
            
        except Exception as e:
            print(f"✗ Error loading {baseline_file.name}: {e}")
    
    if len(baseline_spectra) == 0:
        print("\nWARNING: No baseline files loaded!")
        return None
    
    print(f"\nCropping all baselines to common size: (513, {min_time_bins})")
    
    # Crop all to same size
    baseline_spectra_cropped = []
    for spec in baseline_spectra:
        cropped = spec[:, :min_time_bins]
        baseline_spectra_cropped.append(cropped)
    
    # Average in log domain (this is the least squares approach)
    baseline_avg = np.mean(baseline_spectra_cropped, axis=0)
    
    print(f"\n✓ Interference channel estimated from {len(baseline_spectra)} baselines")
    print(f"  Channel shape: {baseline_avg.shape}")
    print("="*60 + "\n")
    
    return baseline_avg

# ----- MAIN PROCESSING -----

# Get all PCM files
all_pcm_files = list(Path(raw_data_folder).glob("*.pcm"))

# Separate baselines from data files
baseline_files = [f for f in all_pcm_files if 'baseline' in f.name.lower() or 'idle' in f.name.lower()]
data_files = [f for f in all_pcm_files if f not in baseline_files]

if not all_pcm_files:
    print(f"No .pcm files found in '{raw_data_folder}' folder.")
    exit()

print(f"Found {len(all_pcm_files)} total PCM files:")
print(f"  - {len(baseline_files)} baseline files")
print(f"  - {len(data_files)} data files")

# Estimate interference channel
if len(baseline_files) > 0:
    interference_channel = estimate_interference_channel(baseline_files, samplerate)
else:
    interference_channel = None
    print("\nNo baseline files found - proceeding without interference cancellation\n")

# Process each data file
for pcm_file in data_files:
    print(f"\n{'='*60}")
    print(f"Processing: {pcm_file.name}")
    print(f"{'='*60}")
    
    file_basename = pcm_file.stem
    output_folder = Path(analyzed_data_folder) / file_basename
    os.makedirs(output_folder, exist_ok=True)
    
    # ----- 1. LOAD PCM (STEREO) -----
    try:
        data_raw = np.fromfile(pcm_file, dtype=np.int16)
        data_stereo = data_raw.reshape(-1, 2)
        data = data_stereo[:, 0].astype(np.float32) / 32768.0
        print(f"Loaded {len(data)} samples (stereo)")
    except Exception as e:
        print(f"Error loading {pcm_file.name}: {e}")
        continue
    
    # ----- 2. DISCARD DIRECT PATH -----
    reflection_samples = int((0.30 / 343) * samplerate)
    data = data[reflection_samples:]
    print(f"Remaining: {len(data)} samples")
    
    # ----- 3. SEGMENT SIGNAL (Peak Detection) -----
    abs_data = np.abs(data)
    
    # Smooth envelope
    window = 2205
    envelope = np.convolve(abs_data, np.ones(window)/window, mode='same')
    
    # Find peaks
    threshold = np.mean(envelope) + np.std(envelope)
    peaks, _ = find_peaks(envelope, height=threshold, distance=1000)
    
    if len(peaks) > 0:
        start = max(0, peaks[0] - 1000)
        data = data[start:]
        print(f"Segmented at sample {start}")
    
    # ----- 4. STFT (NO FILTERING) -----
    f, t, Zxx = stft(data, fs=samplerate, nperseg=1024, noverlap=512)
    
    # Convert to dB
    Zxx_db = 20 * np.log10(np.abs(Zxx) + 1e-10)
    
    # Store original for comparison
    Zxx_db_original = Zxx_db.copy()
    t_original = t.copy()
    
    # ----- 5. INTERFERENCE CANCELLATION (Least Squares) -----
    if interference_channel is not None:
        # Crop both to same size
        min_time = min(Zxx_db.shape[1], interference_channel.shape[1])
        
        Zxx_db = Zxx_db[:, :min_time]
        t = t[:min_time]
        channel_cropped = interference_channel[:, :min_time]
        
        # Subtract in log domain (log(a) - log(b) = log(a/b))
        Zxx_db_clean = Zxx_db - channel_cropped
        print(f"✓ Applied interference cancellation (shape: {Zxx_db_clean.shape})")
    else:
        Zxx_db_clean = Zxx_db
    
    # ----- 6. PLOT SPECTROGRAMS -----
    
    # Calculate vmin/vmax
    freq_mask = (f >= 18000) & (f <= 20000)
    relevant_data = Zxx_db_clean[freq_mask, :]
    
    vmin = np.percentile(relevant_data, 1)
    vmax = np.percentile(relevant_data, 95)
    
    # Create comparison plot
    if interference_channel is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Before interference cancellation
        ax1.pcolormesh(t, f, Zxx_db, shading='gouraud', cmap='jet', vmin=vmin, vmax=vmax)
        ax1.set_title(f"Before Interference Cancellation - {file_basename}")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Frequency (Hz)")
        ax1.set_ylim(15000, 22000)
        ax1.set_xlim(0, t[-1])
        
        # After interference cancellation
        im = ax2.pcolormesh(t, f, Zxx_db_clean, shading='gouraud', cmap='jet', vmin=vmin, vmax=vmax)
        ax2.set_title(f"After Interference Cancellation - {file_basename}")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Frequency (Hz)")
        ax2.set_ylim(15000, 22000)
        ax2.set_xlim(0, t[-1])
        
        plt.colorbar(im, ax=[ax1, ax2], label='Magnitude (dB)')
        plt.tight_layout()
        
    else:
        # Single plot
        fig, ax = plt.subplots(1, 1, figsize=(14, 7))
        im = ax.pcolormesh(t, f, Zxx_db_clean, shading='gouraud', cmap='jet', vmin=vmin, vmax=vmax)
        ax.set_title(f"Spectrogram - {file_basename}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_ylim(15000, 22000)
        ax.set_xlim(0, t[-1])
        plt.colorbar(im, label='Magnitude (dB)')
        plt.tight_layout()
    
    plt.savefig(output_folder / "spectrogram.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_folder / 'spectrogram.png'}")

print(f"\n{'='*60}")
print("Processing complete!")
print(f"{'='*60}")