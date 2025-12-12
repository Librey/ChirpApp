# ChirpApp

A machine learning-powered audio analysis system designed to detect chirp signals in acoustic environments. ChirpApp combines mobile data collection with advanced signal processing and ML models to classify audio as containing chirp signals or not.

---

## 🎯 Project Overview

ChirpApp addresses the problem of detecting ultrasonic chirp signals in noisy environments. The project uses a data-driven approach to:

1. **Collect audio samples** via a mobile app
2. **Process signals** with bandpass filtering and feature extraction
3. **Train multiple ML models** to classify chirp presence
4. **Evaluate performance** across different chirp period durations (1s, 2s, 4s)
5. **Select optimal parameters** based on model prediction accuracy

The system is designed to be robust, generalizable, and suitable for deployment in real-world acoustic monitoring applications.

---

## 📁 Project Structure

### **mobileApp/** — Android Data Collection
Contains the Android mobile application for recording and collecting audio samples.

**Key Components:**
- `src/main/` — Source code for the Android app
  - `MainActivity.kt` — Main application activity and UI
  - `ui/theme/` — Custom Material Design theme configuration
- `AndroidManifest.xml` — App permissions and configurations
- `build.gradle.kts` — Gradle build configuration
- `res/` — Resource files (layouts, strings, drawables, icons)

**Purpose:** Records audio clips in natural environments and labels them as positive (contains chirp) or negative (no chirp) for training data collection.

---

### **analysis/** — Signal Processing & Machine Learning
Core data analysis and model training pipeline.

**Key Files:**

#### `audio-filter.py` (Main Pipeline)
The comprehensive analysis script that:

1. **Loads & Cleans Audio**
   - Bandpass filters signals (17.5 kHz - 20.5 kHz)
   - Converts `.wav` and `.pcm` files
   - Normalizes amplitude

2. **Segments Signals**
   - Divides long audio into fixed-duration chunks based on chirp period T
   - Tests three periods: T ∈ {1s, 2s, 4s}
   - Each segment becomes one training/test sample

3. **Extracts Features** (4 modalities)
   - **Statistical Features**: Mean, std, max, min, power, zero-crossing rate
   - **Wavelet Features**: Discrete wavelet coefficients (db4, level 4)
   - **MFCC (Mel-Frequency Cepstral Coefficients)**: 40 coefficients, 64 frames
   - **Spectrograms**: Mel-spectrogram and STFT (64×64 normalized images)

4. **Trains Models**
   - **Classical ML** (on statistical/wavelet features):
     - Support Vector Machine (SVM)
     - Random Forest (100 estimators)
     - k-Nearest Neighbors (k=5)
   - **Deep Learning** (on spectral features):
     - Simple 2D CNN with Conv2D → MaxPool → Dense layers
     - Trained separately on mel-spec, MFCC, wavelet scalogram, STFT

5. **Evaluates Performance**
   - **Cross-validation**: 5-fold stratified K-fold (or fewer with small datasets)
   - **Train/Test Split**: 80% train, 20% test (when sample size permits)
   - **Metrics Recorded**:
     - Accuracy, Precision, Recall, F1-score
     - Mean ± std for all metrics

6. **Saves Results**
   - Outputs `model_results.csv` with comprehensive results:
     - T value, model type, feature modality, evaluation type
     - Full performance metrics for every model×modality×T combination

#### Data Directories
- `raw_data/` — Positive samples (contain chirp signals)
- `negative_data/` — Negative samples (no chirp signals)
- `analyzed_data/` — Output folder for processed results

#### Requirements
```
numpy, pandas, librosa, PyWavelets, opencv-python, scipy, scikit-learn, tensorflow
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Virtual environment (recommended)

### Installation

```bash
# Clone the repository
git clone https://github.com/DaveTron4/ChirpApp.git
cd ChirpApp/analysis

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install numpy pandas librosa PyWavelets opencv-python scipy scikit-learn tensorflow
```

### Running the Analysis

```bash
python audio-filter.py
```

**Output:**
- Console logs showing progress for each T value
- `model_results.csv` — Full results table with all metrics

---

## 📊 Results & Interpretation

The pipeline tests three chirp periods and evaluates each model across all feature modalities.

### Key Metrics per Model:
- **Accuracy**: Overall correctness
- **Precision**: True positive rate among predicted positives
- **Recall**: Detection rate of actual positives
- **F1-Score**: Harmonic mean of precision & recall

### Expected Performance:
- **T=1s**: Best results (23 samples) — Models typically achieve 95-100% accuracy
- **T=2s**: Good results (11 samples) — Models typically achieve 90-100% accuracy
- **T=4s**: Limited by data (5 samples) — Less reliable; ~58% accuracy

### Choosing the Optimal T:
Based on results in `model_results.csv`, **T=1s** is recommended for production because:
- Most training samples
- Highest and most stable accuracy
- Best generalization to test data
- Consistent performance across all model types

---

## 🔧 Configuration

Edit `audio-filter.py` constants to customize the pipeline:

```python
SAMPLE_RATE = 44100        # Audio sample rate (Hz)
LOWCUT = 17500.0           # Bandpass filter lower cutoff (Hz)
HIGHCUT = 20500.0          # Bandpass filter upper cutoff (Hz)
FILTER_ORDER = 6           # Butterworth filter order
T_VALUES = [1, 2, 4]       # Chirp periods to test (seconds)
LOAD_POSITIVE = True       # Include positive samples
LOAD_NEGATIVE = True       # Include negative samples
```

---

## 📈 Model Details

### Classical ML Pipeline
```
Raw Features → StandardScaler → SVM/RF/kNN → Predictions
```
Uses 5-fold stratified cross-validation for robust evaluation.

### CNN Architecture
```
Input (64×64×1 image) 
  → Conv2D(16, 3×3) + ReLU
  → MaxPooling2D(2×2)
  → Conv2D(32, 3×3) + ReLU
  → MaxPooling2D(2×2)
  → Flatten
  → Dense(64) + ReLU + Dropout(0.4)
  → Dense(1) + Sigmoid
```

Simple, lightweight design to avoid overfitting on small datasets.

---

## 📝 Output Format

`model_results.csv` contains:

| Column | Description |
|--------|-------------|
| `T` | Chirp period (1, 2, or 4 seconds) |
| `model` | Model type (SVM, RF, kNN, CNN) |
| `modality` | Feature type (stat, wavelet, mel_spec, mfcc, wavelet_scalo, stft) |
| `eval_type` | Evaluation method (cross_validation or 80_20_split) |
| `accuracy_mean` | Mean accuracy score |
| `accuracy_std` | Standard deviation of accuracy |
| `precision_mean` | Mean precision |
| `recall_mean` | Mean recall |
| `f1_mean` | Mean F1-score |

---

## ⚙️ Troubleshooting

### "No samples found" error
- Check that `raw_data/` and `negative_data/` folders contain `.wav` or `.pcm` files
- Ensure files are not corrupted

### NaN values in results
- Occurs when dataset is too small (T=4s with <5 samples)
- kNN in particular struggles with minimal data
- Recommendation: Collect more audio samples

### TensorFlow warnings
- "oneDNN custom operations" and "function retracing" warnings are informational
- They don't affect results; suppress with `TF_ENABLE_ONEDNN_OPTS=0`

---

## 🎓 Future Improvements

- Add data augmentation techniques (time-stretch, pitch-shift)
- Implement ensemble methods combining multiple modalities
- Deploy optimal model to mobile app
- Support real-time inference on device
- Add confusion matrices and ROC curves to analysis

---

## 📄 License

This project is part of Georgia State University coursework.

---

## 👤 Author

David (DaveTron4)

**Last Updated:** December 9, 2025
