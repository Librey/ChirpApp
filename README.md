# ChirpApp → SensEat

A research project that evolved from a prototype chirp signal detector into a full smartphone-based dietary and mental stress monitoring system.

**Developer:** LaxmiPrasanna Ravikanti  
**Affiliation:** Georgia State University, MS Computer Science  
**Submission:** ACM UbiComp 2026

---

## Project Journey

This repository contains two phases of work:

1. **ChirpApp** — A collaborative prototype built to explore ultrasonic chirp signal detection using a smartphone and machine learning.
2. **SensEat** — The full research system developed by LaxmiPrasanna Ravikanti, extending ChirpApp into a complete dietary and stress monitoring pipeline.

---

## Phase 1 — ChirpApp (Prototype)

**Team:** LaxmiPrasanna Ravikanti, David Salas Carrascal, Liberty Ikpeogu, Syed Shuja Syed

ChirpApp was the initial proof-of-concept that established the core idea: a smartphone can emit inaudible ultrasonic chirps (18–20 kHz) and analyze the reflections to detect acoustic events.

### What ChirpApp Does
- Emits ultrasonic chirp signals via the smartphone speaker
- Records reflections through the microphone
- Detects whether a chirp signal is present in a given audio segment (binary classification)
- Evaluates performance across different chirp period durations (T = 1s, 2s, 4s)

### Android App (mobileApp/)
The Android prototype app for data collection.

- `src/main/MainActivity.kt` — Main activity and recording logic
- `AndroidManifest.xml` — App permissions (microphone, storage)
- `build.gradle.kts` — Build configuration
- `res/` — UI layouts, icons, and resources

### Analysis Pipeline (audio-filter.py)
The prototype ML pipeline that:
- Bandpass filters signals (17.5–20.5 kHz)
- Segments audio into fixed-duration chunks based on chirp period T
- Extracts 4 feature modalities: Statistical, Wavelet, MFCC, Spectrogram (STFT)
- Trains and evaluates classical ML models (SVM, Random Forest, kNN) and a lightweight CNN
- Outputs results to `model_results.csv`

### Quick Start (ChirpApp)
```bash
git clone https://github.com/Librey/ChirpApp.git
cd ChirpApp/analysis
pip install numpy pandas librosa PyWavelets opencv-python scipy scikit-learn tensorflow
python audio-filter.py
```

### ChirpApp Model Details

**Classical ML:**
```
Raw Features → StandardScaler → SVM / RF / kNN → Predictions
```

**CNN Architecture:**
```
Input (64×64×1)
  → Conv2D(16, 3×3) + ReLU → MaxPooling
  → Conv2D(32, 3×3) + ReLU → MaxPooling
  → Flatten → Dense(64) + Dropout(0.4)
  → Dense(1) + Sigmoid
```

### Configuration
```python
SAMPLE_RATE = 44100        # Audio sample rate (Hz)
LOWCUT      = 17500.0      # Bandpass filter lower cutoff (Hz)
HIGHCUT     = 20500.0      # Bandpass filter upper cutoff (Hz)
FILTER_ORDER = 6           # Butterworth filter order
T_VALUES    = [1, 2, 4]   # Chirp periods to test (seconds)
```

### Output Format (model_results.csv)

| Column | Description |
|--------|-------------|
| `T` | Chirp period (1, 2, or 4 seconds) |
| `model` | Model type (SVM, RF, kNN, CNN) |
| `modality` | Feature type (stat, wavelet, mel_spec, mfcc, stft) |
| `eval_type` | cross_validation or 80_20_split |
| `accuracy_mean` | Mean accuracy |
| `precision_mean` | Mean precision |
| `recall_mean` | Mean recall |
| `f1_mean` | Mean F1-score |

---

## Phase 2 — SensEat (Main System)

**Developer:** LaxmiPrasanna Ravikanti

Building on the ChirpApp prototype, LaxmiPrasanna Ravikanti designed and developed SensEat — a complete system for passive dietary monitoring and mental stress prediction using only a smartphone.

### What SensEat Does

| Task | Description | Approach |
|------|-------------|----------|
| **T1** | Eating detection (eating vs. not eating) | Binary classification |
| **T2** | Food recognition (10 food types) | Multiclass classification + Personalization |
| **T3** | Mental stress prediction (scale 1–5) | Regression |

**Food categories:** Tortilla, Fruit, Chicken, Cracker, Carrot, Chocolate, Yogurt, Noodles, Water, Soft Drink

---

### Android App (mobileApp/)

The SensEat Android app developed by LaxmiPrasanna Ravikanti handles continuous ultrasonic sensing during IRB study sessions.

- Records stereo `.pcm` audio with participant and food session metadata
- Supports all 10 food categories used in the IRB study
- Optimized for continuous background sensing
- **Battery performance:** ~400 mAh/hour on Samsung Galaxy S25 (5% drain over 30 minutes)

---

### Analysis Pipeline (analysis/)

Full ML pipeline developed by LaxmiPrasanna Ravikanti.

#### Project Structure

```
analysis/
├── audio_pipeline_multibranch.py              # Main pipeline: Multi-Branch Attention Fusion (T1 + T2 + personalized T2)
├── audio_pipeline_lopo_multiclass_balanced.py # LOPO multiclass with per-fold undersampling
├── audio_pipeline_lopo_multiclass.py          # LOPO multiclass baseline
├── audio_pipeline_lopo_balanced.py            # LOPO binary with balancing
├── audio_pipeline_lopo.py                     # LOPO binary baseline
├── audio_pipeline_8020.py                     # 80/20 train-test split pipeline
├── audio_pipeline_grid_search.py              # Hyperparameter grid search
├── audio_pipeline_per_participant.py          # Per-participant training and evaluation
├── audio_pipeline_robustness.py               # Robustness evaluation across conditions
├── audio_pipeline_sar_classifier.py           # SAR-style echo profile classifier
├── audio_pipeline_1_5s.py                    # 1.5s segment pipeline variant
├── evaluate_personalized.py                   # Evaluates pre-trained personalized models (users 001-020)
├── generate_personalized_eval_report.py       # PDF report for personalized model evaluation
├── generate_t2_report.py                      # PDF report for T2 personalization (users 022-041)
├── generate_report.py                         # General PDF report generator
├── feature_extraction.py                      # STFT / MFCC / GFCC feature extraction utilities
├── signal_analysis.py                         # Raw signal visualization and analysis
├── plot_signal.py                             # Time-domain signal plotting
├── plot_time_freq.py                          # Time-frequency spectrogram plotting
├── segmentation_chirp_level.py               # Chirp-level segmentation
├── segmentation_servings_level.py            # Serving-level segmentation
├── senseat/
│   └── training/
│       └── trainer.py                         # Trainer wrapper: LOPO, personalization, augmentation
├── personalized_t2_results.csv               # T2 per-user evaluation results (users 001-020)
├── personalized_t3_predictions.csv           # T3 raw stress predictions (users 001-020)
├── pipeline_multibranch_results.csv          # T1/T2/T2-personalized results
├── pipeline_lopo_*_results.csv               # LOPO experiment results
└── pipeline_*_foldwise.csv                   # Per-fold breakdowns
```

#### Setup

**Requirements:** Python 3.9+

```bash
pip install numpy scipy matplotlib seaborn pandas scikit-learn tensorflow keras reportlab
```

#### Running the Pipelines

**T1 + T2 + Personalized T2 (Multi-Branch):**
```bash
python audio_pipeline_multibranch.py
```

**LOPO Multiclass (Balanced):**
```bash
python audio_pipeline_lopo_multiclass_balanced.py
```

**Evaluate Pre-Trained Personalized Models (Users 001–020):**
```bash
python evaluate_personalized.py
```
Requires:
- `feature_cache_multiclass_balanced_v2.npz`
- `personalized_models/T2_user_XXX_multiclass.keras`
- `personalized_models/T2_user_XXX_normalization.npz`

**Generate PDF Reports:**
```bash
python generate_personalized_eval_report.py
python generate_t2_report.py
```

---

### Model Architecture

The **Multi-Branch Attention Fusion** model combines four parallel branches:

| Branch | Input | Type |
|--------|-------|------|
| 1 | STFT spectrogram (64×64) | 2D CNN |
| 2 | MFCC features | 2D CNN |
| 3 | GFCC (Gammatone) features | 2D CNN |
| 4 | Statistical features | 1D CNN |

All branches are fused via an **attention-weighted layer** before the final classification/regression head.

---

### Personalization Approach

Food recognition accuracy varies significantly between users due to individual eating patterns and acoustic environments. SensEat addresses this with:

- **LOPO Fine-tuning:** Global model trained on N−1 participants, then fine-tuned on 20% of the held-out participant's data and evaluated on the remaining 80%.
- **Per-user normalization:** STFT features z-score normalized per user using saved `normalization.npz` files.
- **Per-fold undersampling:** Food classes balanced to minority class count within each fold to prevent class imbalance bias.

---

### Key Findings

- **T1 binary detection** achieves strong performance across participants without personalization, confirming that eating events produce consistent ultrasonic signatures.
- **Cross-user food recognition (T2)** is inherently limited because individual eating patterns vary significantly between users.
- **Personalization is the key factor:** Per-user fine-tuned models show substantial improvement over cross-user baselines, demonstrating that user-specific adaptation is essential for practical food recognition.
- **T3 stress prediction** generates per-user stress level estimates.

These results support the feasibility of passive, camera-free dietary and stress monitoring on commodity smartphones.

---

## Notes

- Raw audio data (`.pcm`), feature caches (`feature_cache_*.npz`), and model files (`personalized_models/`) are not included in this repository due to size.
- All SensEat pipelines tested on Windows 11, Python 3.13, TensorFlow 2.x (CPU only).

---

## License

This project is part of Georgia State University research.

---

## Contact

**LaxmiPrasanna Ravikanti**  
MS Computer Science, Georgia State University  
lravikanti1@student.gsu.edu
