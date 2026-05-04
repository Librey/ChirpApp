# SensEat — Analysis Pipeline

**Developer:** LaxmiPrasanna Raavi  
**Project:** SensEat — Smartphone-Based Dietary and Mental Stress Monitoring  
**Affiliation:** Georgia State University, MS Computer Science  
**Submission:** ACM UbiComp 2026

---

## Overview

This folder contains the full Python analysis pipeline for the SensEat project.
SensEat uses ultrasonic chirp signals emitted from a smartphone to recognize food types
(T2) and predict mental stress levels (T3) through acoustic reflection analysis.

The pipeline covers signal processing, feature extraction, model training (binary and
multiclass), personalization via Leave-One-Participant-Out (LOPO) fine-tuning, and
evaluation of pre-trained personalized models.

---

## Setup

**Requirements:** Python 3.9+

Install dependencies:

```bash
pip install numpy scipy matplotlib seaborn pandas scikit-learn tensorflow keras reportlab
```

---

## Project Structure

```
analysis/
├── audio_pipeline_multibranch.py         # Main pipeline: Multi-Branch Attention Fusion (T1 + T2 + personalized T2)
├── audio_pipeline_lopo_multiclass_balanced.py  # LOPO multiclass with per-fold undersampling
├── audio_pipeline_lopo_multiclass.py     # LOPO multiclass baseline
├── audio_pipeline_lopo_balanced.py       # LOPO binary with balancing
├── audio_pipeline_lopo.py                # LOPO binary baseline
├── audio_pipeline_8020.py                # 80/20 train-test split pipeline
├── audio_pipeline_grid_search.py         # Hyperparameter grid search
├── audio_pipeline_per_participant.py     # Per-participant training and evaluation
├── audio_pipeline_robustness.py          # Robustness evaluation across conditions
├── audio_pipeline_sar_classifier.py      # SAR-style echo profile classifier
├── audio_pipeline_1_5s.py               # 1.5s segment pipeline variant
├── evaluate_personalized.py             # Evaluates professor's pre-trained personalized models (users 001-020)
├── generate_personalized_eval_report.py # Generates PDF report for personalized model evaluation
├── generate_t2_report.py                # Generates PDF report for T2 personalization (users 022-041)
├── generate_report.py                   # General PDF report generator
├── verify_report.py                     # Verifies report output
├── feature_extraction.py                # STFT / MFCC / GFCC feature extraction utilities
├── signal_analysis.py                   # Raw signal visualization and analysis
├── plot_signal.py                        # Time-domain signal plotting
├── plot_time_freq.py                     # Time-frequency spectrogram plotting
├── visualize_results.py                  # Results visualization helpers
├── visualize_signals.py                  # Signal visualization helpers
├── segmentation_chirp_level.py          # Chirp-level segmentation
├── segmentation_servings_level.py       # Serving-level segmentation
├── senseat/
│   └── training/
│       └── trainer.py                   # Trainer wrapper: LOPO, personalization, augmentation
├── personalized_t2_results.csv          # T2 per-user evaluation results (users 001-020)
├── personalized_t3_predictions.csv      # T3 raw stress predictions (users 001-020)
├── pipeline_multibranch_results.csv     # T1/T2/T2-personalized results from multibranch pipeline
├── pipeline_lopo_*_results.csv          # LOPO experiment results (binary and multiclass)
└── pipeline_*_foldwise.csv             # Per-fold breakdowns for LOPO experiments
```

---

## Tasks

| Task | Description | Model |
|------|-------------|-------|
| **T1** | Binary food detection (eating vs. not eating) | Multi-Branch Attention Fusion |
| **T2** | 10-class food recognition (Tortilla, Fruit, Chicken, Cracker, Carrot, Chocolate, Yogurt, Noodles, Water, Soft Drink) | Multi-Branch Attention Fusion + Personalization |
| **T3** | Mental stress regression (scale 1–5) | Personalized Keras model |

---

## Model Architecture

The **Multi-Branch Attention Fusion** model uses three parallel branches:
- **Branch 1:** 2D CNN on STFT spectrogram (64×64×1)
- **Branch 2:** 2D CNN on MFCC features
- **Branch 3:** 2D CNN on GFCC (Gammatone) features
- **Branch 4:** 1D CNN on flattened statistical features

Outputs from all branches are combined using an **attention-weighted fusion** layer before the final classification head.

---

## Running the Main Pipeline

### T1 + T2 + Personalized T2 (Multi-Branch)

```bash
python audio_pipeline_multibranch.py
```

Outputs: `pipeline_multibranch_results.csv`

### LOPO Multiclass (Balanced)

```bash
python audio_pipeline_lopo_multiclass_balanced.py
```

Outputs: `pipeline_lopo_multiclass_balanced_results.csv`, `pipeline_lopo_multiclass_balanced_foldwise.csv`

### Evaluate Professor's Pre-Trained Personalized Models (Users 001–020)

```bash
python evaluate_personalized.py
```

Requires:
- `feature_cache_multiclass_balanced_v2.npz`
- `personalized_models/T2_user_XXX_multiclass.keras`
- `personalized_models/T2_user_XXX_normalization.npz`

Outputs: `personalized_t2_results.csv`, `personalized_t3_predictions.csv`

### Generate PDF Reports

```bash
python generate_personalized_eval_report.py   # Personalized model evaluation report
python generate_t2_report.py                  # T2 personalization report (users 022-041)
```

---

## Key Results

| Experiment | Accuracy | Macro F1 |
|------------|----------|----------|
| T1 Binary (cross-user) | ~95% | — |
| T2 Cross-user baseline | 12.85% | — |
| T2 Personalized LOPO (users 022–041) | 31.77% | — |
| T2 Personalized models (users 001–020) | **73.82%** | **0.4884** |

Personalization provides a **5.75× improvement** over the cross-user baseline.

---

## Personalization Approach

- **LOPO Fine-tuning:** Train global model on N−1 participants, fine-tune on 20% of held-out participant's data, evaluate on remaining 80%.
- **Per-user normalization:** Each user's STFT features are z-score normalized using saved `normalization.npz` files (keys: `mean`, `std`, shape `64×64×1`).
- **Per-fold undersampling:** Food classes are balanced to the minority class count within each fold to prevent class imbalance bias.

---

## Notes

- Feature caches (`feature_cache_*.npz`) and model files (`personalized_models/`) are not included in the repository due to size. Contact the project team to obtain them.
- Raw audio data (`.pcm` files) is also excluded from the repository.
- T3 stress evaluation requires `stress_labels.npz` — contact the professor for this file.
- All pipelines are tested on Windows 11 with Python 3.13 and TensorFlow 2.x (CPU only).

---

## Contact

**LaxmiPrasanna Ravikanti**  
MS Computer Science, Georgia State University  
