---
description: how to run the SensEat audio classification pipelines
---

# SensEat Audio Pipeline Workflow

This workflow describes the steps to execute the audio processing and classification pipelines for the SensEat project.

## 1. Environment Setup
Before running any scripts, ensure all dependencies are installed.
// turbo
1. Run `pip install -r requirements.txt` to install `numpy`, `librosa`, `tensorflow`, `sklearn`, `spafe`, and other required libraries.

## 2. Data Preparation
Ensure the raw PCM data is organized in the `raw_data/` directory.
- Our internal data should be in folders `001/` through `020/`.
- Idle baseline data should be in `Idle/` or `idle-gp/`.
- External validation data (e.g., SDSU) should be in its own named folder.

## 3. Running Binary Classification
This pipeline predicts "Eating" vs "Idle" using 1.5s segments.
1. Run `python audio_pipeline_1_5s.py`.
2. Results will be saved to `pipeline_1_5s_results.csv`.
3. Check the console for ranked accuracy metrics across all feature combinations (STFT, MFCC, Wavelets, etc.).

## 4. Running Multi-class Classification
This pipeline predicts specific food items (e.g., Crackers vs Apple).
1. Run `python audio_pipeline_multiclass.py`.
2. This script uses an **overlapping sliding window** (1.5s window, 0.5s hop) to expand the dataset.
3. It extracts dense features including **80-coefficient MFCCs** and **GFCCs**.
4. Results and confusion matrices will be saved to `pipeline_multiclass_results.csv` and the `figures/` folder.

## 5. Running External Data Validation (SDSU)
To verify the model generalizes to new hardware and participants.
1. Run `python audio_pipeline_sdsu.py`.
2. This script adapts to alternate filename conventions and labels based on keyword matches (e.g., `_idle_`).
3. Results will be saved to `pipeline_sdsu_results.csv`.

## 6. Visualization and Analysis
1. To generate high-resolution comparison charts for a research paper, run `python visualize_results.py`.
2. To audit the physical signal quality and amplitude balance across classes, run `python evaluate_all_magnitudes.py`.
3. To generate magnitude envelope plots for the professor, run `python plot_all_magnitudes.py`.

## 7. Synthetic Aperture Radar (SAR) Exploration
1. To explore jaw motion micro-Doppler signatures via FMCW dechirping, run `python audio_pipeline_sar.py`.
2. This generates Range-Time Intensity (RTI) spectrograms for baseband jaw motion analysis.
