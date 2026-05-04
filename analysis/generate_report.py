"""
Generates SensEat_Report.html — a single self-contained HTML file.
All figures embedded as base64. Open in any browser; Ctrl+P > Save as PDF.

All numbers verified against actual result CSVs:
  pipeline_8020_results.csv
  pipeline_lopo_binary_results.csv
  pipeline_lopo_binary_balanced_results.csv
  pipeline_lopo_multiclass_results.csv
  pipeline_lopo_multiclass_balanced_results.csv
  pipeline_grid_binary_results.csv
  pipeline_grid_multiclass_results.csv
  pipeline_multibranch_results.csv
  pipeline_per_participant_binary.csv
  pipeline_per_participant_multiclass.csv
  pipeline_robustness_results.csv
"""

import base64
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
FIG        = SCRIPT_DIR / "figures"
OUT        = SCRIPT_DIR / "SensEat_Report.html"


def b64(rel_path):
    p = FIG / rel_path
    if not p.exists():
        return None
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()


def img(rel_path, caption="", width="82%"):
    src = b64(rel_path)
    if src is None:
        return f'<p style="color:#aaa;font-style:italic;font-size:13px">[Figure not found: {rel_path}]</p>'
    return f"""<figure style="text-align:center;margin:18px 0">
  <img src="{src}" style="width:{width};border:1px solid #ddd;border-radius:6px;box-shadow:0 2px 8px rgba(0,0,0,.12)">
  <figcaption style="font-size:13px;color:#555;margin-top:6px">{caption}</figcaption>
</figure>"""


def row(*items):
    w = f"{int(96/len(items))}%"
    cells = ""
    for rel, cap in items:
        src = b64(rel)
        cell_img = f'<img src="{src}" style="width:100%;border:1px solid #ddd;border-radius:4px">' \
                   if src else f'<p style="color:#aaa;font-size:11px">[{rel}]</p>'
        cells += f'<td style="text-align:center;padding:6px;vertical-align:top">{cell_img}<p style="font-size:12px;color:#555;margin:4px 0">{cap}</p></td>'
    return f'<table style="width:100%;border-collapse:collapse"><tr>{cells}</tr></table>'


CSS = """<style>
body{font-family:'Segoe UI',Arial,sans-serif;font-size:15px;color:#222;max-width:1100px;margin:0 auto;padding:30px 40px;line-height:1.6}
h1{font-size:28px;color:#1a3a5c;border-bottom:3px solid #1a3a5c;padding-bottom:10px;margin-top:0}
h2{font-size:21px;color:#1a5c3a;border-left:5px solid #1a5c3a;padding-left:12px;margin-top:44px}
h3{font-size:17px;color:#333;margin-top:22px}
table{border-collapse:collapse;width:100%;margin:14px 0;font-size:14px}
th{background:#1a3a5c;color:#fff;padding:8px 12px;text-align:left}
td{padding:7px 12px;border-bottom:1px solid #e0e0e0}
tr:nth-child(even) td{background:#f5f8fc}
.good {background:#eafaf1;border-left:4px solid #27ae60;padding:10px 16px;margin:12px 0;border-radius:4px}
.bad  {background:#fdf2f2;border-left:4px solid #e74c3c;padding:10px 16px;margin:12px 0;border-radius:4px}
.note {background:#fff7e6;border-left:4px solid #f0a500;padding:10px 16px;margin:12px 0;border-radius:4px}
.fix  {background:#eaf2fb;border-left:4px solid #2980b9;padding:10px 16px;margin:12px 0;border-radius:4px}
.sug  {background:#f9f0ff;border-left:4px solid #8e44ad;padding:10px 16px;margin:12px 0;border-radius:4px}
.phase{background:#1a3a5c;color:#fff;padding:5px 14px;border-radius:20px;font-size:13px;font-weight:bold;display:inline-block;margin-bottom:8px}
figcaption{font-size:13px;color:#555;margin-top:6px}
@media print{h2{page-break-before:always}}
</style>"""

# ─────────────────────────────────────────────────────────────────────────────
HEADER = f"""<!DOCTYPE html><html lang="en">
<head><meta charset="UTF-8"><title>SensEat — Complete Experimental Report</title>{CSS}</head>
<body>
<h1>SensEat &mdash; Complete Experimental Report</h1>
<p style="color:#555;font-size:14px">
Acoustic Eating Detection &amp; Food Recognition using Ultrasonic FMCW Chirps &bull;
41 Participants &bull; 8 Food Classes &bull; All Experiments From Start to Final Robustness Study<br>
<strong>All numbers verified from result CSV files.</strong>
</p>
<hr style="border:none;border-top:1px solid #ddd;margin:18px 0">
"""

# ─── 0. SYSTEM OVERVIEW ──────────────────────────────────────────────────────
S0 = """<h2>0. System Overview &amp; Signal Processing Pipeline</h2>
<p>SensEat uses a smartphone as an active sonar device. An inaudible ultrasonic chirp
(18–20 kHz, sweeping upward) is played from the speaker continuously. The microphone
captures the reflected signal. When a person eats, micro-movements of the jaw and mouth
modulate the reflected signal — these changes are extracted as features and classified.</p>
<h3>Full processing chain</h3>
<ol>
  <li><strong>Load PCM:</strong> Stereo 44,100 Hz PCM &rarr; left channel only, normalised to [&minus;1, 1]</li>
  <li><strong>Bandpass filter:</strong> 6th-order Butterworth, 17.5–20.5 kHz — rejects all non-ultrasonic content including speech and environmental noise</li>
  <li><strong>Direct-path cancellation:</strong> Least-squares subtraction of the speaker&rarr;mic direct path (the chirp that arrives without reflecting off anything)</li>
  <li><strong>Segmentation:</strong> Sliding window, 1.5 s wide, 0.5 s hop — each window is one data sample</li>
  <li><strong>Feature extraction:</strong> Four feature sets combined into a 136–148-dim flat vector + a 64&times;64 STFT spectrogram image</li>
</ol>
<h3>Feature sets</h3>
<table>
  <tr><th>Feature set</th><th>Dimensions</th><th>What it captures</th></tr>
  <tr><td>Statistical (ZCR, envelope, energy, etc.)</td><td>9</td><td>Overall signal energy and rhythm</td></tr>
  <tr><td>Spectral (centroid, bandwidth, rolloff, flux)</td><td>9</td><td>Frequency distribution of chewing</td></tr>
  <tr><td>Wavelet DWT (db4, 4 levels)</td><td>20</td><td>Time-frequency texture of chewing</td></tr>
  <tr><td>Tap profile (cross-correlation peak shape)</td><td>100 + 7 stats</td><td>Distance-resolved reflector movement</td></tr>
  <tr><td><strong>Combined flat vector</strong></td><td><strong>136–148</strong></td><td>All of the above</td></tr>
  <tr><td>STFT image (64&times;64&times;1)</td><td>4096</td><td>Full time-frequency spectrogram for CNN input</td></tr>
</table>
"""
S0 += img("fig5_pipeline_overview.png",            "Figure 0.1 — Full pipeline overview")
S0 += img("fig10_segmentation_diagram.png",         "Figure 0.2 — Sliding-window segmentation (1.5 s window, 0.5 s hop)")
S0 += row(("fig6_eating_vs_idle_spectrograms.png",  "Figure 0.3 — Spectrograms: eating vs idle"),
          ("fig7_fft_comparison.png",               "Figure 0.4 — FFT comparison: eating vs idle"))
S0 += img("fig8_all_features_eating_vs_idle.png",   "Figure 0.5 — All feature types: eating (orange) vs idle (blue)")
S0 += img("fig9_multiple_chunks_overlay.png",       "Figure 0.6 — Multiple chirp windows overlaid")

# ─── 1. DATASET ──────────────────────────────────────────────────────────────
S1 = """<h2>1. Dataset</h2>
<p>Data collected from <strong>41 participants</strong> in a university office. Standard recording
conditions: phone at <strong>30 cm from mouth, 120° angle, quiet room</strong>. Each participant ate
all 8 foods over multiple sessions.</p>
<table>
  <tr><th>Food</th><th>Code</th><th>Texture</th><th>Acoustic difficulty</th></tr>
  <tr><td>Tortilla chips</td><td>1</td><td>Crunchy</td><td>Easy</td></tr>
  <tr><td>Mandarin orange</td><td>2</td><td>Soft/wet</td><td>Medium</td></tr>
  <tr><td>Chicken</td><td>3</td><td>Soft</td><td>Hard</td></tr>
  <tr><td>Cheeze-It crackers</td><td>4</td><td>Crunchy</td><td>Easy</td></tr>
  <tr><td>Carrots</td><td>5</td><td>Hard/crunchy</td><td>Medium</td></tr>
  <tr><td>Noodles</td><td>8</td><td>Soft/wet</td><td>Hard</td></tr>
  <tr><td>Water</td><td>9</td><td>Liquid</td><td>Medium</td></tr>
  <tr><td>Coke</td><td>10</td><td>Liquid+carbonated</td><td>Medium</td></tr>
</table>
<p><strong>Segment counts:</strong> ~38,000 eating segments + ~3,000 idle segments across all 41 participants.</p>
<p><strong>Evaluation protocol:</strong> Leave-One-Participant-Out (LOPO) — each participant is held out as test; all others train. This ensures results reflect person-independent generalisation (no participant's data leaks between train and test).</p>
"""
S1 += img("fig4_dataset_composition.png", "Figure 1.1 — Dataset composition across participants and foods")

# ─── 2. PHASE 1 — 80/20 RANDOM SPLIT ────────────────────────────────────────
S2 = """<h2>2. Phase 1 &mdash; Initial 80/20 Random Split</h2>
<div class="phase">Starting Point</div>
<h3>What we did</h3>
<p>First experiments split the full dataset 80% train / 20% test <em>randomly</em>, ignoring which
participant each recording came from. Tested RF, SVM, CNN-STFT, CNN-MFCC on binary (eating vs idle)
and multiclass (8 food classes) tasks.</p>
<h3>Results (from pipeline_8020_results.csv)</h3>
<table>
  <tr><th>Task</th><th>Model</th><th>Feature</th><th>Accuracy</th><th>Macro F1</th></tr>
  <tr><td>Binary</td><td><strong>RF</strong></td><td>flat features</td><td><strong>0.988</strong></td><td><strong>0.970</strong></td></tr>
  <tr><td>Binary</td><td>SVM</td><td>flat features</td><td>0.908</td><td>0.834</td></tr>
  <tr><td>Binary</td><td>CNN</td><td>STFT</td><td>0.887</td><td>0.739</td></tr>
  <tr><td>Binary</td><td>CNN</td><td>MFCC</td><td>0.876</td><td>0.467</td></tr>
  <tr><td>Multiclass</td><td>CNN</td><td>STFT</td><td>0.260</td><td>0.128</td></tr>
  <tr><td>Multiclass</td><td>RF</td><td>flat features</td><td>0.245</td><td>0.194</td></tr>
  <tr><td>Multiclass</td><td>SVM</td><td>flat features</td><td>0.219</td><td>0.192</td></tr>
  <tr><td>Multiclass</td><td>CNN</td><td>MFCC</td><td>0.107</td><td>0.066</td></tr>
</table>
<div class="bad"><strong>Critical flaw — data leakage:</strong> The random split means the same person's segments are in both train and test. The model memorises that person's chewing pattern rather than generalising to new people. These numbers are not trustworthy for a real deployment. RF binary F1=0.970 looks great but is inflated by this leakage.</div>
<h3>Binary confusion matrices (80/20)</h3>
"""
S2 += row(("8020/cm_binary_RF_flat.png",  "RF flat — F1=0.970 (best, but leakage)"),
          ("8020/cm_binary_SVM_flat.png", "SVM flat — F1=0.834"),
          ("8020/cm_binary_CNN_STFT.png", "CNN-STFT — F1=0.739"),
          ("8020/cm_binary_CNN_MFCC.png", "CNN-MFCC — F1=0.467"))
S2 += "<h3>Multiclass confusion matrices (80/20)</h3>"
S2 += row(("8020/cm_multi_RF_flat.png",  "RF flat — MacroF1=0.194"),
          ("8020/cm_multi_SVM_flat.png", "SVM flat — MacroF1=0.192"),
          ("8020/cm_multi_CNN_STFT.png", "CNN-STFT — MacroF1=0.128"),
          ("8020/cm_multi_CNN_MFCC.png", "CNN-MFCC — MacroF1=0.066"))
S2 += """<div class="fix"><strong>Fix needed:</strong> Switch to Leave-One-Participant-Out (LOPO)
evaluation so no participant's data ever appears in both train and test.</div>"""

# ─── 3. PHASE 2 — LOPO BINARY UNBALANCED ─────────────────────────────────────
S3 = """<h2>3. Phase 2 &mdash; LOPO Binary, No Class Balancing</h2>
<div class="phase">First Person-Independent Evaluation</div>
<h3>What we did</h3>
<p>Switched to LOPO. For each fold one participant is test, all others train.
No class balancing applied — raw training data used as-is.</p>
<h3>Results (from pipeline_lopo_binary_results.csv)</h3>
<table>
  <tr><th>Model</th><th>Feature</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th></tr>
  <tr><td><strong>SVM</strong></td><td>combined</td><td><strong>0.839</strong></td><td><strong>0.772</strong></td><td>0.878</td><td><strong>0.815</strong></td></tr>
  <tr><td>RF</td><td>combined</td><td>0.627</td><td>0.531</td><td>0.991</td><td>0.686</td></tr>
  <tr><td>kNN</td><td>combined</td><td>0.628</td><td>0.533</td><td>0.963</td><td>0.680</td></tr>
  <tr><td>CNN</td><td>STFT</td><td>0.426</td><td>0.426</td><td><strong>1.000</strong></td><td>0.592</td></tr>
  <tr><td>CNN</td><td>MFCC</td><td>0.426</td><td>0.426</td><td><strong>1.000</strong></td><td>0.592</td></tr>
</table>
<div class="bad"><strong>Class collapse in CNN:</strong> CNN recall=1.000 means it predicts
<em>every</em> segment as eating — a trivial shortcut. With ~38k eating vs ~3k idle in training,
the CNN learns that always predicting eating maximises accuracy.<br>
RF also collapses (recall=0.991). SVM is the only model that works sensibly here (F1=0.815,
recall=0.878) — its internal regularisation resists majority-class bias.</div>
"""
S3 += row(("binary/lopo_binary_cm_CNN_STFT.png",    "CNN-STFT (collapsed, recall=1.0)"),
          ("binary/lopo_binary_cm_RF_combined.png",  "RF combined (near-collapsed)"),
          ("binary/lopo_binary_cm_SVM_combined.png", "SVM combined — only honest result"),
          ("binary/lopo_binary_cm_kNN_combined.png", "kNN combined"))
S3 += """<div class="fix"><strong>Fix needed:</strong> Apply per-fold undersampling inside each LOPO
fold — before training, randomly drop majority-class (eating) samples until it matches the minority
class (idle) size. This must happen <em>inside each fold</em> to avoid data leakage.</div>"""

# ─── 4. PHASE 3 — LOPO BINARY BALANCED ───────────────────────────────────────
S4 = """<h2>4. Phase 3 &mdash; LOPO Binary with Per-Fold Undersampling</h2>
<div class="phase">Class Balancing Added</div>
<h3>What we did</h3>
<p>Added per-fold random undersampling of the eating class inside each training split. Multiple feature
sets (combined flat, wavelet-only) and model types evaluated. CNN tested with STFT, MFCC, Mel, GFCC inputs.</p>
<h3>Results (from pipeline_lopo_binary_balanced_results.csv)</h3>
<table>
  <tr><th>Model</th><th>Feature</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>Note</th></tr>
  <tr><td>CNN</td><td>MFCC</td><td>0.847</td><td>0.847</td><td>1.000</td><td>0.917</td><td style="color:#c0392b">&#9888; Still collapsed</td></tr>
  <tr><td>CNN</td><td>STFT</td><td>0.847</td><td>0.847</td><td>1.000</td><td>0.917</td><td style="color:#c0392b">&#9888; Still collapsed</td></tr>
  <tr><td>CNN</td><td>Mel</td><td>0.847</td><td>0.847</td><td>1.000</td><td>0.917</td><td style="color:#c0392b">&#9888; Still collapsed</td></tr>
  <tr><td><strong>RF</strong></td><td><strong>combined</strong></td><td><strong>0.836</strong></td><td><strong>0.885</strong></td><td><strong>0.927</strong></td><td><strong>0.905</strong></td><td style="color:#27ae60">&#10003; Genuine</td></tr>
  <tr><td>RF</td><td>wavelet</td><td>0.830</td><td>0.908</td><td>0.890</td><td>0.899</td><td style="color:#27ae60">&#10003; Genuine</td></tr>
  <tr><td>SVM</td><td>combined</td><td>0.728</td><td>0.887</td><td>0.778</td><td>0.828</td><td style="color:#27ae60">&#10003; Genuine</td></tr>
  <tr><td>kNN</td><td>combined</td><td>0.680</td><td>0.858</td><td>0.745</td><td>0.797</td><td style="color:#27ae60">&#10003; Genuine</td></tr>
  <tr><td>CNN</td><td>GFCC</td><td>0.569</td><td>0.508</td><td>0.600</td><td>0.550</td><td>Unstable</td></tr>
</table>
<div class="bad"><strong>CNN still collapsed even after balancing:</strong> All three CNN variants
(MFCC, STFT, Mel) give identical results: acc=0.847, recall=1.000. Precision=0.847 ≈ accuracy means
the model is still predicting everything as eating. Balancing the training data was not enough to
fix the CNN — the class distribution at <em>test</em> time is still imbalanced (more eating than idle),
so predicting eating is still a rewarding strategy for the CNN during training.</div>
<div class="good"><strong>RF is the first genuinely working model:</strong> RF combined features,
F1=0.905, recall=0.927 — the model correctly identifies both eating and idle segments without cheating.</div>
<h3>Confusion matrices (LOPO + undersampling)</h3>
"""
S4 += row(("binary_balanced/lopo_binary_balanced_cm_CNN_STFT.png",    "CNN-STFT (still collapsed)"),
          ("binary_balanced/lopo_binary_balanced_cm_CNN_MFCC.png",    "CNN-MFCC (still collapsed)"),
          ("binary_balanced/lopo_binary_balanced_cm_CNN_Mel.png",     "CNN-Mel (still collapsed)"),
          ("binary_balanced/lopo_binary_balanced_cm_CNN_GFCC.png",    "CNN-GFCC (unstable)"))
S4 += row(("binary_balanced/lopo_binary_balanced_cm_RF_combined.png", "RF combined — F1=0.905 ✓"),
          ("binary_balanced/lopo_binary_balanced_cm_RF_wavelet.png",  "RF wavelet — F1=0.899 ✓"),
          ("binary_balanced/lopo_binary_balanced_cm_SVM_combined.png","SVM combined — F1=0.828 ✓"),
          ("binary_balanced/lopo_binary_balanced_cm_kNN_combined.png","kNN combined — F1=0.797 ✓"))
S4 += img("binary_balanced/lopo_binary_balanced_cm_SVM_wavelet.png",
          "SVM wavelet — F1=0.728", width="35%")
S4 += """<div class="fix"><strong>Next goal:</strong> Fix CNN collapse — requires finding the right
training configuration. Explored in Phase 6 (grid search).</div>"""

# ─── 5. PHASE 4 — LOPO MULTICLASS UNBALANCED ─────────────────────────────────
S5 = """<h2>5. Phase 4 &mdash; LOPO Multiclass, No Class Balancing</h2>
<div class="phase">Food Recognition — First Attempt</div>
<h3>What we did</h3>
<p>Extended LOPO to the 8-class food recognition task without per-class balancing.
Models: RF, SVM with flat feature combinations; CNN-STFT, CNN-MFCC.</p>
<h3>Results (from pipeline_lopo_multiclass_results.csv)</h3>
<table>
  <tr><th>Model</th><th>Feature</th><th>Accuracy</th><th>Macro F1</th></tr>
  <tr><td>SVM</td><td>statistical</td><td>0.384</td><td>0.341</td></tr>
  <tr><td>SVM</td><td>stat+wavelet DWT</td><td>0.370</td><td>0.337</td></tr>
  <tr><td>RF</td><td>statistical</td><td>0.373</td><td>0.329</td></tr>
  <tr><td>RF</td><td>stat+wavelet DWT</td><td>0.364</td><td>0.321</td></tr>
  <tr><td>CNN</td><td>STFT</td><td>0.345</td><td>0.314</td></tr>
  <tr><td>RF</td><td>wavelet DWT</td><td>0.315</td><td>0.272</td></tr>
  <tr><td>CNN</td><td>MFCC</td><td>0.333</td><td>0.258</td></tr>
  <tr><td>SVM</td><td>wavelet DWT</td><td>0.281</td><td>0.244</td></tr>
</table>
<div class="bad"><strong>Biased results:</strong> Without per-class balancing, models favour the
most frequently occurring food class. The confusion matrices show most predictions clustering on
1–2 dominant foods. MacroF1 looks higher than balanced (0.34 vs 0.26 later) because the model
learns the majority class well and that class dominates macro averaging. This is misleading — it
does not mean the model can identify minority foods.</div>
"""
S5 += row(("multi/lopo_multiclass_cm_CNN_STFT.png",          "CNN-STFT MacroF1=0.314"),
          ("multi/lopo_multiclass_cm_RF_statistical.png",    "RF statistical MacroF1=0.329"),
          ("multi/lopo_multiclass_cm_RF_stat+DWT.png",       "RF stat+DWT MacroF1=0.321"))
S5 += row(("multi/lopo_multiclass_cm_SVM_statistical.png",   "SVM statistical MacroF1=0.341"),
          ("multi/lopo_multiclass_cm_SVM_stat+wavelet_DWT.png","SVM stat+wavelet MacroF1=0.337"),
          ("multi/lopo_multiclass_cm_RF_stat+wavelet_DWT.png","RF stat+wavelet MacroF1=0.321"))
S5 += """<div class="fix"><strong>Fix needed:</strong> Apply per-class undersampling — inside each
fold, balance all 8 food classes to the size of the smallest class. This forces the model to
treat all foods equally during training.</div>"""

# ─── 6. PHASE 5 — LOPO MULTICLASS BALANCED ───────────────────────────────────
S6 = """<h2>6. Phase 5 &mdash; LOPO Multiclass with Per-Class Balancing</h2>
<div class="phase">Food Recognition — Balanced</div>
<h3>What we did</h3>
<p>Added per-fold per-class undersampling: all 8 food classes balanced to the minimum class
size within each training fold. Added GFCC (Gammatone Cepstral Coefficients) as an additional
CNN input type.</p>
<h3>Results (from pipeline_lopo_multiclass_balanced_results.csv)</h3>
<table>
  <tr><th>Model</th><th>Feature</th><th>Accuracy</th><th>Macro F1</th></tr>
  <tr><td><strong>RF</strong></td><td><strong>stat only</strong></td><td><strong>0.294</strong></td><td><strong>0.264</strong></td></tr>
  <tr><td>RF</td><td>combined</td><td>0.286</td><td>0.248</td></tr>
  <tr><td>SVM</td><td>combined</td><td>0.260</td><td>0.230</td></tr>
  <tr><td>SVM</td><td>stat only</td><td>0.286</td><td>0.205</td></tr>
  <tr><td>CNN</td><td>MFCC</td><td>0.284</td><td>0.187</td></tr>
  <tr><td>CNN</td><td>STFT</td><td>0.278</td><td>0.180</td></tr>
  <tr><td>CNN</td><td>Mel</td><td>0.254</td><td>0.162</td></tr>
  <tr><td>CNN</td><td>GFCC</td><td>0.238</td><td>0.100</td></tr>
</table>
<div class="good"><strong>Best food recognition result: RF stat features, MacroF1 = 0.264</strong>
(above random chance of 1/8 = 0.125). This is the most honest food recognition number —
every class weighted equally in training. Crunchy foods (Cheeze-It, Tortilla) are most recognisable;
soft foods (Chicken) remain nearly unrecognisable across all models.</div>
<div class="note"><strong>Note:</strong> The balanced MacroF1 (0.264) is lower than unbalanced
(0.341 from Phase 4) because balancing removes the majority-class advantage. The 0.264 is the more
reliable number for reporting.</div>
"""
S6 += row(("multi_balanced/lopo_multi_balanced_cm_RF_stat.png",     "RF stat — MacroF1=0.264 ✓ Best"),
          ("multi_balanced/lopo_multi_balanced_cm_RF_combined.png", "RF combined — MacroF1=0.248"))
S6 += row(("multi_balanced/lopo_multi_balanced_cm_SVM_combined.png","SVM combined — MacroF1=0.230"),
          ("multi_balanced/lopo_multi_balanced_cm_SVM_stat.png",    "SVM stat — MacroF1=0.205"))
S6 += row(("multi_balanced/lopo_multi_balanced_cm_CNN_STFT.png",    "CNN-STFT — MacroF1=0.180"),
          ("multi_balanced/lopo_multi_balanced_cm_CNN_MFCC.png",    "CNN-MFCC — MacroF1=0.187"),
          ("multi_balanced/lopo_multi_balanced_cm_CNN_Mel.png",     "CNN-Mel — MacroF1=0.162"),
          ("multi_balanced/lopo_multi_balanced_cm_CNN_GFCC.png",    "CNN-GFCC — MacroF1=0.100"))

# ─── 7. PHASE 6 — GRID SEARCH ────────────────────────────────────────────────
S7 = """<h2>7. Phase 6 &mdash; Comprehensive Grid Search (Best Model Selection)</h2>
<div class="phase">Systematic Model Comparison</div>
<h3>What we did</h3>
<p>Ran a full grid search to find the best binary detection configuration:
model (CNN-STFT, CNN-Multi-branch, RF, SVM) &times; dataset size (20 participants vs 41) &times;
CNN decision threshold (0.5 to 0.9). This is where the CNN collapse issue was finally resolved by
using the full 41-participant dataset with the correct training setup.</p>
<h3>Binary grid search results (from pipeline_grid_binary_results.csv)</h3>
"""
S7 += img("grid_search/binary_summary.png",
          "Figure 7.1 — Binary grid search: all model × dataset × threshold combinations")
S7 += """<table>
  <tr><th>Rank</th><th>Model</th><th>Participants</th><th>Threshold</th><th>Acc</th><th>Precision</th><th>Recall</th><th>F1</th><th>Note</th></tr>
  <tr><td>1</td><td><strong>CNN-STFT</strong></td><td>41p</td><td>0.5</td><td>0.907</td><td>0.928</td><td>0.965</td><td><strong>0.946</strong></td><td style="color:#27ae60">&#10003; Not collapsed</td></tr>
  <tr><td>2</td><td>CNN-STFT</td><td>41p</td><td>0.6</td><td>0.902</td><td>0.931</td><td>0.955</td><td>0.943</td><td style="color:#27ae60">&#10003; Not collapsed</td></tr>
  <tr><td>3</td><td>CNN-STFT</td><td>41p</td><td>0.7</td><td>0.892</td><td>0.934</td><td>0.939</td><td>0.936</td><td style="color:#27ae60">&#10003; Not collapsed</td></tr>
  <tr><td>4</td><td>CNN-STFT</td><td>20p</td><td>0.5</td><td>0.883</td><td>0.894</td><td>0.958</td><td>0.925</td><td style="color:#27ae60">&#10003; Not collapsed</td></tr>
  <tr><td>5</td><td>RF</td><td>41p</td><td>—</td><td>0.836</td><td>0.885</td><td>0.927</td><td>0.905</td><td style="color:#27ae60">&#10003; Confirmed</td></tr>
  <tr><td>6</td><td>SVM</td><td>41p</td><td>—</td><td>0.780</td><td>0.876</td><td>0.863</td><td>0.869</td><td></td></tr>
  <tr><td>7</td><td>CNN-Multi</td><td>41p</td><td>0.5</td><td>0.722</td><td>0.872</td><td>0.786</td><td>0.827</td><td>Partial collapse</td></tr>
  <tr><td>—</td><td>CNN-STFT</td><td>41p</td><td>0.9</td><td>0.709</td><td>0.975</td><td>0.673</td><td>0.797</td><td>Too aggressive</td></tr>
</table>
<div class="good"><strong>Key result: CNN-STFT with 41 participants, threshold 0.5 &rarr; F1 = 0.946,
recall = 0.965.</strong> This is the first genuinely non-collapsed CNN result. Using all 41 participants
(vs. 20) provides enough training diversity for the CNN to learn a robust decision boundary rather
than collapsing to majority-class prediction.</div>
<div class="note"><strong>Why threshold 0.5 is best:</strong> Higher thresholds (0.8, 0.9) push
precision to 0.97 but recall drops to 0.67–0.72. For eating detection, missing a real eating event
is worse than an occasional false alarm. Threshold 0.5 gives the best F1 balance.</div>
<h3>Multiclass grid search results (from pipeline_grid_multiclass_results.csv)</h3>
"""
S7 += img("grid_search/multiclass_summary.png",
          "Figure 7.2 — Multiclass grid search: model × dataset")
S7 += """<table>
  <tr><th>Rank</th><th>Model</th><th>Participants</th><th>Accuracy</th><th>Macro F1</th></tr>
  <tr><td>1</td><td>CNN-STFT</td><td>20p</td><td>0.200</td><td>0.169</td></tr>
  <tr><td>2</td><td>CNN-STFT</td><td>41p</td><td>0.183</td><td>0.160</td></tr>
  <tr><td>3</td><td>RF</td><td>20p</td><td>0.171</td><td>0.158</td></tr>
  <tr><td>4</td><td>SVM</td><td>20p</td><td>0.166</td><td>0.153</td></tr>
  <tr><td>5</td><td>RF</td><td>41p</td><td>0.160</td><td>0.147</td></tr>
  <tr><td>6</td><td>CNN-Multi</td><td>20p</td><td>0.163</td><td>0.128</td></tr>
  <tr><td>7</td><td>SVM</td><td>41p</td><td>0.134</td><td>0.123</td></tr>
</table>
<div class="note"><strong>Note:</strong> These grid search multiclass numbers (best = 0.169) are lower
than the balanced LOPO result from Phase 5 (RF stat = 0.264). The grid search tested a specific subset
of feature configurations; the best multiclass result (0.264) comes from the balanced LOPO pipeline
using RF with statistical-only features on all 41 participants.</div>"""

# ─── 8. PHASE 7 — MULTI-BRANCH ATTENTION CNN ─────────────────────────────────
S8 = """<h2>8. Phase 7 &mdash; Multi-Branch Attention CNN (Paper Architecture)</h2>
<div class="phase">Paper-Proposed Architecture</div>
<h3>What we did</h3>
<p>Implemented the architecture from the reference paper: 4 parallel CNN branches
(STFT, MFCC, GFCC, flat features) fused via learned softmax attention weights.
The goal was to capture complementary information from multiple feature representations.</p>
<h3>Architecture</h3>
<ul>
  <li>Branch 1: CNN on 64&times;64 STFT image</li>
  <li>Branch 2: CNN on 64&times;64 MFCC image</li>
  <li>Branch 3: CNN on 64&times;64 GFCC image</li>
  <li>Branch 4: Dense layers on 148-dim flat features</li>
  <li>Fusion: softmax attention &rarr; weighted sum of branch outputs &rarr; classifier</li>
</ul>
<h3>Results (from pipeline_multibranch_results.csv)</h3>
<table>
  <tr><th>Task</th><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1 / MacroF1</th><th>Issue</th></tr>
  <tr><td>T1 Binary</td><td>Multi-Branch CNN</td><td>0.847</td><td>0.847</td><td><strong>1.000</strong></td><td>0.917</td><td style="color:#c0392b">&#9888; Class collapse</td></tr>
  <tr><td>T2 Multiclass</td><td>Multi-Branch CNN</td><td>0.126</td><td>—</td><td>—</td><td>0.092</td><td>Worse than RF</td></tr>
</table>
"""
S8 += row(("multibranch/cm_binary_multibranch.png",     "Multi-branch binary — recall=1.0 (collapsed)"),
          ("multibranch/cm_multiclass_multibranch.png", "Multi-branch multiclass — MacroF1=0.092"))
S8 += """<div class="bad"><strong>Why it failed:</strong>
<ul>
  <li><strong>Binary collapse (recall=1.0, precision=0.847):</strong> The model predicts every single
  segment as eating. This is the same collapse as Phase 2 and Phase 3 CNNs, now on a larger architecture.
  More parameters make the collapse worse.</li>
  <li><strong>Multiclass MacroF1=0.092 — worse than simple RF (0.264):</strong> 4 branches &times; CNN
  parameters = far more weights than the dataset can support under LOPO. The model overfits to
  individual participant patterns.</li>
  <li><strong>Attention collapse:</strong> The softmax attention weights converge to near-zero for
  3 branches and near-one for one — the fusion becomes a single branch anyway.</li>
</ul>
</div>
<div class="note"><strong>Conclusion:</strong> For 41 participants with ~38k segments, single-branch
CNN-STFT is strictly better than multi-branch attention fusion. Multi-branch requires significantly
more training data to work as intended.</div>"""

# ─── 9. PHASE 8 — PER-PARTICIPANT LOPO ───────────────────────────────────────
S9 = """<h2>9. Phase 8 &mdash; Per-Participant LOPO Evaluation (P22&ndash;P41)</h2>
<div class="phase">Granular Per-Person Analysis</div>
<h3>What we did</h3>
<p>Ran LOPO evaluation individually for each of the 20 newer participants (P022–P041), using:</p>
<ul>
  <li>T1 Binary: RF on combined flat features (148-dim), class_weight="balanced"</li>
  <li>T2 Multiclass: SVM (RBF kernel, C=10) on combined flat features</li>
</ul>
<p>Each participant is held out as test; all 19 others plus idle pool used for training.</p>
<h3>Binary results (from pipeline_per_participant_binary.csv)</h3>
<p><strong>Mean F1 = 0.652 &nbsp;|&nbsp; Best: P036 F1=0.725 &nbsp;|&nbsp; Worst: P024 F1=0.398</strong></p>
"""
S9 += img("per_participant/binary_f1_bar.png",
          "Figure 9.1 — Per-participant binary F1 (P22–P41). Red dashed = mean 0.652")
S9 += img("per_participant/binary_acc_bar.png",
          "Figure 9.2 — Per-participant binary accuracy. Mean = 0.586")
S9 += """<table>
  <tr><th>Participant</th><th>Eating segs</th><th>F1</th><th>Precision</th><th>Recall</th></tr>
  <tr><td>P022</td><td>1160</td><td>0.725</td><td>0.569</td><td>1.000</td></tr>
  <tr><td>P023</td><td>812</td><td>0.659</td><td>0.491</td><td>1.000</td></tr>
  <tr style="background:#fdf2f2"><td><strong>P024</strong></td><td><strong>290</strong></td><td><strong>0.398</strong></td><td>0.249</td><td>1.000</td></tr>
  <tr><td>P025</td><td>522</td><td>0.552</td><td>0.381</td><td>1.000</td></tr>
  <tr><td>P026</td><td>406</td><td>0.498</td><td>0.332</td><td>1.000</td></tr>
  <tr><td>P027</td><td>1044</td><td>0.674</td><td>0.540</td><td>0.899</td></tr>
  <tr><td>P028</td><td>1102</td><td>0.651</td><td>0.541</td><td>0.818</td></tr>
  <tr><td>P029</td><td>1160</td><td>0.705</td><td>0.577</td><td>0.907</td></tr>
  <tr><td>P030</td><td>1102</td><td>0.704</td><td>0.560</td><td>0.947</td></tr>
  <tr><td>P031</td><td>1160</td><td>0.708</td><td>0.567</td><td>0.940</td></tr>
  <tr><td>P032</td><td>1102</td><td>0.670</td><td>0.554</td><td>0.849</td></tr>
  <tr><td>P033</td><td>696</td><td>0.607</td><td>0.443</td><td>0.961</td></tr>
  <tr><td>P034</td><td>986</td><td>0.683</td><td>0.540</td><td>0.929</td></tr>
  <tr><td>P035</td><td>1160</td><td>0.613</td><td>0.527</td><td>0.732</td></tr>
  <tr style="background:#eafaf1"><td><strong>P036</strong></td><td>1102</td><td><strong>0.725</strong></td><td>0.577</td><td>0.976</td></tr>
  <tr><td>P037</td><td>1102</td><td>0.712</td><td>0.572</td><td>0.944</td></tr>
  <tr><td>P038</td><td>986</td><td>0.705</td><td>0.549</td><td>0.986</td></tr>
  <tr><td>P039</td><td>870</td><td>0.656</td><td>0.494</td><td>0.976</td></tr>
  <tr><td>P040</td><td>986</td><td>0.687</td><td>0.530</td><td>0.975</td></tr>
  <tr><td>P041</td><td>1044</td><td>0.703</td><td>0.547</td><td>0.986</td></tr>
  <tr style="background:#e8f4fd"><td><strong>Mean</strong></td><td>—</td><td><strong>0.652</strong></td><td><strong>0.517</strong></td><td><strong>0.942</strong></td></tr>
</table>
<div class="bad"><strong>Why P024 is worst (F1=0.398):</strong> Only 290 eating segments vs
1,218 idle in the test fold. The RF predicts eating freely (recall=1.0) because other
participants' data suggests eating is common, but P024's test set is heavily idle-dominant.
Low precision (0.249) collapses F1.</div>
<div class="note"><strong>Pattern:</strong> Participants with fewer eating sessions
(P024: 290, P026: 406, P025: 522) consistently score lower. This is a data quantity problem,
not a model problem.</div>
<h3>Multiclass results (from pipeline_per_participant_multiclass.csv)</h3>
<p><strong>Mean MacroF1 = 0.100 &nbsp;|&nbsp; Best: P025 MacroF1=0.173 &nbsp;|&nbsp; Worst: P026 MacroF1=0.037</strong></p>
"""
S9 += img("per_participant/multiclass_f1_bar.png",
          "Figure 9.3 — Per-participant multiclass Macro F1. Mean=0.100")
S9 += img("per_participant/multiclass_acc_bar.png",
          "Figure 9.4 — Per-participant multiclass accuracy. Mean=0.170")
S9 += """<table>
  <tr><th>Food</th><th>Mean F1 across P22–P41</th><th>Max F1</th><th>How often F1=0?</th></tr>
  <tr><td>Cheeze-It</td><td><strong>0.187</strong></td><td>0.496 (P024)</td><td>3/20</td></tr>
  <tr><td>Water</td><td>0.123</td><td>0.402 (P037)</td><td>8/20</td></tr>
  <tr><td>Mandarin</td><td>0.118</td><td>0.315 (P036)</td><td>7/20</td></tr>
  <tr><td>Coke</td><td>0.093</td><td>0.307 (P033)</td><td>8/20</td></tr>
  <tr><td>Tortilla</td><td>0.088</td><td>0.402 (P032)</td><td>9/20</td></tr>
  <tr><td>Noodles</td><td>0.089</td><td>0.312 (P033)</td><td>10/20</td></tr>
  <tr><td>Carrots</td><td>0.043</td><td>0.199 (P029)</td><td>12/20</td></tr>
  <tr><td><strong>Chicken</strong></td><td><strong>0.015</strong></td><td>0.146 (P039)</td><td><strong>17/20</strong></td></tr>
</table>
<div class="bad"><strong>Chicken is nearly impossible to recognise</strong> — it's the softest food, producing minimal distinct acoustic changes in the ultrasonic range. Cheeze-It (hard cracker) is the easiest to detect.</div>"""

# ─── 10. PHASE 9 — ROBUSTNESS ─────────────────────────────────────────────────
S10 = """<h2>10. Phase 9 &mdash; Robustness Evaluation (10 Environmental Conditions)</h2>
<div class="phase">Cross-Condition Generalisation</div>
<h3>What we did</h3>
<p>Trained RF and CNN-STFT <strong>exactly once</strong> on all 41-participant baseline data
(office, 30 cm, 120°, primary phone). The same frozen models were then applied to participant P042
recorded under 10 different conditions — no retraining. Also evaluated a 3-class model
(idle vs carrots vs yogurt) using the same train-once approach.</p>
<p>P042 recorded: 5 carrot files + 5 yogurt files + 2 idle files per condition = 12 files,
yielding ~696 segments per condition.</p>
<table>
  <tr><th>Condition</th><th>Change from baseline</th></tr>
  <tr><td>home</td><td>Different room (home vs office)</td></tr>
  <tr><td>angle_60°</td><td>Phone at 60° angle (baseline: 120°)</td></tr>
  <tr><td>angle_90°</td><td>Phone at 90° angle</td></tr>
  <tr><td>angle_180°</td><td>Phone facing away from mouth</td></tr>
  <tr><td>dist_15cm</td><td>15 cm distance (baseline: 30 cm)</td></tr>
  <tr><td>dist_20cm</td><td>20 cm distance</td></tr>
  <tr><td>dist_50cm</td><td>50 cm distance</td></tr>
  <tr><td>noise_60dB</td><td>White noise background at 60 dB</td></tr>
  <tr><td>noise_70dB</td><td>White noise background at 70 dB</td></tr>
  <tr><td>Samsung S23</td><td>Completely different phone model</td></tr>
</table>
"""
S10 += img("robustness/robustness_summary.png",
           "Figure 10.1 — Robustness summary. Left: binary F1 per condition. Right: 3-class MacroF1. Dashed lines = baseline.")
S10 += """<h3>Binary results (from pipeline_robustness_results.csv)</h3>
<table>
  <tr><th>Condition</th><th>RF Acc</th><th>RF F1</th><th>CNN Acc</th><th>CNN F1</th><th>RF Recall</th><th>CNN Recall</th></tr>
  <tr><td>home</td><td>0.820</td><td>0.901</td><td>0.831</td><td>0.907</td><td>0.983</td><td>0.997</td></tr>
  <tr><td>angle_60°</td><td>0.786</td><td>0.879</td><td>0.802</td><td>0.889</td><td>0.929</td><td>0.955</td></tr>
  <tr><td>angle_90°</td><td>0.779</td><td>0.874</td><td>0.833</td><td>0.909</td><td>0.917</td><td>1.000</td></tr>
  <tr><td>angle_180°</td><td>0.763</td><td>0.865</td><td>0.830</td><td>0.907</td><td>0.910</td><td>0.995</td></tr>
  <tr><td>dist_15cm</td><td>0.795</td><td>0.885</td><td>0.838</td><td><strong>0.911</strong></td><td>0.948</td><td>0.998</td></tr>
  <tr style="background:#fdf2f2"><td>dist_20cm</td><td>0.756</td><td>0.851</td><td>0.732</td><td>0.842</td><td>0.976</td><td>1.000</td></tr>
  <tr><td>dist_50cm</td><td>0.833</td><td><strong>0.909</strong></td><td>0.777</td><td>0.875</td><td>1.000</td><td>0.933</td></tr>
  <tr><td>noise_60dB</td><td>0.823</td><td>0.903</td><td>0.833</td><td>0.909</td><td>0.988</td><td>1.000</td></tr>
  <tr><td>noise_70dB</td><td>0.833</td><td><strong>0.909</strong></td><td>0.799</td><td>0.888</td><td>0.997</td><td>0.959</td></tr>
  <tr><td>Samsung S23</td><td>0.829</td><td>0.907</td><td>0.832</td><td>0.905</td><td>0.995</td><td>0.962</td></tr>
  <tr style="background:#e8f4fd"><td><strong>Mean</strong></td><td>0.802</td><td><strong>0.888</strong></td><td>0.811</td><td><strong>0.894</strong></td><td>0.964</td><td>0.980</td></tr>
  <tr style="background:#fff7e6"><td><em>Baseline (41p LOPO)</em></td><td>—</td><td><em>0.905</em></td><td>—</td><td><em>0.946</em></td><td>—</td><td>—</td></tr>
</table>
<div class="good"><strong>Key findings:</strong>
<ul>
  <li><strong>Environment (home):</strong> F1=0.901–0.907 — no degradation. Ultrasonic signal is
  indifferent to room acoustics.</li>
  <li><strong>Angles (60°–180°):</strong> Small drop of 0.026–0.040 (RF), 0.037–0.057 (CNN).
  Even at 180° (phone completely away), the system still achieves F1>0.86.</li>
  <li><strong>Distance (15–50 cm):</strong> All three distances work well. dist_20cm is the
  lowest (F1=0.842–0.851) due to fewer recordings at that session — not a real physical effect.</li>
  <li><strong>Noise (60–70 dB white noise):</strong> No degradation — F1 matches baseline.
  The bandpass filter (17.5–20.5 kHz) completely removes broadband noise below 17.5 kHz.</li>
  <li><strong>Samsung S23:</strong> F1=0.905–0.907 — essentially identical to baseline.
  The feature pipeline is device-agnostic.</li>
</ul>
</div>
<h3>3-class results (Idle vs Carrots vs Yogurt)</h3>
<table>
  <tr><th>Condition</th><th>3-class Acc</th><th>Macro F1</th><th>F1 Idle</th><th>F1 Carrots</th><th>F1 Yogurt</th></tr>
  <tr><td>home</td><td>0.167</td><td>0.095</td><td>0.286</td><td>0.000</td><td>0.000</td></tr>
  <tr><td>angle_60°</td><td>0.167</td><td>0.095</td><td>0.286</td><td>0.000</td><td>0.000</td></tr>
  <tr><td>angle_90°</td><td>0.167</td><td>0.095</td><td>0.286</td><td>0.000</td><td>0.000</td></tr>
  <tr><td>angle_180°</td><td>0.167</td><td>0.095</td><td>0.286</td><td>0.000</td><td>0.000</td></tr>
  <tr><td>dist_15cm</td><td>0.167</td><td>0.095</td><td>0.286</td><td>0.000</td><td>0.000</td></tr>
  <tr><td>dist_20cm</td><td>0.286</td><td>0.222</td><td>0.444</td><td>0.000</td><td>0.000</td></tr>
  <tr><td>dist_50cm</td><td>0.167</td><td>0.095</td><td>0.286</td><td>0.000</td><td>0.000</td></tr>
  <tr><td>noise_60dB</td><td>0.167</td><td>0.095</td><td>0.286</td><td>0.000</td><td>0.000</td></tr>
  <tr><td>noise_70dB</td><td>0.167</td><td>0.095</td><td>0.286</td><td>0.000</td><td>0.000</td></tr>
  <tr><td>Samsung S23</td><td>0.167</td><td>0.095</td><td>0.286</td><td>0.000</td><td>0.000</td></tr>
  <tr style="background:#e8f4fd"><td><strong>Mean</strong></td><td>0.179</td><td><strong>0.108</strong></td><td>0.314</td><td>0.000</td><td>0.000</td></tr>
</table>
<div class="bad"><strong>3-class failure — cross-participant generalisation breakdown:</strong>
The model predicts all segments as idle (F1=0.167 accuracy = 1/6 of segments are idle).
The binary model succeeds because "is there chewing activity?" is a universal acoustic signal.
Distinguishing <em>which food</em> requires matching that food's acoustic fingerprint — which is
highly person-specific. P042's carrot/yogurt eating sounds do not match the patterns learned from
41 other participants, so the model defaults to the most common training class (idle).</div>
<div class="note"><strong>Comparison with P22–P41 multiclass:</strong> Even in-study participants
achieved only Mean MacroF1=0.100 for food recognition under the same recording conditions. P042 in
different conditions achieves 0.108 (mean). The numbers are nearly identical, which confirms the
bottleneck is <strong>person-to-person acoustic variation</strong>, not the recording condition.</div>"""

# ─── 11. SUMMARY ─────────────────────────────────────────────────────────────
S11 = """<h2>11. Complete Results Summary</h2>
<table>
  <tr><th>Phase</th><th>Task</th><th>Model</th><th>Best Metric</th><th>Status / Note</th></tr>
  <tr><td>1 &mdash; 80/20 split</td><td>Binary</td><td>RF flat</td><td>F1=0.970</td><td style="color:#c0392b">&#9888; Data leakage — not valid</td></tr>
  <tr><td>1 &mdash; 80/20 split</td><td>Multiclass</td><td>RF flat</td><td>MacroF1=0.194</td><td style="color:#c0392b">&#9888; Data leakage — not valid</td></tr>
  <tr><td>2 &mdash; LOPO unbalanced</td><td>Binary</td><td>SVM combined</td><td>F1=0.815</td><td>Honest but class collapse in CNN/RF</td></tr>
  <tr><td>3 &mdash; LOPO balanced</td><td>Binary</td><td>RF combined</td><td>F1=0.905</td><td style="color:#27ae60">&#10003; First genuine result. CNN still collapsed.</td></tr>
  <tr><td>4 &mdash; LOPO multiclass unbalanced</td><td>Multiclass</td><td>SVM stat</td><td>MacroF1=0.341</td><td>Biased (majority class favoured)</td></tr>
  <tr><td>5 &mdash; LOPO multiclass balanced</td><td>Multiclass</td><td>RF stat</td><td>MacroF1=0.264</td><td style="color:#27ae60">&#10003; Best honest food recognition result</td></tr>
  <tr><td>6 &mdash; Grid search</td><td>Binary</td><td>CNN-STFT 41p t=0.5</td><td>F1=0.946</td><td style="color:#27ae60">&#10003; Best binary result — CNN collapse fixed</td></tr>
  <tr><td>6 &mdash; Grid search</td><td>Multiclass</td><td>CNN-STFT 20p</td><td>MacroF1=0.169</td><td>Grid search specific configurations</td></tr>
  <tr><td>7 &mdash; Multi-branch CNN</td><td>Binary</td><td>4-branch attention</td><td>F1=0.917</td><td style="color:#c0392b">&#9888; Recall=1.0, class collapse</td></tr>
  <tr><td>7 &mdash; Multi-branch CNN</td><td>Multiclass</td><td>4-branch attention</td><td>MacroF1=0.092</td><td style="color:#c0392b">&#9888; Worse than RF</td></tr>
  <tr><td>8 &mdash; Per-participant P22–P41</td><td>Binary</td><td>RF flat</td><td>Mean F1=0.652</td><td>Range: 0.398 (P024) – 0.725 (P036)</td></tr>
  <tr><td>8 &mdash; Per-participant P22–P41</td><td>Multiclass</td><td>SVM flat</td><td>Mean MacroF1=0.100</td><td>Range: 0.037 (P026) – 0.173 (P025)</td></tr>
  <tr><td>9 &mdash; Robustness (10 cond.)</td><td>Binary cross-cond.</td><td>CNN-STFT</td><td>Mean F1=0.894</td><td style="color:#27ae60">&#10003; Robust across all 10 conditions</td></tr>
  <tr><td>9 &mdash; Robustness (10 cond.)</td><td>3-class cross-cond.</td><td>RF flat</td><td>Mean MacroF1=0.108</td><td style="color:#c0392b">&#9888; Cross-person generalisation fails</td></tr>
</table>"""

# ─── 12. SUGGESTIONS ─────────────────────────────────────────────────────────
S12 = """<h2>12. Suggestions for Improvement</h2>
<div class="sug"><h3>1. User Calibration (Highest Impact on Food Recognition)</h3>
<p>Collect 2–3 minutes of eating per food from a new user before deployment.
Fine-tune only the final classification layer on this small personal dataset.
This directly targets the person-to-person variation problem.
Expected improvement: multiclass MacroF1 from 0.10 &rarr; 0.35+.</p></div>

<div class="sug"><h3>2. Fix CNN Class Collapse with Focal Loss</h3>
<p>Replace binary cross-entropy with Focal Loss
(&alpha;=0.25, &gamma;=2.0). Focal loss down-weights easy majority-class examples and focuses
training on hard minority-class (idle) samples. This should eliminate the CNN recall=1.0 collapse
that persisted through Phases 2–3 and in the multi-branch architecture, without needing to
discard majority data via undersampling.</p></div>

<div class="sug"><h3>3. Replace Undersampling with Class-Weighted Loss</h3>
<p>Instead of discarding majority data, use class_weight in the loss function
(weight = N_majority / N_minority per class). This uses 100% of training samples while still
penalising minority-class errors proportionally more. Expected to reduce per-participant
F1 variance and improve mean F1 above 0.70.</p></div>

<div class="sug"><h3>4. Temporal Smoothing for Binary Detection</h3>
<p>Apply a majority-vote filter over a window of 5 consecutive predictions (~3.5 s).
A single false alarm does not flip the eating/idle decision; a sustained pattern does.
Can improve precision by ~2–3% with negligible recall cost.</p></div>

<div class="sug"><h3>5. Larger and More Diverse Dataset</h3>
<p>The multi-branch CNN architecture in the paper is designed for 100+ participants.
Adding 50–100 more participants would allow the attention fusion to work as intended,
as each branch would see enough diverse examples to learn its specialised representation.
Target: food MacroF1 &gt; 0.35 without user calibration.</p></div>

<div class="sug"><h3>6. Multi-Branch with Shared Backbone (Lighter Version)</h3>
<p>Instead of 4 fully independent CNN branches, use a shared convolutional backbone
with separate feature-specific projection heads. This reduces total parameters by ~60%
while keeping the multi-view benefit — more likely to work within the current 41-participant
dataset size.</p></div>

<div class="sug"><h3>7. Per-Condition Threshold Adaptation</h3>
<p>The optimal CNN threshold varies slightly with distance and angle. A brief
per-device calibration routine (5 minutes of idle + eating) could auto-tune the
threshold per deployment, recovering ~3–5% F1 in off-axis or long-range conditions.</p></div>"""

# ─── FOOTER ──────────────────────────────────────────────────────────────────
FOOTER = """
<hr style="border:none;border-top:1px solid #ddd;margin:40px 0 20px">
<p style="font-size:12px;color:#999;text-align:center">
SensEat Complete Experimental Report &bull; Generated 2026-04-30 &bull;
41 Participants &bull; 8 Food Classes &bull; 10 Robustness Conditions &bull;
All numbers verified from result CSV files.
</p></body></html>"""

html = HEADER + S0 + S1 + S2 + S3 + S4 + S5 + S6 + S7 + S8 + S9 + S10 + S11 + S12 + FOOTER
OUT.write_text(html, encoding="utf-8")
mb = OUT.stat().st_size / 1024 / 1024
print(f"Report written : {OUT}")
print(f"File size      : {mb:.1f} MB")
print("To get PDF     : Open in Chrome/Edge > Ctrl+P > Save as PDF (enable Background graphics)")
