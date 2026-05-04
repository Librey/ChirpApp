"""
Automated verification: every number in the report vs. source CSVs.
"""
import pandas as pd
from pathlib import Path

base = Path(r"c:\Users\rlpra\AndroidStudioProjects\ChirpApp\analysis")
ok, err = [], []

def check(label, actual, expected, tol=0.002):
    if abs(actual - expected) <= tol:
        ok.append(f"  OK   {label}: {actual:.4f}")
    else:
        err.append(f"  FAIL {label}: report={expected:.4f}  CSV={actual:.4f}  diff={actual-expected:+.4f}")

# ── SECTION 2: 80/20 ──────────────────────────────────────────────────────────
d = pd.read_csv(base / "pipeline_8020_results.csv")
b = d[d.task == "binary"]
m = d[d.task == "multiclass"]

check("80/20 binary RF macro_f1",           float(b[b.model=="RF"].macro_f1),                                  0.970)
check("80/20 binary SVM macro_f1",          float(b[b.model=="SVM"].macro_f1),                                 0.834)
check("80/20 binary CNN-STFT macro_f1",     float(b[(b.model=="CNN")&(b.feature=="STFT")].macro_f1),           0.739)
check("80/20 binary CNN-MFCC macro_f1",     float(b[(b.model=="CNN")&(b.feature=="MFCC")].macro_f1),           0.467)
check("80/20 binary RF accuracy",           float(b[b.model=="RF"].accuracy),                                  0.988)
check("80/20 multi RF macro_f1",            float(m[m.model=="RF"].macro_f1),                                  0.194)
check("80/20 multi SVM macro_f1",           float(m[m.model=="SVM"].macro_f1),                                 0.192)
check("80/20 multi CNN-STFT macro_f1",      float(m[(m.model=="CNN")&(m.feature=="STFT")].macro_f1),           0.128)
check("80/20 multi CNN-MFCC macro_f1",      float(m[(m.model=="CNN")&(m.feature=="MFCC")].macro_f1),           0.066)

# ── SECTION 3: LOPO binary unbalanced ────────────────────────────────────────
d = pd.read_csv(base / "pipeline_lopo_binary_results.csv")
check("LOPO-unbal SVM F1",                  float(d[d.model=="SVM"].f1),                                       0.815)
check("LOPO-unbal SVM accuracy",            float(d[d.model=="SVM"].accuracy),                                 0.839)
check("LOPO-unbal SVM precision",           float(d[d.model=="SVM"].precision),                                0.772)
check("LOPO-unbal SVM recall",              float(d[d.model=="SVM"].recall),                                   0.878)
check("LOPO-unbal RF F1",                   float(d[d.model=="RF"].f1),                                        0.686)
check("LOPO-unbal RF recall",               float(d[d.model=="RF"].recall),                                    0.991)
check("LOPO-unbal kNN F1",                  float(d[d.model=="kNN"].f1),                                       0.680)
check("LOPO-unbal CNN-STFT F1",             float(d[(d.model=="CNN")&(d.feature=="STFT")].f1),                 0.592)
check("LOPO-unbal CNN-STFT recall=1.0",     float(d[(d.model=="CNN")&(d.feature=="STFT")].recall),             1.000)

# ── SECTION 4: LOPO binary balanced ──────────────────────────────────────────
d = pd.read_csv(base / "pipeline_lopo_binary_balanced_results.csv")
rf  = d[(d.model=="RF")  & (d.feature=="combined")]
rfw = d[(d.model=="RF")  & (d.feature=="wavelet")]
svm = d[(d.model=="SVM") & (d.feature=="combined")]
knn = d[(d.model=="kNN") & (d.feature=="combined")]
cnn = d[(d.model=="CNN") & (d.feature=="STFT")]
cnn_mfcc = d[(d.model=="CNN") & (d.feature=="MFCC")]
cnn_mel  = d[(d.model=="CNN") & (d.feature=="Mel")]
cnn_gfcc = d[(d.model=="CNN") & (d.feature=="GFCC")]

check("LOPO-bal RF combined F1",            float(rf.f1),                                                      0.905)
check("LOPO-bal RF combined accuracy",      float(rf.accuracy),                                                0.836)
check("LOPO-bal RF combined precision",     float(rf.precision),                                               0.885)
check("LOPO-bal RF combined recall",        float(rf.recall),                                                  0.927)
check("LOPO-bal RF wavelet F1",             float(rfw.f1),                                                     0.899)
check("LOPO-bal SVM combined F1",           float(svm.f1),                                                     0.828)
check("LOPO-bal kNN combined F1",           float(knn.f1),                                                     0.797)
check("LOPO-bal CNN-STFT F1 (collapsed)",   float(cnn.f1),                                                     0.917)
check("LOPO-bal CNN-STFT recall=1.0",       float(cnn.recall),                                                 1.000, tol=0.001)
check("LOPO-bal CNN-MFCC F1 (collapsed)",   float(cnn_mfcc.f1),                                                0.917)
check("LOPO-bal CNN-Mel F1 (collapsed)",    float(cnn_mel.f1),                                                 0.917)

# ── SECTION 5: LOPO multiclass unbalanced ────────────────────────────────────
d = pd.read_csv(base / "pipeline_lopo_multiclass_results.csv")
check("LOPO-mc-unbal SVM stat mf1",         float(d[(d.model=="SVM")&(d.feature=="statistical")].macro_f1),    0.341)
check("LOPO-mc-unbal SVM stat+wav mf1",     float(d[(d.model=="SVM")&(d.feature=="stat+wavelet_DWT")].macro_f1), 0.337)
check("LOPO-mc-unbal RF stat mf1",          float(d[(d.model=="RF") &(d.feature=="statistical")].macro_f1),    0.329)
check("LOPO-mc-unbal RF stat+wav mf1",      float(d[(d.model=="RF") &(d.feature=="stat+wavelet_DWT")].macro_f1), 0.321)
check("LOPO-mc-unbal CNN-STFT mf1",         float(d[(d.model=="CNN")&(d.feature=="STFT")].macro_f1),           0.314)

# ── SECTION 6: LOPO multiclass balanced ──────────────────────────────────────
d = pd.read_csv(base / "pipeline_lopo_multiclass_balanced_results.csv")
check("LOPO-mc-bal RF stat mf1",            float(d[(d.model=="RF") &(d.feature=="stat")].macro_f1),            0.264)
check("LOPO-mc-bal RF stat acc",            float(d[(d.model=="RF") &(d.feature=="stat")].accuracy),            0.294)
check("LOPO-mc-bal RF combined mf1",        float(d[(d.model=="RF") &(d.feature=="combined")].macro_f1),        0.248)
check("LOPO-mc-bal SVM combined mf1",       float(d[(d.model=="SVM")&(d.feature=="combined")].macro_f1),        0.230)
check("LOPO-mc-bal SVM stat mf1",           float(d[(d.model=="SVM")&(d.feature=="stat")].macro_f1),            0.205)
check("LOPO-mc-bal CNN-MFCC mf1",           float(d[(d.model=="CNN")&(d.feature=="MFCC")].macro_f1),            0.187)
check("LOPO-mc-bal CNN-STFT mf1",           float(d[(d.model=="CNN")&(d.feature=="STFT")].macro_f1),            0.180)
check("LOPO-mc-bal CNN-Mel mf1",            float(d[(d.model=="CNN")&(d.feature=="Mel")].macro_f1),             0.162)
check("LOPO-mc-bal CNN-GFCC mf1",           float(d[(d.model=="CNN")&(d.feature=="GFCC")].macro_f1),            0.100)

# ── SECTION 7: Grid search binary ────────────────────────────────────────────
d = pd.read_csv(base / "pipeline_grid_binary_results.csv")
check("Grid CNN-STFT 41p t0.5 F1",          float(d[(d.tag=="41p_t0.5")&(d.model=="CNN-STFT")].f1),            0.946)
check("Grid CNN-STFT 41p t0.5 recall",      float(d[(d.tag=="41p_t0.5")&(d.model=="CNN-STFT")].recall),        0.965)
check("Grid CNN-STFT 41p t0.5 acc",         float(d[(d.tag=="41p_t0.5")&(d.model=="CNN-STFT")].acc),           0.907)
check("Grid CNN-STFT 41p t0.5 precision",   float(d[(d.tag=="41p_t0.5")&(d.model=="CNN-STFT")].precision),     0.928)
check("Grid CNN-STFT 41p t0.6 F1",          float(d[(d.tag=="41p_t0.6")&(d.model=="CNN-STFT")].f1),            0.943)
check("Grid CNN-STFT 41p t0.7 F1",          float(d[(d.tag=="41p_t0.7")&(d.model=="CNN-STFT")].f1),            0.936)
check("Grid CNN-STFT 20p t0.5 F1",          float(d[(d.tag=="20p_t0.5")&(d.model=="CNN-STFT")].f1),            0.925)
check("Grid RF 41p F1",                     float(d[(d.tag=="41p")    &(d.model=="RF")].f1),                   0.905)
check("Grid SVM 41p F1",                    float(d[(d.tag=="41p")    &(d.model=="SVM")].f1),                  0.869)
check("Grid CNN-Multi 41p t0.5 F1",         float(d[(d.tag=="41p_t0.5")&(d.model=="CNN-Multi")].f1),           0.827)
check("Grid CNN-STFT 41p t0.9 F1",          float(d[(d.tag=="41p_t0.9")&(d.model=="CNN-STFT")].f1),            0.797)

# ── SECTION 7: Grid search multiclass ────────────────────────────────────────
d = pd.read_csv(base / "pipeline_grid_multiclass_results.csv")
check("Grid mc CNN-STFT 20p mf1",           float(d[(d.tag=="20p")&(d.model=="CNN-STFT")].macro_f1),           0.169)
check("Grid mc CNN-STFT 41p mf1",           float(d[(d.tag=="41p")&(d.model=="CNN-STFT")].macro_f1),           0.160)
check("Grid mc RF 20p mf1",                 float(d[(d.tag=="20p")&(d.model=="RF")].macro_f1),                 0.158)
check("Grid mc SVM 20p mf1",                float(d[(d.tag=="20p")&(d.model=="SVM")].macro_f1),                0.153)
check("Grid mc RF 41p mf1",                 float(d[(d.tag=="41p")&(d.model=="RF")].macro_f1),                 0.147)
check("Grid mc CNN-Multi 20p mf1",          float(d[(d.tag=="20p")&(d.model=="CNN-Multi")].macro_f1),          0.128)
check("Grid mc SVM 41p mf1",                float(d[(d.tag=="41p")&(d.model=="SVM")].macro_f1),                0.123)

# ── SECTION 8: Multibranch ────────────────────────────────────────────────────
d = pd.read_csv(base / "pipeline_multibranch_results.csv")
check("Multibranch T1 binary F1",           float(d[d.task=="T1_binary"].f1),                                  0.917)
check("Multibranch T1 recall=1.0",          float(d[d.task=="T1_binary"].recall),                              1.000, tol=0.001)
check("Multibranch T1 accuracy",            float(d[d.task=="T1_binary"].accuracy),                            0.847)
check("Multibranch T1 precision",           float(d[d.task=="T1_binary"].precision),                           0.847)
check("Multibranch T2 macro_f1",            float(d[d.task=="T2_multiclass"].macro_f1),                        0.092)
check("Multibranch T2 accuracy",            float(d[d.task=="T2_multiclass"].accuracy),                        0.126)

# ── SECTION 9: Per-participant binary ────────────────────────────────────────
d = pd.read_csv(base / "pipeline_per_participant_binary.csv")
check("PP binary mean F1",                  d.f1.mean(),                                                       0.652)
check("PP binary best F1 (P036)",           d.f1.max(),                                                        0.725)
check("PP binary worst F1 (P024)",          d.f1.min(),                                                        0.398)
check("PP binary best participant = 36",    float(d.loc[d.f1.idxmax(), "participant"]),                        36,   tol=0.1)
check("PP binary worst participant = 24",   float(d.loc[d.f1.idxmin(), "participant"]),                        24,   tol=0.1)
check("PP binary mean accuracy",            d.accuracy.mean(),                                                 0.586)
check("PP binary P024 n_eating",            float(d[d.participant==24].n_eating_test),                         290,  tol=1)
check("PP binary P024 precision",           float(d[d.participant==24].precision),                             0.249)
check("PP binary P036 recall",              float(d[d.participant==36].recall),                                0.976)

# ── SECTION 9: Per-participant multiclass ────────────────────────────────────
d = pd.read_csv(base / "pipeline_per_participant_multiclass.csv")
check("PP mc mean mf1",                     d.macro_f1.mean(),                                                 0.100)
check("PP mc best mf1 (P025)",              d.macro_f1.max(),                                                  0.173)
check("PP mc worst mf1 (P026)",             d.macro_f1.min(),                                                  0.037)
check("PP mc best participant = 25",        float(d.loc[d.macro_f1.idxmax(), "participant"]),                  25,   tol=0.1)
check("PP mc worst participant = 26",       float(d.loc[d.macro_f1.idxmin(), "participant"]),                  26,   tol=0.1)
check("PP mc Cheeze_It mean f1",            d["f1_Cheeze_It"].mean(),                                          0.187)
check("PP mc Chicken mean f1",              d["f1_Chicken"].mean(),                                            0.015)
check("PP mc Chicken F1=0 count (17/20)",   (d["f1_Chicken"] == 0).sum(),                                      17,   tol=0.1)
check("PP mc mean accuracy",                d.accuracy.mean(),                                                 0.170)

# ── SECTION 10: Robustness ────────────────────────────────────────────────────
d = pd.read_csv(base / "pipeline_robustness_results.csv")
check("Robust RF mean F1",                  d.RF_f1.mean(),                                                    0.888)
check("Robust CNN mean F1",                 d.CNN_f1.mean(),                                                   0.894)
check("Robust 3c mean mf1",                 d["3c_macro_f1"].mean(),                                           0.108)
check("Robust RF min = dist_20cm",          d.RF_f1.min(),                                                     0.851)
check("Robust CNN min = dist_20cm",         d.CNN_f1.min(),                                                    0.842)
check("Robust RF max (dist_50cm/noise_70)", d.RF_f1.max(),                                                     0.909)
check("Robust CNN max (dist_15cm)",         d.CNN_f1.max(),                                                    0.911)
check("Robust 3c carrots all zero",         d["3c_f1_carrots"].sum(),                                          0.0,  tol=0.001)
check("Robust 3c yogurt all zero",          d["3c_f1_yogurt"].sum(),                                           0.0,  tol=0.001)
check("Robust home RF F1",                  float(d[d.condition=="home"].RF_f1),                               0.901)
check("Robust home CNN F1",                 float(d[d.condition=="home"].CNN_f1),                              0.907)
check("Robust samsung RF F1",               float(d[d.condition=="samsung"].RF_f1),                            0.907)
check("Robust samsung CNN F1",              float(d[d.condition=="samsung"].CNN_f1),                           0.905)
check("Robust noise60 CNN F1",              float(d[d.condition=="noise_60db"].CNN_f1),                        0.909)
check("Robust noise70 RF F1",               float(d[d.condition=="noise_70db"].RF_f1),                         0.909)

# ── FIGURE FILE CHECK ─────────────────────────────────────────────────────────
fig_base = base / "figures"
required_figs = [
    "fig5_pipeline_overview.png", "fig10_segmentation_diagram.png",
    "fig6_eating_vs_idle_spectrograms.png", "fig7_fft_comparison.png",
    "fig8_all_features_eating_vs_idle.png", "fig9_multiple_chunks_overlay.png",
    "fig4_dataset_composition.png",
    "8020/cm_binary_RF_flat.png", "8020/cm_binary_SVM_flat.png",
    "8020/cm_binary_CNN_STFT.png", "8020/cm_binary_CNN_MFCC.png",
    "8020/cm_multi_RF_flat.png", "8020/cm_multi_SVM_flat.png",
    "8020/cm_multi_CNN_STFT.png", "8020/cm_multi_CNN_MFCC.png",
    "binary/lopo_binary_cm_CNN_STFT.png", "binary/lopo_binary_cm_RF_combined.png",
    "binary/lopo_binary_cm_SVM_combined.png", "binary/lopo_binary_cm_kNN_combined.png",
    "binary_balanced/lopo_binary_balanced_cm_CNN_STFT.png",
    "binary_balanced/lopo_binary_balanced_cm_CNN_MFCC.png",
    "binary_balanced/lopo_binary_balanced_cm_CNN_Mel.png",
    "binary_balanced/lopo_binary_balanced_cm_CNN_GFCC.png",
    "binary_balanced/lopo_binary_balanced_cm_RF_combined.png",
    "binary_balanced/lopo_binary_balanced_cm_RF_wavelet.png",
    "binary_balanced/lopo_binary_balanced_cm_SVM_combined.png",
    "binary_balanced/lopo_binary_balanced_cm_SVM_wavelet.png",
    "binary_balanced/lopo_binary_balanced_cm_kNN_combined.png",
    "multi/lopo_multiclass_cm_CNN_STFT.png",
    "multi/lopo_multiclass_cm_RF_statistical.png",
    "multi/lopo_multiclass_cm_RF_stat+DWT.png",
    "multi/lopo_multiclass_cm_SVM_statistical.png",
    "multi/lopo_multiclass_cm_SVM_stat+wavelet_DWT.png",
    "multi/lopo_multiclass_cm_RF_stat+wavelet_DWT.png",
    "multi_balanced/lopo_multi_balanced_cm_RF_stat.png",
    "multi_balanced/lopo_multi_balanced_cm_RF_combined.png",
    "multi_balanced/lopo_multi_balanced_cm_SVM_combined.png",
    "multi_balanced/lopo_multi_balanced_cm_SVM_stat.png",
    "multi_balanced/lopo_multi_balanced_cm_CNN_STFT.png",
    "multi_balanced/lopo_multi_balanced_cm_CNN_MFCC.png",
    "multi_balanced/lopo_multi_balanced_cm_CNN_Mel.png",
    "multi_balanced/lopo_multi_balanced_cm_CNN_GFCC.png",
    "grid_search/binary_summary.png", "grid_search/multiclass_summary.png",
    "multibranch/cm_binary_multibranch.png",
    "multibranch/cm_multiclass_multibranch.png",
    "per_participant/binary_f1_bar.png", "per_participant/binary_acc_bar.png",
    "per_participant/multiclass_f1_bar.png", "per_participant/multiclass_acc_bar.png",
    "robustness/robustness_summary.png",
]
missing_figs = []
for f in required_figs:
    p = fig_base / f
    if not p.exists():
        missing_figs.append(f"  MISSING FIGURE: {f}")
    else:
        ok.append(f"  OK   figure exists: {f}")

print(f"NUMBER CHECKS  : {len(ok)-len([x for x in ok if 'figure' in x])} passed")
print(f"FIGURE CHECKS  : {len(required_figs)-len(missing_figs)}/{len(required_figs)} found")
print(f"TOTAL FAILURES : {len(err) + len(missing_figs)}")
print()
if err:
    print("NUMBER FAILURES:")
    for x in err: print(x)
else:
    print("All number checks PASSED.")
print()
if missing_figs:
    print("MISSING FIGURES:")
    for x in missing_figs: print(x)
else:
    print("All figures found on disk.")
