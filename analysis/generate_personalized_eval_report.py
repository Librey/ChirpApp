"""
generate_personalized_eval_report.py
=====================================
PDF report for personalized model evaluation on users 001-020.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch
import seaborn as sns

SCRIPT_DIR = Path(__file__).resolve().parent
FIG_DIR    = SCRIPT_DIR / "figures" / "personalized_eval"
OUT_PDF    = SCRIPT_DIR / "personalized_eval_report.pdf"

BLUE   = "#2563EB"
GREEN  = "#16A34A"
ORANGE = "#EA580C"
GRAY   = "#6B7280"
LIGHT  = "#F3F4F6"
RED    = "#DC2626"

FOOD_NAMES = ["Tortilla","Fruit","Chicken","Cracker","Carrot",
              "Chocolate","Yogurt","Noodles","Water","Soft Drink"]

df = pd.read_csv(SCRIPT_DIR / "personalized_t2_results.csv")
df["user_label"] = df["user"].apply(lambda x: f"U{int(x):02d}")

mean_acc  = df["accuracy"].mean()
std_acc   = df["accuracy"].std()
mean_mf1  = df["macro_f1"].mean()
mean_wf1  = df["weighted_f1"].mean()
best_user = df.loc[df["accuracy"].idxmax()]
worst_user= df.loc[df["accuracy"].idxmin()]


# ─────────────────────────────────────────
# PAGE 1 — TITLE
# ─────────────────────────────────────────
def title_page(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    ax.add_patch(FancyBboxPatch((0.05, 0.72), 0.90, 0.20,
                                boxstyle="round,pad=0.01",
                                facecolor=BLUE, edgecolor="none",
                                transform=ax.transAxes))
    ax.text(0.50, 0.85, "SensEat — Personalized Model Evaluation",
            ha="center", va="center", fontsize=20, fontweight="bold",
            color="white", transform=ax.transAxes)
    ax.text(0.50, 0.77, "T2 Food Recognition · T3 Stress Prediction · Users 001–020",
            ha="center", va="center", fontsize=12, color="white",
            transform=ax.transAxes)

    metrics = [
        ("T2 Accuracy",    f"{mean_acc*100:.1f}%",     GREEN),
        ("T2 Macro F1",    f"{mean_mf1:.4f}",           BLUE),
        ("T2 Weighted F1", f"{mean_wf1:.4f}",           BLUE),
        ("Users Evaluated","20",                         ORANGE),
    ]
    for i, (label, value, color) in enumerate(metrics):
        x = 0.07 + i * 0.23
        ax.add_patch(FancyBboxPatch((x, 0.50), 0.19, 0.18,
                                    boxstyle="round,pad=0.01",
                                    facecolor=color, edgecolor="none",
                                    transform=ax.transAxes, alpha=0.9))
        ax.text(x+0.095, 0.61, value, ha="center", va="center",
                fontsize=17, fontweight="bold", color="white",
                transform=ax.transAxes)
        ax.text(x+0.095, 0.53, label, ha="center", va="center",
                fontsize=9, color="white", transform=ax.transAxes)

    # Setup box
    setup = (
        "Models: Pre-trained personalized .keras models (one per user per task)\n"
        "Features: STFT spectrogram (64×64×1) with per-user z-score normalization\n"
        "Data: feature_cache_multiclass_balanced_v2.npz  ·  Users 001–020  ·  16,124 segments\n"
        "T3: Inference run (raw predictions saved); stress labels needed for MAE"
    )
    ax.text(0.50, 0.40, setup, ha="center", va="center", fontsize=10,
            color="#374151", transform=ax.transAxes, linespacing=2.0)

    # Best / worst
    ax.text(0.25, 0.26, f"Best: User {best_user['user']}",
            ha="center", fontsize=11, fontweight="bold",
            color=GREEN, transform=ax.transAxes)
    ax.text(0.25, 0.20, f"Accuracy = {best_user['accuracy']*100:.1f}%",
            ha="center", fontsize=10, color=GREEN, transform=ax.transAxes)

    ax.text(0.75, 0.26, f"Weakest: User {worst_user['user']}",
            ha="center", fontsize=11, fontweight="bold",
            color=RED, transform=ax.transAxes)
    ax.text(0.75, 0.20, f"Accuracy = {worst_user['accuracy']*100:.1f}%",
            ha="center", fontsize=10, color=RED, transform=ax.transAxes)

    ax.text(0.50, 0.08, "SensEat — Smartphone-based Dietary & Stress Monitoring",
            ha="center", fontsize=9, color=GRAY, transform=ax.transAxes)

    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


# ─────────────────────────────────────────
# PAGE 2 — PER-USER BAR + SCATTER
# ─────────────────────────────────────────
def per_user_page(pdf):
    fig, axes = plt.subplots(1, 2, figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    fig.suptitle("T2 Personalized — Per-User Performance (20 users)",
                 fontsize=13, fontweight="bold", y=0.97, color=BLUE)

    # ── Left: accuracy bar ──
    ax = axes[0]
    colors = [GREEN if a >= 0.80 else (ORANGE if a >= 0.65 else RED)
              for a in df["accuracy"]]
    bars = ax.bar(df["user_label"], df["accuracy"]*100, color=colors,
                  edgecolor="white", linewidth=0.5)
    ax.axhline(y=mean_acc*100, color="black", linestyle="--",
               linewidth=1.5, label=f"Mean = {mean_acc*100:.1f}%")
    ax.set_xlabel("User ID", fontsize=10)
    ax.set_ylabel("Accuracy (%)", fontsize=10)
    ax.set_title("Accuracy per User", fontsize=11)
    ax.set_xticklabels(df["user_label"], rotation=45, ha="right", fontsize=7)
    ax.set_ylim(0, 108)
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    for bar, val in zip(bars, df["accuracy"]*100):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+1,
                f"{val:.0f}", ha="center", fontsize=6.5, color="#374151")

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor=GREEN,  label="≥ 80%"),
        Patch(facecolor=ORANGE, label="65–80%"),
        Patch(facecolor=RED,    label="< 65%"),
        plt.Line2D([0],[0], color="black", linestyle="--",
                   label=f"Mean {mean_acc*100:.1f}%"),
    ], fontsize=8)

    # ── Right: accuracy vs macro F1 scatter ──
    ax2 = axes[1]
    sc_colors = [GREEN if a >= 0.80 else (ORANGE if a >= 0.65 else RED)
                 for a in df["accuracy"]]
    ax2.scatter(df["accuracy"]*100, df["macro_f1"],
                c=sc_colors, s=90, edgecolors="white", linewidths=0.5, zorder=3)
    for _, row in df.iterrows():
        ax2.annotate(f"U{int(row['user']):02d}",
                     (row["accuracy"]*100, row["macro_f1"]),
                     textcoords="offset points", xytext=(4, 3), fontsize=7)
    ax2.set_xlabel("Accuracy (%)", fontsize=10)
    ax2.set_ylabel("Macro F1", fontsize=10)
    ax2.set_title("Accuracy vs Macro F1", fontsize=11)
    ax2.grid(True, alpha=0.2)
    ax2.spines[["top", "right"]].set_visible(False)

    stats = (
        f"Mean Acc  : {mean_acc*100:.2f}% (±{std_acc*100:.2f}%)\n"
        f"Mean MF1  : {mean_mf1:.4f}\n"
        f"Mean WF1  : {mean_wf1:.4f}\n"
        f"Best      : U{int(best_user['user']):02d} = {best_user['accuracy']*100:.1f}%\n"
        f"Weakest   : U{int(worst_user['user']):02d} = {worst_user['accuracy']*100:.1f}%"
    )
    ax2.text(0.97, 0.05, stats, transform=ax2.transAxes,
             fontsize=8.5, va="bottom", ha="right",
             bbox=dict(boxstyle="round,pad=0.4", facecolor=LIGHT, edgecolor="#D1D5DB"))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


# ─────────────────────────────────────────
# PAGE 3 — PER-CLASS F1 HEATMAP
# ─────────────────────────────────────────
def per_class_page(pdf):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    fig.suptitle("T2 Personalized — Per-Class F1 Score Heatmap (users 001–020)",
                 fontsize=13, fontweight="bold", y=0.97, color=BLUE)

    f1_cols = [f"f1_{n}" for n in FOOD_NAMES]
    heatmap_data = df[f1_cols].values
    heatmap_df   = pd.DataFrame(heatmap_data,
                                index=df["user_label"],
                                columns=FOOD_NAMES)

    sns.heatmap(heatmap_df, annot=True, fmt=".2f", cmap="RdYlGn",
                vmin=0, vmax=1, ax=ax, linewidths=0.3,
                cbar_kws={"label": "F1 Score"})
    ax.set_xlabel("Food Category", fontsize=10)
    ax.set_ylabel("User", fontsize=10)
    ax.set_title("F1 per User per Food Class", fontsize=11, pad=8)
    plt.xticks(rotation=40, ha="right", fontsize=9)
    plt.yticks(fontsize=8)

    # Class averages below
    class_means = heatmap_df.mean()
    ax.set_title(
        "F1 per User per Food Class\n"
        "Class means: " + "  ".join([f"{n}={v:.2f}" for n, v in class_means.items()]),
        fontsize=9, pad=8
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


# ─────────────────────────────────────────
# PAGE 4 — CONFUSION MATRIX
# ─────────────────────────────────────────
def cm_page(pdf):
    cm_path = FIG_DIR / "cm_t2_personalized.png"
    if not cm_path.exists():
        return
    fig, ax = plt.subplots(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    fig.suptitle("T2 Personalized — Aggregated Confusion Matrix (all 20 users)",
                 fontsize=13, fontweight="bold", y=0.97, color=BLUE)
    img = plt.imread(str(cm_path))
    ax.imshow(img); ax.axis("off")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


# ─────────────────────────────────────────
# PAGE 5 — METRICS TABLE
# ─────────────────────────────────────────
def table_page(pdf):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    fig.suptitle("T2 Personalized — Full Results Table (users 001–020)",
                 fontsize=13, fontweight="bold", y=0.97, color=BLUE)
    ax.axis("off")

    table_data = []
    for _, row in df.iterrows():
        table_data.append([
            f"U{int(row['user']):02d}",
            str(int(row["n_segments"])),
            f"{row['accuracy']*100:.1f}%",
            f"{row['macro_f1']:.4f}",
            f"{row['weighted_f1']:.4f}",
        ])
    # Add mean row
    table_data.append([
        "MEAN", str(int(df["n_segments"].mean())),
        f"{mean_acc*100:.1f}%",
        f"{mean_mf1:.4f}",
        f"{mean_wf1:.4f}",
    ])

    col_labels = ["User", "Segments", "Accuracy", "Macro F1", "Weighted F1"]
    tbl = ax.table(cellText=table_data, colLabels=col_labels,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.4, 1.55)

    # Header
    for j in range(len(col_labels)):
        tbl[(0, j)].set_facecolor(BLUE)
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")

    # Color rows by accuracy
    for i, row in enumerate(df.itertuples(), start=1):
        color = "#D1FAE5" if row.accuracy >= 0.80 else (
                "#FEF3C7" if row.accuracy >= 0.65 else "#FEE2E2")
        for j in range(len(col_labels)):
            tbl[(i, j)].set_facecolor(color)

    # Mean row
    n = len(df) + 1
    for j in range(len(col_labels)):
        tbl[(n, j)].set_facecolor("#DBEAFE")
        tbl[(n, j)].set_text_props(fontweight="bold", color=BLUE)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


# ─────────────────────────────────────────
# PAGE 6 — T3 PREDICTIONS
# ─────────────────────────────────────────
def t3_page(pdf):
    t3_path = SCRIPT_DIR / "personalized_t3_predictions.csv"
    if not t3_path.exists():
        return
    df3 = pd.read_csv(t3_path)
    df3["user_label"] = df3["user"].apply(lambda x: f"U{int(x):02d}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    fig.suptitle("T3 Stress Prediction — Raw Predictions per User (001–020)\n"
                 "(Ground truth stress labels not available — MAE cannot be computed)",
                 fontsize=12, fontweight="bold", y=0.97, color=ORANGE)

    # ── Left: predicted mean per user ──
    ax = axes[0]
    colors = [GREEN if m <= 2.0 else (ORANGE if m <= 3.5 else RED)
              for m in df3["pred_mean"]]
    ax.barh(df3["user_label"], df3["pred_mean"], color=colors,
            edgecolor="white", xerr=df3["pred_std"], capsize=3)
    ax.axvline(x=3.0, color=GRAY, linestyle="--", linewidth=1, label="Midpoint (3.0)")
    ax.set_xlabel("Predicted Stress Level (1–5)", fontsize=10)
    ax.set_title("Mean Predicted Stress per User", fontsize=11)
    ax.set_xlim(0, 5.5)
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    # ── Right: distribution scatter ──
    ax2 = axes[1]
    ax2.scatter(range(len(df3)), df3["pred_mean"], c=colors,
                s=80, zorder=3, edgecolors="white")
    ax2.errorbar(range(len(df3)), df3["pred_mean"], yerr=df3["pred_std"],
                 fmt="none", color=GRAY, alpha=0.5, capsize=3)
    ax2.axhline(y=3.0, color=GRAY, linestyle="--", linewidth=1)
    ax2.set_xticks(range(len(df3)))
    ax2.set_xticklabels(df3["user_label"], rotation=45, ha="right", fontsize=7)
    ax2.set_ylabel("Predicted Stress Level", fontsize=10)
    ax2.set_title("Predicted Stress Distribution per User", fontsize=11)
    ax2.set_ylim(0, 6)
    ax2.grid(True, alpha=0.2)
    ax2.spines[["top", "right"]].set_visible(False)

    note = ("Note: To compute T3 MAE/RMSE, provide stress_labels.npz\n"
            "with keys: 'user_ids' and 'stress_labels' (1-5 scale)")
    fig.text(0.5, 0.02, note, ha="center", fontsize=9,
             color=ORANGE, style="italic")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


# ─────────────────────────────────────────
# GENERATE
# ─────────────────────────────────────────
print("Generating personalized evaluation report ...")
with PdfPages(OUT_PDF) as pdf:
    title_page(pdf)
    per_user_page(pdf)
    per_class_page(pdf)
    cm_page(pdf)
    table_page(pdf)
    t3_page(pdf)

    d = pdf.infodict()
    d["Title"]   = "SensEat Personalized Model Evaluation Report"
    d["Subject"] = "T2 & T3 Evaluation — Users 001-020"

print(f"Report saved: {OUT_PDF}")
