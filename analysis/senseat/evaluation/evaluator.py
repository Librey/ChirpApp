"""
senseat/evaluation/evaluator.py
================================
Evaluation utilities:
  - Confusion matrix plots
  - ROC-AUC curve
  - PR-AUC curve
  - Results CSV saving
  - Per-participant breakdown
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (confusion_matrix, roc_curve, auc,
                              precision_recall_curve, average_precision_score)

from senseat.data.loader import FOOD_NAMES

FIG_DIR = Path(__file__).resolve().parent.parent.parent / "figures"
FIG_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────
# CONFUSION MATRIX
# ─────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, label_names, title="Confusion Matrix",
                          filename="confusion_matrix.png"):
    """
    Plot normalized confusion matrix (percentage per row).
    label_names: list of class name strings in order of class indices.
    """
    cm         = confusion_matrix(y_true, y_pred)
    cm_pct     = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(max(8, len(label_names)), max(6, len(label_names) - 1)))
    sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=label_names, yticklabels=label_names, ax=ax)
    ax.set_title(title)
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    out = FIG_DIR / filename
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"  [eval] Confusion matrix saved → {out.name}")


# ─────────────────────────────────────────
# ROC CURVE
# ─────────────────────────────────────────

def plot_roc_curve(y_true, y_prob, title="ROC Curve", filename="roc_curve.png"):
    """Binary ROC-AUC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc     = auc(fpr, tpr)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, lw=2, label=f"ROC AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc='lower right')
    plt.tight_layout()

    out = FIG_DIR / filename
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"  [eval] ROC curve saved → {out.name}")
    return roc_auc


# ─────────────────────────────────────────
# PR CURVE
# ─────────────────────────────────────────

def plot_pr_curve(y_true, y_prob, title="Precision-Recall Curve",
                  filename="pr_curve.png"):
    """Binary Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, lw=2, label=f"PR AUC = {pr_auc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend(loc='upper right')
    plt.tight_layout()

    out = FIG_DIR / filename
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"  [eval] PR curve saved → {out.name}")
    return pr_auc


# ─────────────────────────────────────────
# RESULTS TABLE
# ─────────────────────────────────────────

def save_results(rows, out_csv_path):
    """Save list of result dicts to CSV, sorted by F1."""
    df = pd.DataFrame(rows)
    if 'f1' in df.columns:
        df = df.sort_values('f1', ascending=False)
    df.to_csv(out_csv_path, index=False)
    print(f"\n  [eval] Results saved → {out_csv_path}")
    print(df.to_string(index=False))
    return df


# ─────────────────────────────────────────
# PER-PARTICIPANT BREAKDOWN
# ─────────────────────────────────────────

def per_participant_report(y_true, y_pred, participant_ids, is_binary=True):
    """
    Print accuracy per participant — useful for LOPO analysis.
    Shows which participants are hardest to classify.
    """
    unique_parts = np.unique(participant_ids)
    rows = []
    for p in unique_parts:
        mask   = participant_ids == p
        yt, yp = y_true[mask], y_pred[mask]
        acc    = np.mean(yt == yp)
        f1     = 0.0
        try:
            from sklearn.metrics import f1_score
            avg = 'binary' if is_binary else 'weighted'
            f1  = f1_score(yt, yp, average=avg, zero_division=0)
        except Exception:
            pass
        rows.append({"participant": p, "n_segments": int(mask.sum()),
                     "accuracy": round(acc, 4), "f1": round(f1, 4)})

    df = pd.DataFrame(rows).sort_values('accuracy')
    print("\n  ── Per-Participant Breakdown ──")
    print(df.to_string(index=False))
    return df


# ─────────────────────────────────────────
# MULTICLASS LABEL HELPERS
# ─────────────────────────────────────────

def get_food_label_names(unique_codes):
    """Return list of food name strings for given food codes."""
    return [FOOD_NAMES.get(c, f"code_{c}") for c in sorted(unique_codes)]
