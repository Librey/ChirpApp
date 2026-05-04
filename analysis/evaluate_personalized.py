"""
evaluate_personalized.py
========================
Evaluates the professor's pre-trained personalized models on the first 20
participants' data (users 001-020).

T2: 10-class food recognition  → accuracy, macro F1, per-class F1, confusion matrix
T3: stress regression           → MAE, RMSE (requires stress_labels.npz — see below)

Models used:
  personalized_models/T2_user_XXX_multiclass.keras
  personalized_models/T2_user_XXX_normalization.npz
  personalized_models/T3_user_XXX_stress.keras
  personalized_models/T3_user_XXX_normalization.npz

Data source: feature_cache_multiclass_balanced_v2.npz (X_stft, groups 1-20)

T3 stress labels: provide stress_labels.npz with keys:
    'user_ids'    : (N,) int   — participant IDs matching cache groups
    'stress_labels': (N,) float — stress levels 1-5 per segment
"""

import os
import warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

import tensorflow as tf
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                              classification_report)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
MODEL_DIR    = SCRIPT_DIR / "personalized_models"
CACHE_PATH   = SCRIPT_DIR / "feature_cache_multiclass_balanced_v2.npz"
STRESS_LABELS_PATH = SCRIPT_DIR / "stress_labels.npz"   # provide this for T3
FIG_DIR      = SCRIPT_DIR / "figures" / "personalized_eval"
FIG_DIR.mkdir(parents=True, exist_ok=True)

FOOD_NAMES = {
    1: "Tortilla",  2: "Fruit",     3: "Chicken",  4: "Cracker",
    5: "Carrot",    6: "Chocolate", 7: "Yogurt",    8: "Noodles",
    9: "Water",    10: "Soft Drink"
}

N_USERS = 20


# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────

def load_user_data_t2(cache, user_id):
    """Return (X_stft, y_food) for a given user from the multiclass cache."""
    groups = cache["groups"]
    mask   = groups == user_id
    X = cache["X_stft"][mask].astype(np.float32)
    y = cache["y"][mask].astype(np.int32)
    return X, y


def normalize(X, norm_path):
    """Apply per-user z-score normalization loaded from .npz file."""
    d    = np.load(norm_path)
    mean = d["mean"].astype(np.float32)
    std  = d["std"].astype(np.float32)
    std  = np.where(std < 1e-8, 1.0, std)
    return (X - mean) / std


def save_cm(y_true, y_pred, label_names, title, filename):
    cm     = confusion_matrix(y_true, y_pred, labels=list(range(1, 11)))
    cm_pct = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_pct, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=label_names, yticklabels=label_names, ax=ax)
    ax.set_title(title, fontsize=13)
    ax.set_ylabel("True Food")
    ax.set_xlabel("Predicted Food")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    out = FIG_DIR / filename
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  CM saved: {out.name}")


# ─────────────────────────────────────────
# T2 EVALUATION
# ─────────────────────────────────────────

def evaluate_t2(cache):
    print("\n" + "=" * 65)
    print("  T2 — Personalized Food Recognition (users 001-020)")
    print("=" * 65)

    label_names = [FOOD_NAMES[i] for i in range(1, 11)]
    rows = []
    all_true, all_pred = [], []

    for uid in range(1, N_USERS + 1):
        user_str  = f"{uid:03d}"
        model_path = MODEL_DIR / f"T2_user_{user_str}_multiclass.keras"
        norm_path  = MODEL_DIR / f"T2_user_{user_str}_normalization.npz"

        if not model_path.exists():
            print(f"  [user {user_str}] model not found, skipping")
            continue

        X, y = load_user_data_t2(cache, uid)
        if len(y) == 0:
            print(f"  [user {user_str}] no data in cache, skipping")
            continue

        X_norm = normalize(X, norm_path)
        model  = tf.keras.models.load_model(str(model_path))

        probs  = model.predict(X_norm, verbose=0)
        # Model outputs 10 classes indexed 0-9; map back to food codes 1-10
        y_pred_idx = np.argmax(probs, axis=1)
        y_pred     = y_pred_idx + 1          # shift to food codes 1-10

        acc      = accuracy_score(y, y_pred)
        macro_f1 = f1_score(y, y_pred, average="macro",    zero_division=0)
        wtd_f1   = f1_score(y, y_pred, average="weighted", zero_division=0)
        per_cls  = f1_score(y, y_pred, average=None,
                            labels=list(range(1, 11)), zero_division=0)

        row = {"user": user_str, "n_segments": len(y),
               "accuracy": round(acc, 4),
               "macro_f1": round(macro_f1, 4),
               "weighted_f1": round(wtd_f1, 4)}
        for i, name in enumerate(label_names):
            row[f"f1_{name}"] = round(float(per_cls[i]), 4)
        rows.append(row)

        all_true.extend(y.tolist())
        all_pred.extend(y_pred.tolist())

        print(f"  [user {user_str}]: Acc={acc:.4f}  MacroF1={macro_f1:.4f}"
              f"  WtdF1={wtd_f1:.4f}  (n={len(y)})")

        tf.keras.backend.clear_session()

    # Aggregate
    df = pd.DataFrame(rows)
    mean_acc = df["accuracy"].mean()
    mean_mf1 = df["macro_f1"].mean()
    mean_wf1 = df["weighted_f1"].mean()

    print(f"\n  T2 FINAL (mean over {len(df)} users):")
    print(f"    Accuracy    = {mean_acc:.4f} ± {df['accuracy'].std():.4f}")
    print(f"    Macro F1    = {mean_mf1:.4f} ± {df['macro_f1'].std():.4f}")
    print(f"    Weighted F1 = {mean_wf1:.4f}")

    # Save CSV
    out_csv = SCRIPT_DIR / "personalized_t2_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n  Results saved: {out_csv.name}")

    # Confusion matrix — all users aggregated
    save_cm(all_true, all_pred, label_names,
            "T2 Personalized — Aggregated Confusion Matrix (users 001-020)",
            "cm_t2_personalized.png")

    # Per-user accuracy bar chart
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#16A34A" if a >= 0.7 else ("#EA580C" if a >= 0.5 else "#2563EB")
              for a in df["accuracy"]]
    ax.bar(df["user"], df["accuracy"] * 100, color=colors)
    ax.axhline(y=mean_acc * 100, color="red", linestyle="--",
               linewidth=1.5, label=f"Mean = {mean_acc*100:.1f}%")
    ax.set_xlabel("User ID"); ax.set_ylabel("Accuracy (%)")
    ax.set_title("T2 Personalized — Per-User Accuracy")
    ax.set_xticklabels(df["user"], rotation=45, ha="right", fontsize=8)
    ax.legend(); ax.set_ylim(0, 105)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig_path = FIG_DIR / "t2_per_user_accuracy.png"
    plt.savefig(fig_path, dpi=150); plt.close()
    print(f"  Bar chart saved: {fig_path.name}")

    return df, {"accuracy": mean_acc, "macro_f1": mean_mf1, "weighted_f1": mean_wf1}


# ─────────────────────────────────────────
# T3 EVALUATION
# ─────────────────────────────────────────

def evaluate_t3(cache):
    print("\n" + "=" * 65)
    print("  T3 — Personalized Stress Prediction (users 001-020)")
    print("=" * 65)

    # Check for stress labels
    if not STRESS_LABELS_PATH.exists():
        print(f"\n  [SKIP] stress_labels.npz not found at:")
        print(f"         {STRESS_LABELS_PATH}")
        print(f"  Please provide this file with keys:")
        print(f"    'user_ids'     : (N,) int   — participant IDs (1-20)")
        print(f"    'stress_labels': (N,) float — stress level per segment (1-5)")
        print(f"\n  Running T3 inference anyway — saving raw predictions per user.")
        _t3_predict_only(cache)
        return None, None

    # Load stress labels
    sl   = np.load(STRESS_LABELS_PATH)
    s_uid = sl["user_ids"].astype(int)
    s_y   = sl["stress_labels"].astype(np.float32)

    rows = []
    all_true, all_pred = [], []

    for uid in range(1, N_USERS + 1):
        user_str   = f"{uid:03d}"
        model_path = MODEL_DIR / f"T3_user_{user_str}_stress.keras"
        norm_path  = MODEL_DIR / f"T3_user_{user_str}_normalization.npz"

        if not model_path.exists():
            continue

        # Get STFT features for this user
        groups = cache["groups"]
        mask_cache = groups == uid
        X = cache["X_stft"][mask_cache].astype(np.float32)

        # Get stress labels for this user
        mask_stress = s_uid == uid
        y_stress = s_y[mask_stress]

        if len(X) == 0 or len(y_stress) == 0:
            print(f"  [user {user_str}] no data, skipping")
            continue

        # Align lengths
        n = min(len(X), len(y_stress))
        X, y_stress = X[:n], y_stress[:n]

        X_norm = normalize(X, norm_path)
        model  = tf.keras.models.load_model(str(model_path))

        y_pred = model.predict(X_norm, verbose=0).ravel()
        y_pred = np.clip(y_pred, 1.0, 5.0)

        mae  = float(np.mean(np.abs(y_pred - y_stress)))
        rmse = float(np.sqrt(np.mean((y_pred - y_stress) ** 2)))

        rows.append({"user": user_str, "n_segments": n,
                     "mae": round(mae, 4), "rmse": round(rmse, 4)})
        all_true.extend(y_stress.tolist())
        all_pred.extend(y_pred.tolist())

        print(f"  [user {user_str}]: MAE={mae:.4f}  RMSE={rmse:.4f}  (n={n})")
        tf.keras.backend.clear_session()

    df = pd.DataFrame(rows)
    mean_mae  = df["mae"].mean()
    mean_rmse = df["rmse"].mean()

    print(f"\n  T3 FINAL (mean over {len(df)} users):")
    print(f"    MAE  = {mean_mae:.4f} ± {df['mae'].std():.4f}")
    print(f"    RMSE = {mean_rmse:.4f} ± {df['rmse'].std():.4f}")

    out_csv = SCRIPT_DIR / "personalized_t3_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n  Results saved: {out_csv.name}")

    return df, {"mae": mean_mae, "rmse": mean_rmse}


def _t3_predict_only(cache):
    """Run T3 inference without ground truth — saves raw predictions."""
    rows = []
    for uid in range(1, N_USERS + 1):
        user_str   = f"{uid:03d}"
        model_path = MODEL_DIR / f"T3_user_{user_str}_stress.keras"
        norm_path  = MODEL_DIR / f"T3_user_{user_str}_normalization.npz"
        if not model_path.exists():
            continue
        groups = cache["groups"]
        X = cache["X_stft"][groups == uid].astype(np.float32)
        if len(X) == 0:
            continue
        X_norm = normalize(X, norm_path)
        model  = tf.keras.models.load_model(str(model_path))
        y_pred = model.predict(X_norm, verbose=0).ravel()
        y_pred = np.clip(y_pred, 1.0, 5.0)
        rows.append({"user": user_str, "n_segments": len(X),
                     "pred_mean": round(float(y_pred.mean()), 4),
                     "pred_std":  round(float(y_pred.std()),  4),
                     "pred_min":  round(float(y_pred.min()),  4),
                     "pred_max":  round(float(y_pred.max()),  4)})
        print(f"  [user {user_str}]: pred_mean={y_pred.mean():.3f}"
              f"  std={y_pred.std():.3f}  (n={len(X)})")
        tf.keras.backend.clear_session()

    df = pd.DataFrame(rows)
    out = SCRIPT_DIR / "personalized_t3_predictions.csv"
    df.to_csv(out, index=False)
    print(f"\n  Raw predictions saved: {out.name}")
    print(f"  (Provide stress_labels.npz to compute MAE/RMSE)")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

def main():
    print("=" * 65)
    print("  SensEat — Personalized Model Evaluation (users 001-020)")
    print("=" * 65)

    print("\nLoading feature cache ...")
    cache = np.load(CACHE_PATH, allow_pickle=False)
    groups = cache["groups"]
    print(f"  Total segments: {len(groups)}")
    print(f"  Users 1-20 segments: {((groups >= 1) & (groups <= 20)).sum()}")

    t2_df, t2_summary = evaluate_t2(cache)
    t3_df, t3_summary = evaluate_t3(cache)

    # Final summary
    print("\n" + "=" * 65)
    print("  FINAL SUMMARY")
    print("=" * 65)
    if t2_summary:
        print(f"  T2 Accuracy    : {t2_summary['accuracy']:.4f}")
        print(f"  T2 Macro F1    : {t2_summary['macro_f1']:.4f}")
        print(f"  T2 Weighted F1 : {t2_summary['weighted_f1']:.4f}")
    if t3_summary:
        print(f"  T3 MAE         : {t3_summary['mae']:.4f}")
        print(f"  T3 RMSE        : {t3_summary['rmse']:.4f}")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
