"""
SensEat — Per-Participant LOPO Evaluation (Participants 22-41)
==============================================================
Full Leave-One-Participant-Out: each of the 20 new participants
is held out once as test; all others train.

T1 Binary    : RF on combined flat features   (best stable binary model)
T2 Multiclass: SVM on combined flat features  (best multiclass model)

Outputs:
    pipeline_per_participant_binary.csv
    pipeline_per_participant_multiclass.csv
    figures/per_participant/binary_f1_bar.png
    figures/per_participant/multiclass_f1_bar.png
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
SEED = 42
RNG  = np.random.default_rng(SEED)

SCRIPT_DIR = Path(__file__).resolve().parent
FIG_DIR    = SCRIPT_DIR / "figures" / "per_participant"
FIG_DIR.mkdir(parents=True, exist_ok=True)

BINARY_CACHE     = SCRIPT_DIR / "feature_cache_binary_v4.npz"
MULTICLASS_CACHE = SCRIPT_DIR / "feature_cache_multiclass_balanced_v2.npz"

FOOD_NAMES = {
    1: "Tortilla", 2: "Mandarin",  3: "Chicken",
    4: "Cheeze_It", 5: "Carrots",  8: "Noodles",
    9: "Water",    10: "Coke",
}

NEW_P_MIN, NEW_P_MAX = 22, 41   # participants to evaluate


# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
def undersample_binary(X, y):
    eating = np.where(y == 1)[0]
    idle   = np.where(y == 0)[0]
    if len(eating) <= len(idle):
        return X, y
    chosen = RNG.choice(eating, size=len(idle), replace=False)
    idx    = np.sort(np.concatenate([chosen, idle]))
    return X[idx], y[idx]


def undersample_multiclass(X, y):
    classes, counts = np.unique(y, return_counts=True)
    n_min = counts.min()
    selected = []
    for cls in classes:
        idx = np.where(y == cls)[0]
        selected.append(RNG.choice(idx, size=n_min, replace=False))
    idx = np.concatenate(selected)
    return X[idx], y[idx]


def bar_chart(values, labels, title, ylabel, out_path, mean_val=None):
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#4C72B0"] * len(labels)
    bars = ax.bar(labels, values, color=colors, edgecolor="white", width=0.6)
    if mean_val is not None:
        ax.axhline(mean_val, color="red", linewidth=1.5,
                   linestyle="--", label=f"Mean = {mean_val:.3f}")
        ax.legend(fontsize=10)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("Participant ID", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Chart saved: {out_path.name}")


# ─────────────────────────────────────────
# T1 — BINARY (RF)
# ─────────────────────────────────────────
def run_t1_per_participant():
    print("\n" + "=" * 68)
    print("  T1 — Binary LOPO Per-Participant (RF on flat features)")
    print(f"  Participants {NEW_P_MIN}-{NEW_P_MAX} | Eating vs Idle")
    print("=" * 68)

    d      = np.load(BINARY_CACHE, allow_pickle=False)
    X_all  = d["X_flat"].astype(np.float32)
    y_all  = d["y"].astype(np.int32)
    g_all  = d["groups"]

    # Idle pools (shared across all folds)
    idle_tr_idx = np.where(g_all == -1)[0]
    idle_te_idx = np.where(g_all == -2)[0]

    # Eating segments for new participants only
    new_eat_mask = (g_all >= NEW_P_MIN) & (g_all <= NEW_P_MAX)
    new_eat_idx  = np.where(new_eat_mask)[0]
    participants = sorted(np.unique(g_all[new_eat_idx]).tolist())

    print(f"  Participants: {participants}")
    print(f"  Eating segments: {len(new_eat_idx)} | Idle pool: {len(idle_tr_idx)+len(idle_te_idx)}\n")

    rows = []
    all_true, all_pred = [], []

    for p in participants:
        te_eat = new_eat_idx[g_all[new_eat_idx] == p]
        tr_eat = new_eat_idx[g_all[new_eat_idx] != p]

        tr_idx = np.concatenate([tr_eat, idle_tr_idx])
        te_idx = np.concatenate([te_eat, idle_te_idx])

        X_tr, y_tr = X_all[tr_idx], y_all[tr_idx]
        X_te, y_te = X_all[te_idx], y_all[te_idx]

        X_tr, y_tr = undersample_binary(X_tr, y_tr)

        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(X_tr)
        X_te   = scaler.transform(X_te)

        clf = RandomForestClassifier(
            n_estimators=200, random_state=SEED, n_jobs=-1, class_weight="balanced")
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)

        acc  = accuracy_score(y_te, y_pred)
        prec = float(np.sum((y_pred == 1) & (y_te == 1))) / (np.sum(y_pred == 1) + 1e-8)
        rec  = float(np.sum((y_pred == 1) & (y_te == 1))) / (np.sum(y_te == 1) + 1e-8)
        f1   = 2 * prec * rec / (prec + rec + 1e-8)

        n_e = int(np.sum(y_te == 1))
        n_i = int(np.sum(y_te == 0))
        print(f"  P{p:03d}: test={len(y_te)} (E:{n_e} I:{n_i})  "
              f"Acc={acc:.3f}  P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}")

        rows.append({
            "participant": p, "n_eating_test": n_e, "n_idle_test": n_i,
            "accuracy": round(acc, 4), "precision": round(prec, 4),
            "recall": round(rec, 4), "f1": round(f1, 4),
        })
        all_true.extend(y_te.tolist())
        all_pred.extend(y_pred.tolist())

    df = pd.DataFrame(rows)
    mean_f1 = df["f1"].mean()
    mean_acc = df["accuracy"].mean()
    print(f"\n  MEAN across participants: Acc={mean_acc:.4f}  F1={mean_f1:.4f}")

    out_csv = SCRIPT_DIR / "pipeline_per_participant_binary.csv"
    df.to_csv(out_csv, index=False)
    print(f"  Saved: {out_csv.name}")

    bar_chart(
        df["f1"].tolist(),
        [f"P{p}" for p in df["participant"]],
        title=f"Binary (Eating vs Idle) — Per-Participant F1\nModel: RF | LOPO | Mean F1 = {mean_f1:.3f}",
        ylabel="F1 Score",
        out_path=FIG_DIR / "binary_f1_bar.png",
        mean_val=mean_f1,
    )
    bar_chart(
        df["accuracy"].tolist(),
        [f"P{p}" for p in df["participant"]],
        title=f"Binary (Eating vs Idle) — Per-Participant Accuracy\nModel: RF | LOPO | Mean Acc = {mean_acc:.3f}",
        ylabel="Accuracy",
        out_path=FIG_DIR / "binary_acc_bar.png",
        mean_val=mean_acc,
    )
    return df


# ─────────────────────────────────────────
# T2 — MULTICLASS (SVM)
# ─────────────────────────────────────────
def run_t2_per_participant():
    print("\n" + "=" * 68)
    print("  T2 — Multiclass LOPO Per-Participant (SVM on flat features)")
    print(f"  Participants {NEW_P_MIN}-{NEW_P_MAX} | Food recognition")
    print("=" * 68)

    d      = np.load(MULTICLASS_CACHE, allow_pickle=False)
    X_all  = d["X_flat"].astype(np.float32)
    y_all  = d["y"].astype(np.int32)
    g_all  = d["groups"]

    new_mask = (g_all >= NEW_P_MIN) & (g_all <= NEW_P_MAX)
    X_all  = X_all[new_mask]
    y_all  = y_all[new_mask]
    g_all  = g_all[new_mask]

    participants  = sorted(np.unique(g_all).tolist())
    food_codes    = sorted(np.unique(y_all).tolist())
    food_labels   = [FOOD_NAMES.get(c, str(c)) for c in food_codes]

    print(f"  Participants: {participants}")
    print(f"  Food classes: {food_labels}\n")

    rows = []
    all_true, all_pred = [], []

    for p in participants:
        tr_mask = g_all != p
        te_mask = g_all == p

        X_tr, y_tr = X_all[tr_mask], y_all[tr_mask]
        X_te, y_te = X_all[te_mask], y_all[te_mask]

        X_tr, y_tr = undersample_multiclass(X_tr, y_tr)

        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(X_tr)
        X_te   = scaler.transform(X_te)

        clf = SVC(kernel="rbf", C=10, gamma="scale", random_state=SEED)
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)

        acc      = accuracy_score(y_te, y_pred)
        macro_f1 = f1_score(y_te, y_pred, average="macro", zero_division=0)
        wtd_f1   = f1_score(y_te, y_pred, average="weighted", zero_division=0)

        per_class = f1_score(y_te, y_pred, labels=food_codes,
                             average=None, zero_division=0)
        per_class_str = {FOOD_NAMES.get(c, str(c)): round(float(v), 3)
                         for c, v in zip(food_codes, per_class)}

        print(f"  P{p:03d}: n={len(y_te):4d}  Acc={acc:.3f}  "
              f"MacroF1={macro_f1:.3f}  WtdF1={wtd_f1:.3f}")
        print(f"         Per-class: {per_class_str}")

        row = {
            "participant": p, "n_test": len(y_te),
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(wtd_f1, 4),
        }
        for c, v in zip(food_codes, per_class):
            row[f"f1_{FOOD_NAMES.get(c, str(c))}"] = round(float(v), 4)
        rows.append(row)
        all_true.extend(y_te.tolist())
        all_pred.extend(y_pred.tolist())

    df = pd.DataFrame(rows)
    mean_macro = df["macro_f1"].mean()
    mean_acc   = df["accuracy"].mean()
    print(f"\n  MEAN across participants: Acc={mean_acc:.4f}  MacroF1={mean_macro:.4f}")

    out_csv = SCRIPT_DIR / "pipeline_per_participant_multiclass.csv"
    df.to_csv(out_csv, index=False)
    print(f"  Saved: {out_csv.name}")

    bar_chart(
        df["macro_f1"].tolist(),
        [f"P{p}" for p in df["participant"]],
        title=f"Multiclass Food Recognition — Per-Participant Macro F1\nModel: SVM | LOPO | Mean MacroF1 = {mean_macro:.3f}",
        ylabel="Macro F1",
        out_path=FIG_DIR / "multiclass_f1_bar.png",
        mean_val=mean_macro,
    )
    bar_chart(
        df["accuracy"].tolist(),
        [f"P{p}" for p in df["participant"]],
        title=f"Multiclass Food Recognition — Per-Participant Accuracy\nModel: SVM | LOPO | Mean Acc = {mean_acc:.3f}",
        ylabel="Accuracy",
        out_path=FIG_DIR / "multiclass_acc_bar.png",
        mean_val=mean_acc,
    )
    return df


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def main():
    print("=" * 68)
    print("  SensEat — Per-Participant LOPO Evaluation")
    print(f"  Participants {NEW_P_MIN}-{NEW_P_MAX} (20 new participants)")
    print("=" * 68)

    df_bin = run_t1_per_participant()
    df_mc  = run_t2_per_participant()

    print("\n" + "=" * 68)
    print("  SUMMARY")
    print("=" * 68)
    print(f"\n  T1 Binary (RF):")
    print(f"    Mean F1       = {df_bin['f1'].mean():.4f}")
    print(f"    Mean Accuracy = {df_bin['accuracy'].mean():.4f}")
    print(f"    Min F1        = {df_bin['f1'].min():.4f} (P{df_bin.loc[df_bin['f1'].idxmin(),'participant']})")
    print(f"    Max F1        = {df_bin['f1'].max():.4f} (P{df_bin.loc[df_bin['f1'].idxmax(),'participant']})")

    print(f"\n  T2 Multiclass (SVM):")
    print(f"    Mean MacroF1  = {df_mc['macro_f1'].mean():.4f}")
    print(f"    Mean Accuracy = {df_mc['accuracy'].mean():.4f}")
    print(f"    Min MacroF1   = {df_mc['macro_f1'].min():.4f} (P{df_mc.loc[df_mc['macro_f1'].idxmin(),'participant']})")
    print(f"    Max MacroF1   = {df_mc['macro_f1'].max():.4f} (P{df_mc.loc[df_mc['macro_f1'].idxmax(),'participant']})")

    print("\nPer-participant evaluation complete.")


if __name__ == "__main__":
    main()
