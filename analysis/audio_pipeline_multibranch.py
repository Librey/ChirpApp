"""
SensEat — Multi-Branch Attention Fusion Model
=============================================
Implements the architecture from Section 5.1/5.2 of the paper:
  - Branch 1: 2D CNN on STFT spectrogram
  - Branch 2: 2D CNN on MFCC
  - Branch 3: 2D CNN on GFCC
  - Branch 4: 1D CNN on flat features (stat + spectral + wavelet + tap profile)
  - Attention-based fusion of all branch latents
  - Final dense classifier

T1 — Binary (eating vs idle):
    41 participants, GroupKFold(5), per-fold undersampling
    Cache: feature_cache_binary_v4.npz

T2 — Multiclass (7 food classes):
    20 new participants (022-041), GroupKFold(5), per-fold undersampling
    Cache: feature_cache_multiclass_balanced_v2.npz

Output:
    pipeline_multibranch_results.csv
    figures/multibranch/cm_*.png
"""

import os
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.metrics import (accuracy_score, f1_score,
                              confusion_matrix, precision_score, recall_score)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
SEED      = 42
N_SPLITS  = 5
np.random.seed(SEED)
tf.random.set_seed(SEED)
RNG = np.random.default_rng(SEED)

SCRIPT_DIR = Path(__file__).resolve().parent
FIG_DIR    = SCRIPT_DIR / "figures" / "multibranch"
FIG_DIR.mkdir(parents=True, exist_ok=True)

BINARY_CACHE     = SCRIPT_DIR / "feature_cache_binary_v4.npz"
MULTICLASS_CACHE = SCRIPT_DIR / "feature_cache_multiclass_balanced_v2.npz"

FOOD_NAMES = {
    1: "Tortilla", 2: "Mandarin",  4: "Cheeze_It",
    5: "Carrots",  8: "Noodles",   9: "Water", 10: "Coke",
}


# ─────────────────────────────────────────
# UNDERSAMPLING
# ─────────────────────────────────────────
def undersample_binary(X_dict, y):
    eating = np.where(y == 1)[0]
    idle   = np.where(y == 0)[0]
    if len(eating) <= len(idle):
        return X_dict, y
    chosen = RNG.choice(eating, size=len(idle), replace=False)
    idx    = np.concatenate([chosen, idle])
    RNG.shuffle(idx)
    return {k: v[idx] for k, v in X_dict.items()}, y[idx]


def undersample_multiclass(X_dict, y):
    classes, counts = np.unique(y, return_counts=True)
    n_min = counts.min()
    selected = []
    for cls in classes:
        idx = np.where(y == cls)[0]
        selected.append(RNG.choice(idx, size=n_min, replace=False))
    idx = np.concatenate(selected)
    RNG.shuffle(idx)
    return {k: v[idx] for k, v in X_dict.items()}, y[idx]


# ─────────────────────────────────────────
# MODEL — Multi-Branch Attention Fusion
# ─────────────────────────────────────────
def cnn2d_branch(x, name):
    """2D CNN branch for image features (STFT / MFCC / GFCC)."""
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu", name=name)(x)
    return x


def cnn1d_branch(x, n_features, name):
    """1D CNN branch for flat feature vector."""
    x = layers.Reshape((n_features, 1))(x)
    x = layers.Conv1D(32, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(64, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(128, 3, activation="relu", padding="same")(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation="relu", name=name)(x)
    return x


def build_multibranch(stft_shape, mfcc_shape, gfcc_shape, n_flat, n_classes, task="binary"):
    """
    Multi-branch attention fusion model.
    Attention: concatenate all latents → Dense(4, softmax) → scalar weight per branch
               → weighted sum of latents → final dense classifier.
    """
    # Inputs
    stft_in = Input(shape=stft_shape, name="stft_in")
    mfcc_in = Input(shape=mfcc_shape, name="mfcc_in")
    gfcc_in = Input(shape=gfcc_shape, name="gfcc_in")
    flat_in = Input(shape=(n_flat,),  name="flat_in")

    # Branch latents (each 128-dim)
    lat_stft = cnn2d_branch(stft_in, "lat_stft")
    lat_mfcc = cnn2d_branch(mfcc_in, "lat_mfcc")
    lat_gfcc = cnn2d_branch(gfcc_in, "lat_gfcc")
    lat_flat = cnn1d_branch(flat_in, n_flat, "lat_flat")

    # Attention fusion
    # Stack all latents → compute attention weights → weighted sum
    concat   = layers.Concatenate(name="concat_latents")([lat_stft, lat_mfcc, lat_gfcc, lat_flat])
    attn_w   = layers.Dense(4, activation="softmax", name="attn_weights")(concat)  # (batch, 4)

    # Apply attention: each branch latent scaled by its weight
    w_stft = layers.Lambda(lambda x: x[0] * tf.expand_dims(x[1][:, 0], 1))([lat_stft, attn_w])
    w_mfcc = layers.Lambda(lambda x: x[0] * tf.expand_dims(x[1][:, 1], 1))([lat_mfcc, attn_w])
    w_gfcc = layers.Lambda(lambda x: x[0] * tf.expand_dims(x[1][:, 2], 1))([lat_gfcc, attn_w])
    w_flat = layers.Lambda(lambda x: x[0] * tf.expand_dims(x[1][:, 3], 1))([lat_flat, attn_w])

    fused = layers.Add(name="fused")([w_stft, w_mfcc, w_gfcc, w_flat])

    # Final classifier
    x = layers.Dense(64, activation="relu")(fused)
    x = layers.Dropout(0.4)(x)

    if task == "binary":
        out = layers.Dense(1, activation="sigmoid", name="output")(x)
        model = models.Model(
            inputs=[stft_in, mfcc_in, gfcc_in, flat_in], outputs=out)
        model.compile(optimizer="adam", loss="binary_crossentropy",
                      metrics=["accuracy"])
    else:
        out = layers.Dense(n_classes, activation="softmax", name="output")(x)
        model = models.Model(
            inputs=[stft_in, mfcc_in, gfcc_in, flat_in], outputs=out)
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
    return model


# ─────────────────────────────────────────
# CONFUSION MATRIX
# ─────────────────────────────────────────
def save_cm_binary(y_true, y_pred, tag):
    cm  = confusion_matrix(y_true, y_pred, labels=[0, 1])
    pct = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8) * 100
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(pct, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=["Idle", "Eating"],
                yticklabels=["Idle", "Eating"], ax=ax)
    ax.set_title(f"Multi-Branch Binary CM (%)\n{tag}")
    ax.set_ylabel("True"); ax.set_xlabel("Predicted")
    plt.tight_layout()
    out = FIG_DIR / f"cm_binary_{tag}.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"    CM saved: {out.name}")


def save_cm_multiclass(y_true, y_pred, labels, label_names, tag):
    cm  = confusion_matrix(y_true, y_pred, labels=labels)
    pct = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8) * 100
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pct, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=label_names, yticklabels=label_names, ax=ax)
    ax.set_title(f"Multi-Branch Multiclass CM (%)\n{tag}")
    ax.set_ylabel("True Food"); ax.set_xlabel("Predicted Food")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    out = FIG_DIR / f"cm_multiclass_{tag}.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"    CM saved: {out.name}")


# ─────────────────────────────────────────
# T1 — BINARY CLASSIFICATION
# ─────────────────────────────────────────
def run_t1():
    print("\n" + "=" * 72)
    print("  T1 — Binary Classification (Multi-Branch)")
    print("  41 participants | GroupKFold(5) | Per-fold undersampling")
    print("=" * 72)

    print("\nLoading binary cache ...")
    d = np.load(BINARY_CACHE, allow_pickle=False)
    X = {
        "stft": d["X_stft"].astype(np.float32),
        "mfcc": d["X_mfcc"].astype(np.float32),
        "gfcc": d["X_gfcc"].astype(np.float32),
        "flat": d["X_flat"].astype(np.float32),
    }
    y      = d["y"].astype(np.int32)
    groups = d["groups"]

    n_eating = int(np.sum(y == 1))
    n_idle   = int(np.sum(y == 0))
    print(f"  Segments: {len(y)} | Eating: {n_eating} | Idle: {n_idle}")

    stft_shape = X["stft"].shape[1:]
    mfcc_shape = X["mfcc"].shape[1:]
    gfcc_shape = X["gfcc"].shape[1:]
    n_flat     = X["flat"].shape[1]

    # Mask out group IDs -1 and -2 (idle pool) for GroupKFold splitting
    # but keep them in train/test by appending after fold split
    IDLE_TR = -1; IDLE_TE = -2
    idle_tr_idx = np.where(groups == IDLE_TR)[0]
    idle_te_idx = np.where(groups == IDLE_TE)[0]
    real_idx    = np.where((groups != IDLE_TR) & (groups != IDLE_TE))[0]

    gkf = GroupKFold(n_splits=N_SPLITS)
    fold_metrics = defaultdict(list)
    all_true, all_pred = [], []

    for fold, (tr, te) in enumerate(gkf.split(real_idx, groups=groups[real_idx])):
        tr_idx = np.concatenate([real_idx[tr], idle_tr_idx])
        te_idx = np.concatenate([real_idx[te], idle_te_idx])

        X_tr = {k: v[tr_idx] for k, v in X.items()}
        X_te = {k: v[te_idx] for k, v in X.items()}
        y_tr, y_te = y[tr_idx], y[te_idx]

        # Per-fold undersampling on training only
        X_tr, y_tr = undersample_binary(X_tr, y_tr)

        # Normalise each modality using training stats
        for key in ["stft", "mfcc", "gfcc", "flat"]:
            mn = X_tr[key].mean(); sd = X_tr[key].std() or 1.0
            X_tr[key] = (X_tr[key] - mn) / sd
            X_te[key] = (X_te[key] - mn) / sd

        print(f"\n  Fold {fold}: train={len(y_tr)} "
              f"(E:{np.sum(y_tr==1)} I:{np.sum(y_tr==0)}) | "
              f"test={len(y_te)} (E:{np.sum(y_te==1)} I:{np.sum(y_te==0)})")

        model = build_multibranch(stft_shape, mfcc_shape, gfcc_shape,
                                   n_flat, n_classes=1, task="binary")
        es = EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
        lr = ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.5)

        model.fit(
            [X_tr["stft"], X_tr["mfcc"], X_tr["gfcc"], X_tr["flat"]], y_tr,
            epochs=20, batch_size=64, validation_split=0.1,
            callbacks=[es, lr], verbose=0
        )

        y_prob = model.predict(
            [X_te["stft"], X_te["mfcc"], X_te["gfcc"], X_te["flat"]], verbose=0
        ).ravel()
        y_pred = (y_prob > 0.5).astype(int)
        tf.keras.backend.clear_session()

        acc  = accuracy_score(y_te, y_pred)
        prec = precision_score(y_te, y_pred, zero_division=0)
        rec  = recall_score(y_te, y_pred, zero_division=0)
        f1   = f1_score(y_te, y_pred, zero_division=0)

        fold_metrics["accuracy"].append(acc)
        fold_metrics["precision"].append(prec)
        fold_metrics["recall"].append(rec)
        fold_metrics["f1"].append(f1)

        print(f"    Acc={acc:.4f}  P={prec:.4f}  R={rec:.4f}  F1={f1:.4f}")
        all_true.extend(y_te); all_pred.extend(y_pred)

    save_cm_binary(np.array(all_true), np.array(all_pred), "multibranch")

    print(f"\n  T1 FINAL:")
    print(f"    Acc  = {np.mean(fold_metrics['accuracy']):.4f} +- {np.std(fold_metrics['accuracy']):.4f}")
    print(f"    Prec = {np.mean(fold_metrics['precision']):.4f}")
    print(f"    Rec  = {np.mean(fold_metrics['recall']):.4f}")
    print(f"    F1   = {np.mean(fold_metrics['f1']):.4f} +- {np.std(fold_metrics['f1']):.4f}")

    return {
        "task": "T1_binary", "model": "MultiBranch",
        "accuracy":  round(np.mean(fold_metrics["accuracy"]),  4),
        "precision": round(np.mean(fold_metrics["precision"]), 4),
        "recall":    round(np.mean(fold_metrics["recall"]),    4),
        "f1":        round(np.mean(fold_metrics["f1"]),        4),
        "f1_std":    round(np.std(fold_metrics["f1"]),         4),
    }


# ─────────────────────────────────────────
# T2 — MULTICLASS FOOD RECOGNITION
# ─────────────────────────────────────────
def run_t2():
    print("\n" + "=" * 72)
    print("  T2 — Multiclass Food Recognition (Multi-Branch)")
    print("  20 new participants (022-041) | 7 food classes | GroupKFold(5)")
    print("=" * 72)

    print("\nLoading multiclass cache ...")
    d = np.load(MULTICLASS_CACHE, allow_pickle=False)

    # Filter: only new participants 022-041 (consistent 7 food classes)
    groups_all = d["groups"]
    y_all      = d["y"]
    new_mask   = (groups_all >= 22) & (groups_all <= 41)

    X = {
        "stft": d["X_stft"][new_mask].astype(np.float32),
        "mfcc": d["X_mfcc"][new_mask].astype(np.float32),
        "gfcc": d["X_gfcc"][new_mask].astype(np.float32),
        "flat": d["X_flat"][new_mask].astype(np.float32),
    }
    y      = y_all[new_mask]
    groups = groups_all[new_mask]

    # Map food codes to 0-based indices
    unique_codes = sorted(np.unique(y))
    code_to_idx  = {c: i for i, c in enumerate(unique_codes)}
    idx_to_code  = {i: c for c, i in code_to_idx.items()}
    y_idx        = np.array([code_to_idx[c] for c in y])
    n_classes    = len(unique_codes)
    label_names  = [FOOD_NAMES.get(c, str(c)) for c in unique_codes]

    print(f"  Segments: {len(y)} | Participants: {len(np.unique(groups))}")
    for code in unique_codes:
        print(f"    {FOOD_NAMES.get(code, str(code)):12s}: {int(np.sum(y==code))} segments")

    stft_shape = X["stft"].shape[1:]
    mfcc_shape = X["mfcc"].shape[1:]
    gfcc_shape = X["gfcc"].shape[1:]
    n_flat     = X["flat"].shape[1]

    gkf = GroupKFold(n_splits=N_SPLITS)
    fold_metrics = defaultdict(list)
    all_true, all_pred = [], []

    for fold, (tr, te) in enumerate(gkf.split(X["stft"], groups=groups)):
        X_tr = {k: v[tr] for k, v in X.items()}
        X_te = {k: v[te] for k, v in X.items()}
        y_tr_idx, y_te_idx = y_idx[tr], y_idx[te]
        y_te_codes = y[te]

        # Per-fold undersampling on training only
        X_tr, y_tr_idx = undersample_multiclass(X_tr, y_tr_idx)

        # Normalise
        for key in ["stft", "mfcc", "gfcc", "flat"]:
            mn = X_tr[key].mean(); sd = X_tr[key].std() or 1.0
            X_tr[key] = (X_tr[key] - mn) / sd
            X_te[key] = (X_te[key] - mn) / sd

        print(f"\n  Fold {fold}: train={len(y_tr_idx)} | test={len(y_te_idx)}")

        model = build_multibranch(stft_shape, mfcc_shape, gfcc_shape,
                                   n_flat, n_classes=n_classes, task="multiclass")
        es = EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
        lr = ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.5)

        model.fit(
            [X_tr["stft"], X_tr["mfcc"], X_tr["gfcc"], X_tr["flat"]], y_tr_idx,
            epochs=20, batch_size=64, validation_split=0.1,
            callbacks=[es, lr], verbose=0
        )

        y_pred_idx  = np.argmax(model.predict(
            [X_te["stft"], X_te["mfcc"], X_te["gfcc"], X_te["flat"]], verbose=0
        ), axis=1)
        y_pred_codes = np.array([idx_to_code[i] for i in y_pred_idx])
        tf.keras.backend.clear_session()

        acc      = accuracy_score(y_te_codes, y_pred_codes)
        macro_f1 = f1_score(y_te_codes, y_pred_codes, average="macro", zero_division=0)
        wtd_f1   = f1_score(y_te_codes, y_pred_codes, average="weighted", zero_division=0)

        fold_metrics["accuracy"].append(acc)
        fold_metrics["macro_f1"].append(macro_f1)
        fold_metrics["weighted_f1"].append(wtd_f1)

        print(f"    Acc={acc:.4f}  MacroF1={macro_f1:.4f}  WtdF1={wtd_f1:.4f}")
        all_true.extend(y_te_codes); all_pred.extend(y_pred_codes)

    save_cm_multiclass(np.array(all_true), np.array(all_pred),
                       unique_codes, label_names, "multibranch")

    print(f"\n  T2 FINAL:")
    print(f"    Acc      = {np.mean(fold_metrics['accuracy']):.4f} +- {np.std(fold_metrics['accuracy']):.4f}")
    print(f"    MacroF1  = {np.mean(fold_metrics['macro_f1']):.4f} +- {np.std(fold_metrics['macro_f1']):.4f}")
    print(f"    WtdF1    = {np.mean(fold_metrics['weighted_f1']):.4f}")

    return {
        "task": "T2_multiclass", "model": "MultiBranch",
        "accuracy":     round(np.mean(fold_metrics["accuracy"]),     4),
        "macro_f1":     round(np.mean(fold_metrics["macro_f1"]),     4),
        "macro_f1_std": round(np.std(fold_metrics["macro_f1"]),      4),
        "weighted_f1":  round(np.mean(fold_metrics["weighted_f1"]),  4),
    }


# ─────────────────────────────────────────
# T2 — PERSONALIZED (LOPO + fine-tune)
# ─────────────────────────────────────────
def run_t2_personalized(finetune_ratio=0.2, finetune_epochs=10, finetune_lr=1e-4):
    """
    Personalized T2: LOPO across 20 participants + per-participant fine-tuning.

    For each participant p (20 folds):
      1. Train global multi-branch model on all other 19 participants.
      2. Take finetune_ratio of p's segments as a personal calibration set.
      3. Fine-tune the global model on that calibration set (low LR).
      4. Evaluate on remaining (1-finetune_ratio) of p's segments.

    This typically raises accuracy from ~20% (cross-user) to 70-90% because
    each person's chewing acoustics are highly individual.
    """
    print("\n" + "=" * 72)
    print("  T2 — Personalized Food Recognition (Multi-Branch + LOPO fine-tune)")
    print(f"  finetune_ratio={finetune_ratio:.0%}  finetune_epochs={finetune_epochs}  lr={finetune_lr}")
    print("=" * 72)

    d = np.load(MULTICLASS_CACHE, allow_pickle=False)
    groups_all = d["groups"]
    y_all      = d["y"]
    new_mask   = (groups_all >= 22) & (groups_all <= 41)

    X_all = {
        "stft": d["X_stft"][new_mask].astype(np.float32),
        "mfcc": d["X_mfcc"][new_mask].astype(np.float32),
        "gfcc": d["X_gfcc"][new_mask].astype(np.float32),
        "flat": d["X_flat"][new_mask].astype(np.float32),
    }
    y_all      = y_all[new_mask]
    groups_all = groups_all[new_mask]

    unique_codes = sorted(np.unique(y_all))
    code_to_idx  = {c: i for i, c in enumerate(unique_codes)}
    idx_to_code  = {i: c for c, i in code_to_idx.items()}
    y_idx_all    = np.array([code_to_idx[c] for c in y_all])
    n_classes    = len(unique_codes)
    label_names  = [FOOD_NAMES.get(c, str(c)) for c in unique_codes]

    stft_shape = X_all["stft"].shape[1:]
    mfcc_shape = X_all["mfcc"].shape[1:]
    gfcc_shape = X_all["gfcc"].shape[1:]
    n_flat     = X_all["flat"].shape[1]

    unique_parts = np.unique(groups_all)
    fold_metrics = defaultdict(list)
    all_true, all_pred = [], []

    for participant in unique_parts:
        test_mask  = groups_all == participant
        train_mask = ~test_mask

        X_tr = {k: v[train_mask] for k, v in X_all.items()}
        X_te_all = {k: v[test_mask] for k, v in X_all.items()}
        y_tr_idx  = y_idx_all[train_mask]
        y_te_codes_all = y_all[test_mask]
        y_te_idx_all   = y_idx_all[test_mask]

        # Split test participant: finetune vs eval
        n_total   = len(y_te_idx_all)
        n_ft      = max(1, int(n_total * finetune_ratio))
        ft_idx    = RNG.choice(n_total, size=n_ft, replace=False)
        eval_mask = np.ones(n_total, dtype=bool)
        eval_mask[ft_idx] = False

        X_ft   = {k: v[ft_idx]    for k, v in X_te_all.items()}
        X_eval = {k: v[eval_mask] for k, v in X_te_all.items()}
        y_ft_idx    = y_te_idx_all[ft_idx]
        y_eval_codes = y_te_codes_all[eval_mask]
        y_eval_idx   = y_te_idx_all[eval_mask]

        if len(y_eval_idx) == 0:
            continue

        # Per-fold undersampling on training data
        X_tr, y_tr_idx = undersample_multiclass(X_tr, y_tr_idx)

        # Normalize using training set statistics
        stats = {}
        for key in ["stft", "mfcc", "gfcc", "flat"]:
            mn = X_tr[key].mean(); sd = float(X_tr[key].std()) or 1.0
            stats[key] = (mn, sd)
            X_tr[key]   = (X_tr[key] - mn) / sd
            X_ft[key]   = (X_ft[key] - mn) / sd
            X_eval[key] = (X_eval[key] - mn) / sd

        # ── Step 1: Train global model ──
        model = build_multibranch(stft_shape, mfcc_shape, gfcc_shape,
                                   n_flat, n_classes=n_classes, task="multiclass")
        es = EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
        lr_cb = ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.5)

        model.fit(
            [X_tr["stft"], X_tr["mfcc"], X_tr["gfcc"], X_tr["flat"]], y_tr_idx,
            epochs=20, batch_size=64, validation_split=0.1,
            callbacks=[es, lr_cb], verbose=0
        )

        # ── Step 2: Fine-tune on participant's own calibration data ──
        model.optimizer.learning_rate = finetune_lr
        model.fit(
            [X_ft["stft"], X_ft["mfcc"], X_ft["gfcc"], X_ft["flat"]], y_ft_idx,
            epochs=finetune_epochs,
            batch_size=min(32, len(y_ft_idx)),
            callbacks=[EarlyStopping(patience=3, restore_best_weights=True,
                                     monitor="loss", verbose=0)],
            verbose=0
        )

        # ── Step 3: Evaluate ──
        y_pred_idx = np.argmax(model.predict(
            [X_eval["stft"], X_eval["mfcc"], X_eval["gfcc"], X_eval["flat"]], verbose=0
        ), axis=1)
        y_pred_codes = np.array([idx_to_code[i] for i in y_pred_idx])
        tf.keras.backend.clear_session()

        acc      = accuracy_score(y_eval_codes, y_pred_codes)
        macro_f1 = f1_score(y_eval_codes, y_pred_codes, average="macro", zero_division=0)
        wtd_f1   = f1_score(y_eval_codes, y_pred_codes, average="weighted", zero_division=0)

        fold_metrics["accuracy"].append(acc)
        fold_metrics["macro_f1"].append(macro_f1)
        fold_metrics["weighted_f1"].append(wtd_f1)

        print(f"  [{participant:03d}]: Acc={acc:.4f}  MacroF1={macro_f1:.4f}"
              f"  (ft={n_ft}, eval={eval_mask.sum()})")

        all_true.extend(y_eval_codes.tolist())
        all_pred.extend(y_pred_codes.tolist())

    save_cm_multiclass(np.array(all_true), np.array(all_pred),
                       unique_codes, label_names, "personalized")

    acc_mean  = np.mean(fold_metrics["accuracy"])
    mac_mean  = np.mean(fold_metrics["macro_f1"])
    wtd_mean  = np.mean(fold_metrics["weighted_f1"])
    print(f"\n  T2 Personalized FINAL:")
    print(f"    Acc      = {acc_mean:.4f} +- {np.std(fold_metrics['accuracy']):.4f}")
    print(f"    MacroF1  = {mac_mean:.4f} +- {np.std(fold_metrics['macro_f1']):.4f}")
    print(f"    WtdF1    = {wtd_mean:.4f}")

    return {
        "task": "T2_personalized", "model": "MultiBranch+Finetune",
        "accuracy":     round(acc_mean, 4),
        "macro_f1":     round(mac_mean, 4),
        "macro_f1_std": round(np.std(fold_metrics["macro_f1"]), 4),
        "weighted_f1":  round(wtd_mean, 4),
        "finetune_ratio": finetune_ratio,
    }


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def main():
    print("=" * 72)
    print("  SensEat — Multi-Branch Attention Fusion Pipeline")
    print("  Architecture: 2D CNN (STFT) + 2D CNN (MFCC) + 2D CNN (GFCC)")
    print("               + 1D CNN (flat) + Attention Fusion")
    print("=" * 72)

    results = []
    # T1 already completed: F1=0.9170 (Acc=0.8467, Prec=0.8467, Rec=1.0000)
    results.append({
        "task": "T1_binary", "model": "MultiBranch",
        "accuracy": 0.8467, "precision": 0.8467, "recall": 1.0000,
        "f1": 0.9170, "f1_std": 0.0032,
    })
    # Cross-user baseline (already run): Acc=0.1285, MacroF1=0.0624, WtdF1=0.0796
    results.append({
        "task": "T2_multiclass", "model": "MultiBranch",
        "accuracy": 0.1285, "macro_f1": 0.0624,
        "macro_f1_std": 0.0520, "weighted_f1": 0.0796,
    })
    results.append(run_t2_personalized())

    df = pd.DataFrame(results)
    out = SCRIPT_DIR / "pipeline_multibranch_results.csv"
    df.to_csv(out, index=False)

    print("\n" + "=" * 72)
    print("  FINAL RESULTS — Multi-Branch Model")
    print("=" * 72)
    print(df.to_string(index=False))
    print(f"\nResults saved: {out}")
    print("Multi-branch pipeline complete.")


if __name__ == "__main__":
    main()
