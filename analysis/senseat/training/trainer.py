"""
senseat/training/trainer.py
============================
Training pipeline with:
  - GroupKFold (participant-based) — prevents data leakage
  - Leave-One-Participant-Out (LOPO) — strongest validation
  - Class weights — fixes imbalance
  - SpecAugment — data augmentation during training
  - Early stopping + LR reduction
"""

import numpy as np
from collections import defaultdict
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, balanced_accuracy_score,
                              roc_auc_score, average_precision_score)
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from senseat.features.extractor import augment_batch
from senseat.models.architectures import get_model

SEED = 42


# ─────────────────────────────────────────
# NORMALIZATION
# ─────────────────────────────────────────

def normalize(X_train, X_test):
    """Z-score normalize using train statistics only."""
    mean = X_train.mean()
    std  = X_train.std() if X_train.std() > 1e-8 else 1.0
    return (X_train - mean) / std, (X_test - mean) / std


# ─────────────────────────────────────────
# CLASS WEIGHTS
# ─────────────────────────────────────────

def get_class_weights(y):
    """Compute balanced class weights to handle imbalance."""
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    return dict(zip(classes.tolist(), weights.tolist()))


# ─────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────

def get_callbacks(patience_es=7, patience_lr=4):
    return [
        EarlyStopping(patience=patience_es, restore_best_weights=True,
                      monitor='val_loss', verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=patience_lr, min_lr=1e-6, verbose=0)
    ]


# ─────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────

def compute_metrics(y_true, y_pred, y_prob=None, is_binary=True):
    """Compute full set of evaluation metrics."""
    m = {
        'accuracy':          accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'f1':                f1_score(y_true, y_pred, average='binary' if is_binary else 'weighted',
                                      zero_division=0),
        'precision':         precision_score(y_true, y_pred, average='binary' if is_binary else 'weighted',
                                             zero_division=0),
        'recall':            recall_score(y_true, y_pred, average='binary' if is_binary else 'weighted',
                                          zero_division=0),
    }
    if y_prob is not None and is_binary:
        try:
            m['roc_auc']  = roc_auc_score(y_true, y_prob)
            m['pr_auc']   = average_precision_score(y_true, y_prob)
        except Exception:
            m['roc_auc'] = m['pr_auc'] = 0.0
    return m


# ─────────────────────────────────────────
# GROUP K-FOLD TRAINING
# ─────────────────────────────────────────

def train_group_kfold(X, y, groups, model_name="resnet", feature_name="stft",
                      n_splits=5, epochs=50, batch_size=32,
                      use_augment=True, is_binary=True):
    """
    Train with GroupKFold — each fold keeps all segments from a participant
    in either train or test, never both.

    X      : np.ndarray (N, H, W, 1)
    y      : np.ndarray (N,)
    groups : np.ndarray (N,) — participant IDs
    """
    gkf        = GroupKFold(n_splits=n_splits)
    fold_metrics = defaultdict(list)
    num_classes  = 1 if is_binary else len(np.unique(y))

    print(f"\n{'='*60}")
    print(f"  GroupKFold ({n_splits} folds) | {model_name} + {feature_name}")
    print(f"  Samples: {len(y)} | Classes: {num_classes if not is_binary else 2}")
    print(f"{'='*60}")

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        X_tr, X_te = normalize(X_tr, X_te)

        if use_augment:
            X_tr = augment_batch(X_tr, apply_prob=0.5)

        class_weights = get_class_weights(y_tr)
        model         = get_model(model_name, X_tr[0].shape, num_classes=num_classes)

        # Map labels for multiclass
        if not is_binary:
            unique = sorted(np.unique(y))
            lmap   = {c: i for i, c in enumerate(unique)}
            y_tr_m = np.array([lmap[c] for c in y_tr])
            y_te_m = np.array([lmap[c] for c in y_te])
        else:
            y_tr_m, y_te_m = y_tr, y_te

        model.fit(
            X_tr, y_tr_m,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            callbacks=get_callbacks(),
            class_weight=class_weights,
            verbose=0
        )

        y_prob = model.predict(X_te, verbose=0)
        if is_binary:
            y_prob_flat = y_prob.ravel()
            y_pred      = (y_prob_flat > 0.5).astype(int)
        else:
            y_prob_flat = None
            y_pred      = np.argmax(y_prob, axis=1)
            # Map back to original codes for metrics
            inv_lmap = {i: c for c, i in lmap.items()}
            y_pred   = np.array([inv_lmap[i] for i in y_pred])
            y_te_m   = y_te  # use original codes

        m = compute_metrics(y_te_m if is_binary else y_te,
                            y_pred, y_prob_flat, is_binary)

        for k, v in m.items():
            fold_metrics[k].append(v)

        print(f"  Fold {fold+1}: Acc={m['accuracy']:.4f} | "
              f"BalAcc={m['balanced_accuracy']:.4f} | F1={m['f1']:.4f}"
              + (f" | ROC-AUC={m.get('roc_auc', 0):.4f}" if is_binary else ""))

        tf.keras.backend.clear_session()

    # Aggregate
    summary = {k: (np.mean(v), np.std(v)) for k, v in fold_metrics.items()}
    print(f"\n  ── GroupKFold Summary ──")
    for k, (mean, std) in summary.items():
        print(f"  {k:20s}: {mean:.4f} ± {std:.4f}")

    return summary, fold_metrics


# ─────────────────────────────────────────
# LEAVE-ONE-PARTICIPANT-OUT (LOPO)
# ─────────────────────────────────────────

def train_lopo(X, y, groups, model_name="resnet", feature_name="stft",
               epochs=50, batch_size=32, use_augment=True, is_binary=True):
    """
    Leave-One-Participant-Out cross-validation.
    Each fold: train on 19 participants, test on 1.
    This is the strongest validation — proves model generalizes to new people.

    If LOPO accuracy >> random baseline, the model learned real eating patterns.
    If LOPO accuracy ≈ random baseline, the model memorized recording artifacts.
    """
    logo         = LeaveOneGroupOut()
    fold_metrics = defaultdict(list)
    num_classes  = 1 if is_binary else len(np.unique(y))
    unique_parts = np.unique(groups)

    print(f"\n{'='*60}")
    print(f"  LOPO ({len(unique_parts)} participants) | {model_name} + {feature_name}")
    print(f"  Samples: {len(y)} | Classes: {num_classes if not is_binary else 2}")
    print(f"{'='*60}")

    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
        test_participant = np.unique(groups[test_idx])[0]
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        # Skip if test fold has only one class
        if len(np.unique(y_te)) < 2 and is_binary:
            print(f"  Fold {fold+1} [{test_participant}]: skipped (single class in test)")
            continue

        X_tr, X_te = normalize(X_tr, X_te)

        if use_augment:
            X_tr = augment_batch(X_tr, apply_prob=0.5)

        class_weights = get_class_weights(y_tr)
        model         = get_model(model_name, X_tr[0].shape, num_classes=num_classes)

        if not is_binary:
            unique = sorted(np.unique(y))
            lmap   = {c: i for i, c in enumerate(unique)}
            y_tr_m = np.array([lmap[c] for c in y_tr])
            y_te_m = np.array([lmap[c] for c in y_te])
        else:
            y_tr_m, y_te_m = y_tr, y_te

        model.fit(
            X_tr, y_tr_m,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            callbacks=get_callbacks(),
            class_weight=class_weights,
            verbose=0
        )

        y_prob = model.predict(X_te, verbose=0)
        if is_binary:
            y_prob_flat = y_prob.ravel()
            y_pred      = (y_prob_flat > 0.5).astype(int)
        else:
            y_prob_flat = None
            y_pred      = np.argmax(y_prob, axis=1)
            inv_lmap    = {i: c for c, i in lmap.items()}
            y_pred      = np.array([inv_lmap[i] for i in y_pred])

        m = compute_metrics(y_te_m if is_binary else y_te,
                            y_pred, y_prob_flat, is_binary)

        for k, v in m.items():
            fold_metrics[k].append(v)

        print(f"  [{test_participant}]: Acc={m['accuracy']:.4f} | "
              f"BalAcc={m['balanced_accuracy']:.4f} | F1={m['f1']:.4f}"
              + (f" | ROC-AUC={m.get('roc_auc', 0):.4f}" if is_binary else ""))

        tf.keras.backend.clear_session()

    summary = {k: (np.mean(v), np.std(v)) for k, v in fold_metrics.items()}
    print(f"\n  ── LOPO Summary ──")
    for k, (mean, std) in summary.items():
        print(f"  {k:20s}: {mean:.4f} ± {std:.4f}")

    return summary, fold_metrics


# ─────────────────────────────────────────
# PERSONALIZED LOPO
# ─────────────────────────────────────────

def train_lopo_personalized(X, y, groups, model_name="resnet", feature_name="stft",
                             epochs=50, batch_size=32, use_augment=True, is_binary=True,
                             finetune_ratio=0.2, finetune_epochs=10, finetune_lr=1e-4):
    """
    Personalized LOPO cross-validation.

    For each participant p:
      1. Train a global model on all other participants (standard LOPO).
      2. Split p's data: finetune_ratio → fine-tune set, rest → test set.
      3. Fine-tune the global model on p's fine-tune set (low LR, few epochs).
      4. Evaluate on p's test set.

    This mimics deployment: the app collects a short calibration session per
    user and then adapts, dramatically improving accuracy for person-specific
    eating patterns.

    finetune_ratio : fraction of each participant's segments used for fine-tuning
                     (default 0.2 → 20% adapt, 80% test)
    finetune_epochs: epochs for fine-tune step (small to avoid overfitting)
    finetune_lr    : lower LR for fine-tuning to preserve global weights
    """
    logo         = LeaveOneGroupOut()
    fold_metrics = defaultdict(list)
    num_classes  = 1 if is_binary else len(np.unique(y))
    unique_parts = np.unique(groups)

    print(f"\n{'='*60}")
    print(f"  Personalized LOPO ({len(unique_parts)} participants) | {model_name} + {feature_name}")
    print(f"  Finetune ratio: {finetune_ratio:.0%} | Finetune epochs: {finetune_epochs}")
    print(f"{'='*60}")

    if not is_binary:
        unique   = sorted(np.unique(y))
        lmap     = {c: i for i, c in enumerate(unique)}
        inv_lmap = {i: c for c, i in lmap.items()}
    else:
        lmap = inv_lmap = None

    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
        test_participant = np.unique(groups[test_idx])[0]
        X_tr, X_te_all = X[train_idx], X[test_idx]
        y_tr, y_te_all = y[train_idx], y[test_idx]

        if len(np.unique(y_te_all)) < 2 and is_binary:
            print(f"  [{test_participant}]: skipped (single class)")
            continue

        # Split test participant's data into fine-tune and eval sets
        n_finetune = max(1, int(len(y_te_all) * finetune_ratio))
        rng        = np.random.default_rng(SEED + fold)
        ft_idx     = rng.choice(len(y_te_all), size=n_finetune, replace=False)
        eval_mask  = np.ones(len(y_te_all), dtype=bool)
        eval_mask[ft_idx] = False

        X_ft, y_ft   = X_te_all[ft_idx],   y_te_all[ft_idx]
        X_eval, y_eval = X_te_all[eval_mask], y_te_all[eval_mask]

        if len(y_eval) == 0:
            print(f"  [{test_participant}]: skipped (no eval samples after split)")
            continue

        # Normalize using training set statistics
        X_tr_n, X_ft_n   = normalize(X_tr, X_ft)
        _,      X_eval_n = normalize(X_tr, X_eval)

        if use_augment:
            X_tr_n = augment_batch(X_tr_n, apply_prob=0.5)

        # ── Step 1: Train global model ──
        class_weights = get_class_weights(y_tr)
        model = get_model(model_name, X_tr_n[0].shape, num_classes=num_classes)

        if not is_binary:
            y_tr_m  = np.array([lmap[c] for c in y_tr])
            y_ft_m  = np.array([lmap[c] for c in y_ft])
            y_eval_m = np.array([lmap[c] for c in y_eval])
        else:
            y_tr_m, y_ft_m, y_eval_m = y_tr, y_ft, y_eval

        model.fit(
            X_tr_n, y_tr_m,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            callbacks=get_callbacks(),
            class_weight=class_weights,
            verbose=0
        )

        # ── Step 2: Fine-tune on participant's own data ──
        model.optimizer.learning_rate = finetune_lr
        ft_class_weights = get_class_weights(y_ft) if len(np.unique(y_ft)) > 1 else None

        model.fit(
            X_ft_n, y_ft_m,
            epochs=finetune_epochs,
            batch_size=min(batch_size, len(y_ft_m)),
            callbacks=[EarlyStopping(patience=3, restore_best_weights=True,
                                     monitor='loss', verbose=0)],
            class_weight=ft_class_weights,
            verbose=0
        )

        # ── Step 3: Evaluate ──
        y_prob = model.predict(X_eval_n, verbose=0)
        if is_binary:
            y_prob_flat = y_prob.ravel()
            y_pred      = (y_prob_flat > 0.5).astype(int)
        else:
            y_prob_flat = None
            y_pred_idx  = np.argmax(y_prob, axis=1)
            y_pred      = np.array([inv_lmap[i] for i in y_pred_idx])

        m = compute_metrics(y_eval, y_pred, y_prob_flat if is_binary else None, is_binary)
        for k, v in m.items():
            fold_metrics[k].append(v)

        print(f"  [{test_participant}]: Acc={m['accuracy']:.4f} | "
              f"BalAcc={m['balanced_accuracy']:.4f} | F1={m['f1']:.4f}"
              + (f" | ROC-AUC={m.get('roc_auc', 0):.4f}" if is_binary else "")
              + f"  (ft={n_finetune}, eval={eval_mask.sum()})")

        tf.keras.backend.clear_session()

    summary = {k: (np.mean(v), np.std(v)) for k, v in fold_metrics.items()}
    print(f"\n  ── Personalized LOPO Summary ──")
    for k, (mean, std) in summary.items():
        print(f"  {k:20s}: {mean:.4f} ± {std:.4f}")

    return summary, fold_metrics
