"""
SensEat — Comprehensive Grid Search Pipeline
============================================
Binary   T1: RF | SVM | CNN-STFT | CNN-Multi  ×  {41p, 20p}  ×  thresholds 0.5–0.9
Multiclass T2: RF | SVM | CNN-STFT | CNN-Multi  ×  {20p, 41p}

Outputs:
    pipeline_grid_binary_results.csv
    pipeline_grid_multiclass_results.csv
    figures/grid_search/*.png
"""

import os, warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (accuracy_score, f1_score,
                              precision_score, recall_score, confusion_matrix)

import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

SEED     = 42
N_SPLITS = 5
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]
RNG = np.random.default_rng(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

SCRIPT_DIR = Path(__file__).resolve().parent
FIG_DIR    = SCRIPT_DIR / "figures" / "grid_search"
FIG_DIR.mkdir(parents=True, exist_ok=True)

BINARY_CACHE     = SCRIPT_DIR / "feature_cache_binary_v4.npz"
MULTICLASS_CACHE = SCRIPT_DIR / "feature_cache_multiclass_balanced_v2.npz"

FOOD_NAMES = {
    1: "Tortilla", 2: "Mandarin",  3: "Chicken",
    4: "Cheeze_It", 5: "Carrots",  6: "Chocolate",
    7: "Yogurt",   8: "Noodles",   9: "Water", 10: "Coke",
}


# ─── HELPERS ────────────────────────────────────────────────────────────────
def undersample_binary(Xd, y):
    eat = np.where(y == 1)[0]; idl = np.where(y == 0)[0]
    if len(eat) <= len(idl):
        return Xd, y
    chosen = RNG.choice(eat, size=len(idl), replace=False)
    idx    = np.concatenate([chosen, idl]); RNG.shuffle(idx)
    return {k: v[idx] for k, v in Xd.items()}, y[idx]


def undersample_mc(Xd, y):
    classes, counts = np.unique(y, return_counts=True)
    n_min = counts.min(); sel = []
    for c in classes:
        idx = np.where(y == c)[0]
        sel.append(RNG.choice(idx, size=n_min, replace=False))
    idx = np.concatenate(sel); RNG.shuffle(idx)
    return {k: v[idx] for k, v in Xd.items()}, y[idx]


def norm_all(Xtr, Xte):
    for key in Xtr:
        mn = Xtr[key].mean(); sd = Xtr[key].std() or 1.0
        Xtr[key] = (Xtr[key] - mn) / sd
        Xte[key] = (Xte[key] - mn) / sd


# ─── CNN BUILDERS ───────────────────────────────────────────────────────────
def _cnn2d(x, name):
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)
    return layers.Dense(128, activation='relu', name=name)(x)


def _cnn1d(x, n, name):
    x = layers.Reshape((n, 1))(x)
    x = layers.Conv1D(32, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(64, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(128, 3, activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling1D()(x)
    return layers.Dense(128, activation='relu', name=name)(x)


def build_stft_cnn(stft_shape, n_classes, task):
    inp = Input(shape=stft_shape)
    x   = _cnn2d(inp, 'lat')
    x   = layers.Dropout(0.4)(x)
    if task == 'binary':
        out = layers.Dense(1, activation='sigmoid')(x)
        m = models.Model(inp, out)
        m.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    else:
        out = layers.Dense(n_classes, activation='softmax')(x)
        m = models.Model(inp, out)
        m.compile('adam', 'sparse_categorical_crossentropy', metrics=['accuracy'])
    return m


def build_multi_cnn(shapes, n_flat, n_classes, task):
    stft_in = Input(shape=shapes[0], name='stft_in')
    mfcc_in = Input(shape=shapes[1], name='mfcc_in')
    gfcc_in = Input(shape=shapes[2], name='gfcc_in')
    flat_in = Input(shape=(n_flat,),  name='flat_in')

    ls = _cnn2d(stft_in, 'l_stft')
    lm = _cnn2d(mfcc_in, 'l_mfcc')
    lg = _cnn2d(gfcc_in, 'l_gfcc')
    lf = _cnn1d(flat_in, n_flat, 'l_flat')

    cat  = layers.Concatenate()([ls, lm, lg, lf])
    attn = layers.Dense(4, activation='softmax')(cat)
    ws   = layers.Lambda(lambda x: x[0] * tf.expand_dims(x[1][:,0], 1))([ls, attn])
    wm   = layers.Lambda(lambda x: x[0] * tf.expand_dims(x[1][:,1], 1))([lm, attn])
    wg   = layers.Lambda(lambda x: x[0] * tf.expand_dims(x[1][:,2], 1))([lg, attn])
    wf   = layers.Lambda(lambda x: x[0] * tf.expand_dims(x[1][:,3], 1))([lf, attn])
    fused = layers.Add()([ws, wm, wg, wf])

    x = layers.Dense(64, activation='relu')(fused)
    x = layers.Dropout(0.4)(x)
    if task == 'binary':
        out = layers.Dense(1, activation='sigmoid')(x)
        m = models.Model([stft_in, mfcc_in, gfcc_in, flat_in], out)
        m.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    else:
        out = layers.Dense(n_classes, activation='softmax')(x)
        m = models.Model([stft_in, mfcc_in, gfcc_in, flat_in], out)
        m.compile('adam', 'sparse_categorical_crossentropy', metrics=['accuracy'])
    return m


# ─── DATA LOADERS ───────────────────────────────────────────────────────────
def load_binary(new_only=False):
    d = np.load(BINARY_CACHE, allow_pickle=False)
    X = {k: d[f'X_{k}'].astype(np.float32) for k in ['stft','mfcc','gfcc','flat']}
    y = d['y'].astype(np.int32)
    g = d['groups']
    if new_only:
        eat_idx = np.where((g >= 22) & (g <= 41))[0]
        idle_tr = np.where(g == -1)[0]
        idle_te = np.where(g == -2)[0]
        keep    = np.concatenate([eat_idx, idle_tr, idle_te])
        X = {k: v[keep] for k, v in X.items()}
        y = y[keep]; g = g[keep]
    return X, y, g


def load_multiclass(new_only=False):
    d = np.load(MULTICLASS_CACHE, allow_pickle=False)
    X = {k: d[f'X_{k}'].astype(np.float32) for k in ['stft','mfcc','gfcc','flat']}
    y = d['y'].astype(np.int32)
    g = d['groups']
    if new_only:
        mask = (g >= 22) & (g <= 41)
        X = {k: v[mask] for k, v in X.items()}
        y = y[mask]; g = g[mask]
    return X, y, g


def get_binary_folds(g):
    IDLE_TR, IDLE_TE = -1, -2
    idle_tr = np.where(g == IDLE_TR)[0]
    idle_te = np.where(g == IDLE_TE)[0]
    real    = np.where((g != IDLE_TR) & (g != IDLE_TE))[0]
    gkf = GroupKFold(n_splits=N_SPLITS)
    splits = []
    for tr, te in gkf.split(real, groups=g[real]):
        splits.append((
            np.concatenate([real[tr], idle_tr]),
            np.concatenate([real[te], idle_te])
        ))
    return splits


# ─── BINARY RUNNERS ─────────────────────────────────────────────────────────
def binary_classical(X, y, g, clf_name, tag):
    print(f"  [Binary/{tag}] {clf_name} ...", flush=True)
    folds = get_binary_folds(g)
    all_t, all_p = [], []
    for tr, te in folds:
        Xtr = {'flat': X['flat'][tr].copy()}; ytr = y[tr]
        Xte = {'flat': X['flat'][te].copy()}; yte = y[te]
        Xtr, ytr = undersample_binary(Xtr, ytr)
        sc = StandardScaler()
        Xtr['flat'] = sc.fit_transform(Xtr['flat'])
        Xte['flat'] = sc.transform(Xte['flat'])
        if clf_name == 'RF':
            clf = RandomForestClassifier(200, random_state=SEED, n_jobs=-1, class_weight='balanced')
        else:
            clf = SVC(kernel='rbf', C=10, gamma='scale', random_state=SEED, class_weight='balanced')
        clf.fit(Xtr['flat'], ytr)
        yp = clf.predict(Xte['flat'])
        all_t.extend(yte); all_p.extend(yp)

    acc = accuracy_score(all_t, all_p)
    p   = precision_score(all_t, all_p, zero_division=0)
    r   = recall_score(all_t, all_p, zero_division=0)
    f1  = f1_score(all_t, all_p, zero_division=0)
    print(f"    Acc={acc:.4f}  P={p:.4f}  R={r:.4f}  F1={f1:.4f}")
    return [{"tag": tag, "model": clf_name, "threshold": "—",
             "acc": round(acc,4), "precision": round(p,4),
             "recall": round(r,4), "f1": round(f1,4)}]


def binary_cnn(X, y, g, arch, tag, epochs=30):
    print(f"  [Binary/{tag}] CNN-{arch} ({epochs} ep) + threshold sweep ...", flush=True)
    folds = get_binary_folds(g)
    all_probs, all_true = [], []

    for fi, (tr, te) in enumerate(folds):
        Xtr = {k: v[tr].copy() for k, v in X.items()}; ytr = y[tr]
        Xte = {k: v[te].copy() for k, v in X.items()}; yte = y[te]
        Xtr, ytr = undersample_binary(Xtr, ytr)
        norm_all(Xtr, Xte)

        cb = [EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
              ReduceLROnPlateau(monitor='val_loss', patience=4, factor=0.5, verbose=0)]

        if arch == 'STFT':
            m = build_stft_cnn(Xtr['stft'].shape[1:], 1, 'binary')
            itr = Xtr['stft']; ite = Xte['stft']
        else:
            m = build_multi_cnn(
                [Xtr['stft'].shape[1:], Xtr['mfcc'].shape[1:], Xtr['gfcc'].shape[1:]],
                Xtr['flat'].shape[1], 1, 'binary')
            itr = [Xtr['stft'], Xtr['mfcc'], Xtr['gfcc'], Xtr['flat']]
            ite = [Xte['stft'], Xte['mfcc'], Xte['gfcc'], Xte['flat']]

        m.fit(itr, ytr, epochs=epochs, batch_size=32,
              validation_split=0.1, callbacks=cb, verbose=0)
        probs = m.predict(ite, verbose=0).ravel()
        tf.keras.backend.clear_session()

        all_probs.append(probs); all_true.append(yte)
        print(f"    Fold {fi} done.", flush=True)

    probs_cat = np.concatenate(all_probs)
    true_cat  = np.concatenate(all_true)

    rows = []
    best_f1 = -1
    for t in THRESHOLDS:
        yp  = (probs_cat > t).astype(int)
        acc = accuracy_score(true_cat, yp)
        p   = precision_score(true_cat, yp, zero_division=0)
        r   = recall_score(true_cat, yp, zero_division=0)
        f1  = f1_score(true_cat, yp, zero_division=0)
        rows.append({"tag": f"{tag}_t{t}", "model": f"CNN-{arch}", "threshold": t,
                     "acc": round(acc,4), "precision": round(p,4),
                     "recall": round(r,4), "f1": round(f1,4)})
        marker = " <-- BEST" if f1 > best_f1 else ""
        print(f"    t={t}: Acc={acc:.4f}  P={p:.4f}  R={r:.4f}  F1={f1:.4f}{marker}")
        if f1 > best_f1:
            best_f1 = f1

    return rows


# ─── MULTICLASS RUNNERS ─────────────────────────────────────────────────────
def mc_remap(y):
    codes = sorted(np.unique(y))
    c2i   = {c: i for i, c in enumerate(codes)}
    i2c   = {i: c for c, i in c2i.items()}
    return np.array([c2i[c] for c in y]), c2i, i2c


def mc_classical(X, y, g, clf_name, tag):
    print(f"  [Multiclass/{tag}] {clf_name} ...", flush=True)
    yi, c2i, i2c = mc_remap(y)
    gkf = GroupKFold(n_splits=N_SPLITS)
    all_t, all_p = [], []

    for tr, te in gkf.split(X['flat'], groups=g):
        Xtr = {'flat': X['flat'][tr].copy()}; ytr = yi[tr]
        Xte = {'flat': X['flat'][te].copy()}; yte = y[te]
        Xtr, ytr = undersample_mc(Xtr, ytr)
        sc = StandardScaler()
        Xtr['flat'] = sc.fit_transform(Xtr['flat'])
        Xte['flat'] = sc.transform(Xte['flat'])
        if clf_name == 'RF':
            clf = RandomForestClassifier(200, random_state=SEED, n_jobs=-1)
        else:
            clf = SVC(kernel='rbf', C=10, gamma='scale', random_state=SEED)
        clf.fit(Xtr['flat'], ytr)
        yp_i = clf.predict(Xte['flat'])
        yp   = np.array([i2c[i] for i in yp_i])
        all_t.extend(yte); all_p.extend(yp)

    acc    = accuracy_score(all_t, all_p)
    macro  = f1_score(all_t, all_p, average='macro',    zero_division=0)
    wtd    = f1_score(all_t, all_p, average='weighted', zero_division=0)
    print(f"    Acc={acc:.4f}  MacroF1={macro:.4f}  WtdF1={wtd:.4f}")
    return [{"tag": tag, "model": clf_name,
             "acc": round(acc,4), "macro_f1": round(macro,4), "weighted_f1": round(wtd,4)}]


def mc_cnn(X, y, g, arch, tag, epochs=50):
    print(f"  [Multiclass/{tag}] CNN-{arch} ({epochs} ep) ...", flush=True)
    yi, c2i, i2c = mc_remap(y)
    n_classes = len(c2i)
    gkf = GroupKFold(n_splits=N_SPLITS)
    all_t, all_p = [], []
    fold_mf1 = []

    for fi, (tr, te) in enumerate(gkf.split(X['stft'], groups=g)):
        Xtr = {k: v[tr].copy() for k, v in X.items()}; ytr = yi[tr]
        Xte = {k: v[te].copy() for k, v in X.items()}; yte = y[te]
        Xtr, ytr = undersample_mc(Xtr, ytr)
        norm_all(Xtr, Xte)

        cb = [EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
              ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, verbose=0)]

        if arch == 'STFT':
            m   = build_stft_cnn(Xtr['stft'].shape[1:], n_classes, 'multiclass')
            itr = Xtr['stft']; ite = Xte['stft']
        else:
            m = build_multi_cnn(
                [Xtr['stft'].shape[1:], Xtr['mfcc'].shape[1:], Xtr['gfcc'].shape[1:]],
                Xtr['flat'].shape[1], n_classes, 'multiclass')
            itr = [Xtr['stft'], Xtr['mfcc'], Xtr['gfcc'], Xtr['flat']]
            ite = [Xte['stft'], Xte['mfcc'], Xte['gfcc'], Xte['flat']]

        m.fit(itr, ytr, epochs=epochs, batch_size=32,
              validation_split=0.1, callbacks=cb, verbose=0)
        yp_i = np.argmax(m.predict(ite, verbose=0), axis=1)
        yp   = np.array([i2c[i] for i in yp_i])
        tf.keras.backend.clear_session()

        mf1 = f1_score(yte, yp, average='macro', zero_division=0)
        fold_mf1.append(mf1)
        all_t.extend(yte); all_p.extend(yp)
        print(f"    Fold {fi}: MacroF1={mf1:.4f}", flush=True)

    acc   = accuracy_score(all_t, all_p)
    macro = float(np.mean(fold_mf1))
    wtd   = f1_score(all_t, all_p, average='weighted', zero_division=0)
    print(f"    Final: Acc={acc:.4f}  MacroF1={macro:.4f}  WtdF1={wtd:.4f}")
    return [{"tag": tag, "model": f"CNN-{arch}",
             "acc": round(acc,4), "macro_f1": round(macro,4), "weighted_f1": round(wtd,4)}]


# ─── VISUALISE ──────────────────────────────────────────────────────────────
def plot_binary_summary(df):
    # Show F1 for every experiment (best threshold per CNN)
    best_per_model = (df.sort_values('f1', ascending=False)
                        .drop_duplicates(subset=['model'])
                        .reset_index(drop=True))
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ['#4C72B0' if 'RF' in r else
              '#DD8452' if 'SVM' in r else
              '#55A868' if 'STFT' in r else '#C44E52'
              for r in best_per_model['model']]
    bars = ax.bar(best_per_model['model'], best_per_model['f1'], color=colors)
    for b, v in zip(bars, best_per_model['f1']):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.003,
                f"{v:.3f}", ha='center', va='bottom', fontsize=9)
    ax.set_ylim(0, 1.1); ax.set_ylabel('F1'); ax.set_title('Binary: Best F1 per Model')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.xticks(rotation=30, ha='right'); plt.tight_layout()
    plt.savefig(FIG_DIR / 'binary_summary.png', dpi=150); plt.close()


def plot_mc_summary(df):
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#4C72B0' if 'RF' in r else
              '#DD8452' if 'SVM' in r else '#55A868'
              for r in df['model']]
    bars = ax.bar(df['tag'], df['macro_f1'], color=colors)
    for b, v in zip(bars, df['macro_f1']):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.002,
                f"{v:.3f}", ha='center', va='bottom', fontsize=9)
    ax.set_ylim(0, max(df['macro_f1'].max() + 0.1, 0.5))
    ax.set_ylabel('Macro F1'); ax.set_title('Multiclass: Macro F1 per Configuration')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.xticks(rotation=30, ha='right'); plt.tight_layout()
    plt.savefig(FIG_DIR / 'multiclass_summary.png', dpi=150); plt.close()


# ─── MAIN ───────────────────────────────────────────────────────────────────
def main():
    print("=" * 72)
    print("  SensEat — Comprehensive Grid Search")
    print("=" * 72)

    # ── BINARY ──────────────────────────────────────────────────────────────
    print("\n\n" + "=" * 72)
    print("  BINARY EXPERIMENTS")
    print("=" * 72)

    Xb_all, yb_all, gb_all = load_binary(new_only=False)   # 41p
    Xb_new, yb_new, gb_new = load_binary(new_only=True)    # 20p

    n41 = int(np.sum((gb_all >= 22) | (gb_all < 0)))
    print(f"\n  41p: {len(yb_all)} segments | 20p: {len(yb_new)} segments")

    bin_rows = []
    bin_rows += binary_classical(Xb_all, yb_all, gb_all, 'RF',  '41p')
    bin_rows += binary_classical(Xb_all, yb_all, gb_all, 'SVM', '41p')
    bin_rows += binary_classical(Xb_new, yb_new, gb_new, 'RF',  '20p')
    bin_rows += binary_classical(Xb_new, yb_new, gb_new, 'SVM', '20p')
    bin_rows += binary_cnn(Xb_all, yb_all, gb_all, 'STFT',  '41p', epochs=30)
    bin_rows += binary_cnn(Xb_all, yb_all, gb_all, 'Multi', '41p', epochs=30)
    bin_rows += binary_cnn(Xb_new, yb_new, gb_new, 'STFT',  '20p', epochs=30)
    bin_rows += binary_cnn(Xb_new, yb_new, gb_new, 'Multi', '20p', epochs=30)

    df_bin = pd.DataFrame(bin_rows).sort_values('f1', ascending=False).reset_index(drop=True)
    out_bin = SCRIPT_DIR / "pipeline_grid_binary_results.csv"
    df_bin.to_csv(out_bin, index=False)
    plot_binary_summary(df_bin)

    print(f"\n  TOP-5 BINARY:")
    print(df_bin.head(5).to_string(index=False))

    # ── MULTICLASS ──────────────────────────────────────────────────────────
    print("\n\n" + "=" * 72)
    print("  MULTICLASS EXPERIMENTS")
    print("=" * 72)

    Xmc_new, ymc_new, gmc_new = load_multiclass(new_only=True)    # 20p
    Xmc_all, ymc_all, gmc_all = load_multiclass(new_only=False)   # 41p

    codes_20p = sorted(np.unique(ymc_new).tolist())
    codes_41p = sorted(np.unique(ymc_all).tolist())
    print(f"\n  20p classes ({len(codes_20p)}): {[FOOD_NAMES.get(c,c) for c in codes_20p]}")
    print(f"  41p classes ({len(codes_41p)}): {[FOOD_NAMES.get(c,c) for c in codes_41p]}")

    mc_rows = []
    mc_rows += mc_classical(Xmc_new, ymc_new, gmc_new, 'RF',  '20p')
    mc_rows += mc_classical(Xmc_new, ymc_new, gmc_new, 'SVM', '20p')
    mc_rows += mc_classical(Xmc_all, ymc_all, gmc_all, 'RF',  '41p')
    mc_rows += mc_classical(Xmc_all, ymc_all, gmc_all, 'SVM', '41p')
    mc_rows += mc_cnn(Xmc_new, ymc_new, gmc_new, 'STFT',  '20p', epochs=50)
    mc_rows += mc_cnn(Xmc_new, ymc_new, gmc_new, 'Multi', '20p', epochs=50)
    mc_rows += mc_cnn(Xmc_all, ymc_all, gmc_all, 'STFT',  '41p', epochs=50)

    df_mc = pd.DataFrame(mc_rows).sort_values('macro_f1', ascending=False).reset_index(drop=True)
    out_mc = SCRIPT_DIR / "pipeline_grid_multiclass_results.csv"
    df_mc.to_csv(out_mc, index=False)
    plot_mc_summary(df_mc)

    print(f"\n  TOP-5 MULTICLASS:")
    print(df_mc.head(5).to_string(index=False))

    # ── FINAL SUMMARY ───────────────────────────────────────────────────────
    print("\n\n" + "=" * 72)
    print("  FINAL BEST RESULTS")
    print("=" * 72)

    best_bin = df_bin.iloc[0]
    best_mc  = df_mc.iloc[0]

    print(f"\n  BEST BINARY   : {best_bin['tag']} | {best_bin['model']} "
          f"| threshold={best_bin['threshold']}")
    print(f"    Acc={best_bin['acc']:.4f}  P={best_bin['precision']:.4f}  "
          f"R={best_bin['recall']:.4f}  F1={best_bin['f1']:.4f}")

    print(f"\n  BEST MULTICLASS: {best_mc['tag']} | {best_mc['model']}")
    print(f"    Acc={best_mc['acc']:.4f}  MacroF1={best_mc['macro_f1']:.4f}  "
          f"WtdF1={best_mc['weighted_f1']:.4f}")

    print(f"\n  Results: {out_bin.name}  |  {out_mc.name}")
    print("  Grid search complete.")


if __name__ == "__main__":
    main()
