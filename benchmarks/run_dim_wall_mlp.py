#!/usr/bin/env python3
"""
Standalone Benchmark: The TabPFN Dimensionality Wall (Section 4.1.3)
Path A Implementation: Weakly-correlated noise injection + random column permutation.
Evaluates SmallMLP vs. TabPFN v0.1.11 (PCA-100 & Truncate-100) across d in {35, 70, 100, 130, 200, 300}.
"""

import os
# Строго ограничиваем потоки во избежание segmentation fault / bus error на macOS
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import gc
import shutil
import sys
import time
import typing
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

def log(msg: str) -> None:
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)

log("Step 1/5: Applying PyTorch 2.2+ and scikit-learn 1.6+ compatibility shims...")

import torch
import torch.nn as nn
import torch.nn.modules.transformer as _torch_transformer

torch.set_num_threads(1)

# Monkey-patch PyTorch 2.2+ removed re-exports for tabpfn==0.1.11
_torch_transformer.Optional = typing.Optional
_torch_transformer.Tensor = torch.Tensor
_torch_transformer.Module = nn.Module
_torch_transformer.Linear = nn.Linear
_torch_transformer.Dropout = nn.Dropout
_torch_transformer.LayerNorm = nn.LayerNorm
_torch_transformer.MultiheadAttention = nn.MultiheadAttention

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.utils.validation as _sk_val
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tabpfn import TabPFNClassifier
import tabpfn.scripts.transformer_prediction_interface as _tpi

# Ensure local project root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from smallmlp import SmallMLPClassifier

# Monkey-patch scikit-learn 1.6+ (force_all_finite -> ensure_all_finite)
_orig_check_X_y = _sk_val.check_X_y
_orig_check_array = _sk_val.check_array

def _patched_check_X_y(*args, **kwargs):
    if "force_all_finite" in kwargs:
        kwargs["ensure_all_finite"] = kwargs.pop("force_all_finite")
    return _orig_check_X_y(*args, **kwargs)

def _patched_check_array(*args, **kwargs):
    if "force_all_finite" in kwargs:
        kwargs["ensure_all_finite"] = kwargs.pop("force_all_finite")
    return _orig_check_array(*args, **kwargs)

_tpi.check_X_y = _patched_check_X_y
_tpi.check_array = _patched_check_array

log("Step 2/5: Initializing and warming up TabPFN v0.1.11 on CPU...")
t_warmup = time.time()
pfn_model = TabPFNClassifier(device="cpu", N_ensemble_configurations=8)
pfn_model.fit(
    np.random.randn(20, 4).astype(np.float32),
    np.array([0, 1] * 10),
    overwrite_warning=True,
)
_ = pfn_model.predict(np.random.randn(5, 4).astype(np.float32))
log(f"  -> TabPFN warm-up complete in {time.time() - t_warmup:.2f}s.")


def load_uci_datasets():
    log("Step 3/5: Downloading/loading 4 UCI classification datasets...")
    datasets = {}

    # 1. Connectionist Bench Sonar (UCI: n=208, d=60)
    log("  -> Fetching UCI Sonar...")
    url_sonar = "https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data"
    df_sonar = pd.read_csv(url_sonar, header=None)
    y_sonar = LabelEncoder().fit_transform(df_sonar.iloc[:, -1].astype(str))
    X_sonar = df_sonar.iloc[:, :-1].values.astype(np.float32)
    datasets["Sonar"] = (X_sonar, y_sonar)

    # 2. SPECTF Heart (UCI: n=267, d=44)
    log("  -> Fetching UCI SPECTF Heart...")
    url_spectf_tr = "https://archive.ics.uci.edu/ml/machine-learning-databases/spect/SPECTF.train"
    url_spectf_te = "https://archive.ics.uci.edu/ml/machine-learning-databases/spect/SPECTF.test"
    df_spectf = pd.concat(
        [
            pd.read_csv(url_spectf_tr, header=None),
            pd.read_csv(url_spectf_te, header=None),
        ],
        ignore_index=True,
    )
    y_spectf = LabelEncoder().fit_transform(df_spectf.iloc[:, 0].astype(str))
    X_spectf = df_spectf.iloc[:, 1:].values.astype(np.float32)
    datasets["SPECTF_Heart"] = (X_spectf, y_spectf)

    # 3. Heart Disease Cleveland (UCI: n=303, d=13)
    log("  -> Fetching UCI Heart Disease Cleveland...")
    url_heart = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    df_heart = pd.read_csv(url_heart, header=None, na_values="?")
    y_heart = (df_heart.iloc[:, -1].values.astype(float) > 0).astype(int)
    X_heart = SimpleImputer(strategy="median").fit_transform(
        df_heart.iloc[:, :-1].values.astype(np.float32)
    )
    datasets["Heart_Cleveland"] = (X_heart, y_heart)

    # 4. Glass Identification (UCI: n=214, d=9)
    log("  -> Fetching UCI Glass Identification...")
    url_glass = "https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data"
    df_glass = pd.read_csv(url_glass, header=None)
    y_glass = LabelEncoder().fit_transform(df_glass.iloc[:, -1].astype(str))
    X_glass = df_glass.iloc[:, 1:-1].values.astype(np.float32)
    datasets["Glass"] = (X_glass, y_glass)

    for name, (X_d, y_d) in datasets.items():
        log(
            f"  [OK] {name:16s} | n={X_d.shape[0]:3d}, d={X_d.shape[1]:2d}, K={len(np.unique(y_d))}"
        )
    return datasets


def main():
    uci_sets = load_uci_datasets()

    target_dims = [35, 70, 100, 130, 200, 300]
    models = [
        "SmallMLP",
        "TabPFN (PCA-100 >100d)",
        "TabPFN (Truncate-100 >100d)",
    ]

    per_ds_results = {
        m: np.zeros((len(uci_sets), len(target_dims))) for m in models
    }
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    log("Step 4/5: Starting Dimensionality Wall Sweep (Path A: Weakly-Correlated Noise + Permutation)...")
    t_total = time.time()

    for ds_idx, (ds_name, (X_base, y)) in enumerate(uci_sets.items()):
        n_samples, d_base = X_base.shape
        print("=" * 78, flush=True)
        log(f"DATASET [{ds_idx + 1}/{len(uci_sets)}]: {ds_name} (n={n_samples}, d_base={d_base})")
        print("=" * 78, flush=True)

        for idx_d, d_target in enumerate(target_dims):
            step_t = time.time()
            rng = np.random.RandomState(42 + d_target)
            X_std_base = StandardScaler().fit_transform(X_base).astype(np.float32)

            if d_target > d_base:
                n_extra = d_target - d_base
                # Weakly-correlated noise features: prevents signal duplication across all 300 columns
                # so random subsampling of 100 columns genuinely loses (1 - 100/d_target) of true signal
                W = rng.normal(
                    0.0, 0.05 / np.sqrt(d_base), size=(d_base, n_extra)
                ).astype(np.float32)
                orth_noise = rng.normal(
                    0.0, 1.0, size=(n_samples, n_extra)
                ).astype(np.float32)
                injected = X_std_base @ W + orth_noise
                X = np.hstack([X_std_base, injected])
                # Permute columns so true features are uniformly dispersed across all d_target columns
                perm_idx = rng.permutation(d_target)
                X = X[:, perm_idx]
            else:
                perm_idx = rng.permutation(d_base)[:d_target]
                X = X_std_base[:, perm_idx]

            fold_scores = {m: [] for m in models}

            for fold_i, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
                print(
                    f"  [d={d_target:3d}] Fold {fold_i}/{n_splits}: ",
                    end="",
                    flush=True,
                )
                X_tr, X_te = X[train_idx], X[test_idx]
                y_tr, y_te = y[train_idx], y[test_idx]

                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_tr).astype(np.float32)
                X_te_s = scaler.transform(X_te).astype(np.float32)

                # 1. SmallMLP
                print("SmallMLP..", end="", flush=True)
                mlp = SmallMLPClassifier(alpha=6.0, beta=0.7, random_state=42, width_mode="formula")
                mlp.fit(X_tr_s, y_tr)
                fold_scores["SmallMLP"].append(
                    accuracy_score(y_te, mlp.predict(X_te_s))
                )

                # 2. TabPFN (PCA-100 when d > 100)
                print("PFN-PCA..", end="", flush=True)
                if d_target > 100:
                    pca = PCA(n_components=100, random_state=42)
                    X_tr_p = pca.fit_transform(X_tr_s).astype(np.float32)
                    X_te_p = pca.transform(X_te_s).astype(np.float32)
                else:
                    X_tr_p, X_te_p = X_tr_s, X_te_s
                pfn_model.fit(X_tr_p, y_tr, overwrite_warning=True)
                pfn_pca_acc = accuracy_score(y_te, pfn_model.predict(X_te_p))
                fold_scores["TabPFN (PCA-100 >100d)"].append(pfn_pca_acc)

                # 3. TabPFN (Truncate-100 when d > 100; reuse when d <= 100)
                print("PFN-Trunc..", end="", flush=True)
                if d_target > 100:
                    # Since X columns were randomly permuted, [:100] is a uniform random subsample of 100 features
                    pfn_model.fit(X_tr_s[:, :100], y_tr, overwrite_warning=True)
                    pfn_sub_acc = accuracy_score(
                        y_te, pfn_model.predict(X_te_s[:, :100])
                    )
                else:
                    pfn_sub_acc = pfn_pca_acc
                fold_scores["TabPFN (Truncate-100 >100d)"].append(pfn_sub_acc)

                print("done", flush=True)
                gc.collect()

            for m in models:
                per_ds_results[m][ds_idx, idx_d] = np.mean(fold_scores[m])

            log(
                f"  -> Summary d={d_target:3d} ({time.time() - step_t:.1f}s) | "
                f"SmallMLP: {per_ds_results['SmallMLP'][ds_idx, idx_d]:.4f} | "
                f"PFN-PCA: {per_ds_results['TabPFN (PCA-100 >100d)'][ds_idx, idx_d]:.4f} | "
                f"PFN-Trunc: {per_ds_results['TabPFN (Truncate-100 >100d)'][ds_idx, idx_d]:.4f}"
            )

    log(f"Step 5/5: All datasets finished in {time.time() - t_total:.1f}s. Saving outputs...")

    out_dir = Path(__file__).resolve().parent
    mean_results = {m: np.mean(per_ds_results[m], axis=0) for m in models}
    df_curve = pd.DataFrame(mean_results, index=target_dims)
    df_curve.index.name = "Dimensionality (d)"

    csv_path = out_dir / "tabpfn_dim_wall_uci.csv"
    df_curve.to_csv(csv_path)

    print("\n" + "=" * 78, flush=True)
    print("=== FINAL MEAN 5-FOLD ACCURACY ACROSS 4 UCI DATASETS ===", flush=True)
    print("=" * 78, flush=True)
    print(df_curve.round(4).to_string(), flush=True)
    print("=" * 78, flush=True)

    # Generate publication-grade plot matching Section 4.1.3
    plt.figure(figsize=(7.2, 4.5), dpi=300)
    styles = {
        "SmallMLP": ("#1f77b4", "o", "-", 2.5),
        "TabPFN (PCA-100 >100d)": ("#d62728", "^", "--", 2.2),
        "TabPFN (Truncate-100 >100d)": ("#9467bd", "s", "-.", 2.2),
    }

    for name, (color, marker, ls, lw) in styles.items():
        plt.plot(
            target_dims,
            mean_results[name],
            label=name,
            color=color,
            marker=marker,
            linestyle=ls,
            linewidth=lw,
            markersize=6.5,
        )

    plt.axvline(x=100, color="#d62728", linestyle=":", linewidth=1.8, alpha=0.85)
    ymin, ymax = plt.ylim()

    plt.xlabel(r"Feature Dimensionality ($d$)", fontsize=11.5, labelpad=8)
    plt.ylabel("Mean 5-Fold Test Accuracy", fontsize=11.5, labelpad=8)
    plt.xticks(target_dims, fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(frameon=True, fontsize=9.5, loc="lower left")

    plt.tight_layout()

    # Save to BOTH filenames and copy to project root so LaTeX picks up the new figure automatically
    for fname in ["mlp_dim_wall.pdf", "tabpfn_dim_wall.pdf", "mlp_dim_wall.png", "tabpfn_dim_wall.png"]:
        fpath = out_dir / fname
        plt.savefig(fpath, bbox_inches="tight", dpi=300)
        root_fpath = ROOT_DIR / fname
        if fpath != root_fpath:
            shutil.copyfile(fpath, root_fpath)
        log(f"Saved figure to: {fpath} (and copied to {root_fpath})")


if __name__ == "__main__":
    main()