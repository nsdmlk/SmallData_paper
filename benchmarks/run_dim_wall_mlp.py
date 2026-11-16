#!/usr/bin/env python3
"""
Benchmark Part 2: MLP Models (SmallMLP, TabPFN)
Evaluates PyTorch models across d in {35, 70, 100, 130, 200, 300} on 4 UCI datasets.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
import time
import typing
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.modules.transformer as _torch_transformer

# Monkey-patch for TabPFN 0.1.11 compatibility
_torch_transformer.Optional = typing.Optional
_torch_transformer.Tensor = torch.Tensor
_torch_transformer.Module = nn.Module
_torch_transformer.Linear = nn.Linear
_torch_transformer.Dropout = nn.Dropout
_torch_transformer.LayerNorm = nn.LayerNorm
_torch_transformer.MultiheadAttention = nn.MultiheadAttention

import sklearn.utils.validation as _sk_val
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tabpfn import TabPFNClassifier
import tabpfn.scripts.transformer_prediction_interface as _tpi

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from smallmlp import SmallMLPClassifier

# Compatibility patch for scikit-learn
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

warnings.filterwarnings("ignore")

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def load_uci_datasets():
    log("Loading UCI datasets...")
    datasets = {}
    
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data"
    df = pd.read_csv(url, header=None)
    datasets["Sonar"] = (df.iloc[:, :-1].values.astype(float), LabelEncoder().fit_transform(df.iloc[:, -1].astype(str)))
    
    url_tr = "https://archive.ics.uci.edu/ml/machine-learning-databases/spect/SPECTF.train"
    url_te = "https://archive.ics.uci.edu/ml/machine-learning-databases/spect/SPECTF.test"
    df = pd.concat([pd.read_csv(url_tr, header=None), pd.read_csv(url_te, header=None)], ignore_index=True)
    datasets["SPECTF_Heart"] = (df.iloc[:, 1:].values.astype(float), LabelEncoder().fit_transform(df.iloc[:, 0].astype(str)))
    
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    df = pd.read_csv(url, header=None, na_values="?")
    X = SimpleImputer(strategy="median").fit_transform(df.iloc[:, :-1].values.astype(float))
    y = (df.iloc[:, -1].values.astype(float) > 0).astype(int)
    datasets["Heart_Cleveland"] = (X, y)
    
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data"
    df = pd.read_csv(url, header=None)
    datasets["Glass"] = (df.iloc[:, 1:-1].values.astype(float), LabelEncoder().fit_transform(df.iloc[:, -1].astype(str)))
    
    return datasets

def main():
    log("Warming up TabPFN...")
    pfn_model = TabPFNClassifier(device="cpu", N_ensemble_configurations=8)
    pfn_model.fit(np.random.randn(20, 4), np.array([0, 1] * 10), overwrite_warning=True)
    
    uci_sets = load_uci_datasets()
    target_dims = [35, 70, 100, 130, 200, 300]
    models = ["SmallMLP", "TabPFN (PCA-100 >100d)", "TabPFN (Truncate-100 >100d)"]
    
    per_ds_results = {m: np.zeros((len(uci_sets), len(target_dims))) for m in models}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for ds_idx, (ds_name, (X_base, y)) in enumerate(uci_sets.items()):
        n_samples, d_base = X_base.shape
        log(f"Dataset [{ds_idx+1}/4]: {ds_name} (n={n_samples}, d_base={d_base})")
        
        for idx_d, d_target in enumerate(target_dims):
            rng = np.random.RandomState(42 + d_target)
            X_std_base = StandardScaler().fit_transform(X_base)
            
            if d_target > d_base:
                W = rng.normal(0, 0.5, size=(d_base, d_target - d_base))
                corr_noise = X_std_base @ W + rng.normal(0, 1.0, size=(n_samples, d_target - d_base))
                X = np.hstack([X_std_base, corr_noise])
                X = X[:, rng.permutation(d_target)]
            else:
                X = X_std_base[:, rng.permutation(d_base)[:d_target]]
                
            fold_scores = {m: [] for m in models}
            
            for train_idx, test_idx in skf.split(X, y):
                X_tr, X_te = X[train_idx], X[test_idx]
                y_tr, y_te = y[train_idx], y[test_idx]
                
                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_tr)
                X_te_s = scaler.transform(X_te)
                
                # 1. SmallMLP
                mlp = SmallMLPClassifier(alpha=6.0, beta=0.7, random_state=42)
                mlp.fit(X_tr_s, y_tr)
                fold_scores["SmallMLP"].append(accuracy_score(y_te, mlp.predict(X_te_s)))
                
                # 2. TabPFN (PCA)
                if d_target > 100:
                    pca = PCA(n_components=100, random_state=42)
                    X_tr_p = pca.fit_transform(X_tr_s)
                    X_te_p = pca.transform(X_te_s)
                else:
                    X_tr_p, X_te_p = X_tr_s, X_te_s
                pfn_model.fit(X_tr_p, y_tr, overwrite_warning=True)
                pfn_pca_acc = accuracy_score(y_te, pfn_model.predict(X_te_p))
                fold_scores["TabPFN (PCA-100 >100d)"].append(pfn_pca_acc)
                
                # 3. TabPFN (Truncate)
                if d_target > 100:
                    pfn_model.fit(X_tr_s[:, :100], y_tr, overwrite_warning=True)
                    pfn_sub_acc = accuracy_score(y_te, pfn_model.predict(X_te_s[:, :100]))
                else:
                    pfn_sub_acc = pfn_pca_acc
                fold_scores["TabPFN (Truncate-100 >100d)"].append(pfn_sub_acc)
                
            for m in models:
                per_ds_results[m][ds_idx, idx_d] = np.mean(fold_scores[m])
            log(f"  -> d={d_target:3d} | SmallMLP: {per_ds_results['SmallMLP'][ds_idx, idx_d]:.4f} | PFN-PCA: {per_ds_results['TabPFN (PCA-100 >100d)'][ds_idx, idx_d]:.4f}")

    # Save results
    out_dir = Path(__file__).resolve().parent
    mean_results = {m: np.mean(per_ds_results[m], axis=0) for m in models}
    df_curve = pd.DataFrame(mean_results, index=target_dims)
    df_curve.index.name = "Dimensionality (d)"
    df_curve.to_csv(out_dir / "results_mlp.csv")
    log("Saved results to results_mlp.csv")

if __name__ == "__main__":
    main()