#!/usr/bin/env python3
"""
Benchmark Part 1: Tree Models (XGBoost, RF, SmallGBM)
Evaluates trees across d in {35, 70, 100, 130, 200, 300} on 4 UCI datasets.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from smallgbm import SmallGBMClassifier

warnings.filterwarnings("ignore")

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def load_uci_datasets():
    log("Loading UCI datasets...")
    datasets = {}
    
    # Sonar
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data"
    df = pd.read_csv(url, header=None)
    datasets["Sonar"] = (df.iloc[:, :-1].values.astype(float), LabelEncoder().fit_transform(df.iloc[:, -1].astype(str)))
    
    # SPECTF Heart
    url_tr = "https://archive.ics.uci.edu/ml/machine-learning-databases/spect/SPECTF.train"
    url_te = "https://archive.ics.uci.edu/ml/machine-learning-databases/spect/SPECTF.test"
    df = pd.concat([pd.read_csv(url_tr, header=None), pd.read_csv(url_te, header=None)], ignore_index=True)
    datasets["SPECTF_Heart"] = (df.iloc[:, 1:].values.astype(float), LabelEncoder().fit_transform(df.iloc[:, 0].astype(str)))
    
    # Heart Cleveland
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    df = pd.read_csv(url, header=None, na_values="?")
    X = SimpleImputer(strategy="median").fit_transform(df.iloc[:, :-1].values.astype(float))
    y = (df.iloc[:, -1].values.astype(float) > 0).astype(int)
    datasets["Heart_Cleveland"] = (X, y)
    
    # Glass
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data"
    df = pd.read_csv(url, header=None)
    datasets["Glass"] = (df.iloc[:, 1:-1].values.astype(float), LabelEncoder().fit_transform(df.iloc[:, -1].astype(str)))
    
    return datasets

def main():
    uci_sets = load_uci_datasets()
    target_dims = [35, 70, 100, 130, 200, 300]
    models = ["SmallGBM", "XGBoost (Default)", "Random Forest"]
    
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
                
                # 1. XGBoost
                model_xgb = xgb.XGBClassifier(random_state=42, eval_metric="logloss", n_jobs=1)
                model_xgb.fit(X_tr, y_tr)
                fold_scores["XGBoost (Default)"].append(accuracy_score(y_te, model_xgb.predict(X_te)))
                
                # 2. Random Forest
                rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
                rf.fit(X_tr, y_tr)
                fold_scores["Random Forest"].append(accuracy_score(y_te, rf.predict(X_te)))
                
                # 3. SmallGBM
                sgbm = SmallGBMClassifier(random_state=42)
                sgbm.fit(X_tr, y_tr)
                fold_scores["SmallGBM"].append(accuracy_score(y_te, sgbm.predict(X_te)))
                
            for m in models:
                per_ds_results[m][ds_idx, idx_d] = np.mean(fold_scores[m])
            log(f"  -> d={d_target:3d} | SmallGBM: {per_ds_results['SmallGBM'][ds_idx, idx_d]:.4f} | XGB: {per_ds_results['XGBoost (Default)'][ds_idx, idx_d]:.4f} | RF: {per_ds_results['Random Forest'][ds_idx, idx_d]:.4f}")

    # Save results
    out_dir = Path(__file__).resolve().parent
    mean_results = {m: np.mean(per_ds_results[m], axis=0) for m in models}
    df_curve = pd.DataFrame(mean_results, index=target_dims)
    df_curve.index.name = "Dimensionality (d)"
    df_curve.to_csv(out_dir / "results_trees.csv")
    log("Saved results to results_trees.csv")

if __name__ == "__main__":
    main()