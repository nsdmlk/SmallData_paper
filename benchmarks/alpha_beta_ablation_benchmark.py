import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris, load_wine
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from ucimlrepo import fetch_ucirepo
from smallmlp import SmallMLPClassifier

warnings.filterwarnings("ignore")

ALPHAS = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
BETAS = [0.5, 0.6, 0.7, 0.8, 0.9]

def load_uciml(uid):
    try:
        ds = fetch_ucirepo(id=uid)
        X_df = ds.data.features.select_dtypes(include=[np.number])
        X = X_df.to_numpy(dtype=float)
        y_raw = ds.data.targets.iloc[:, 0]
        y = LabelEncoder().fit_transform(y_raw.astype(str))
        if np.isnan(X).any():
            X = SimpleImputer(strategy="median").fit_transform(X)
        mask = np.isfinite(X).all(axis=1)
        return X[mask], y[mask]
    except Exception:
        return None, None

def get_meta_datasets():
    datasets = []
    datasets.append(("iris", load_iris().data, load_iris().target))
    datasets.append(("wine", load_wine().data, load_wine().target))
    
    uci_ids = {
        "sonar": 151,
        "glass": 42,
        "yeast": 110,
        "heart_statlog": 145,
        "blood_transfusion": 176,
        "banknote": 267,
        "segment": 50,
        "vowel": 59,
        "liver_disorders": 225,
        "cmc": 30
    }
    
    for name, uid in uci_ids.items():
        X, y = load_uciml(uid)
        if X is not None:
            if len(y) > 1000:
                rng = np.random.default_rng(42)
                idx = rng.choice(len(y), 1000, replace=False)
                X, y = X[idx], y[idx]
            
            if np.bincount(y).min() >= 5:
                datasets.append((name, X, y))
            
    return datasets

def eval_single_fold(alpha, beta, X, y, tr, te, seed):
    clf = SmallMLPClassifier(
        width_mode="formula", class_weight=None,
        alpha=alpha, beta=beta,
        max_epochs=500, patience=30, random_state=seed
    )
    clf.fit(X[tr], y[tr])
    return accuracy_score(y[te], clf.predict(X[te]))

def eval_combination(alpha, beta, datasets, n_splits, seed):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    ds_accs = []
    
    for ds_name, X, y in datasets:
        fold_accs = Parallel(n_jobs=-1)(
            delayed(eval_single_fold)(alpha, beta, X, y, tr, te, seed) 
            for tr, te in skf.split(X, y)
        )
        ds_accs.append(np.mean(fold_accs))
        
    return {
        "alpha": alpha,
        "beta": beta,
        "mean_acc": float(np.mean(ds_accs))
    }

def run_grid_search(n_splits=5, seed=42):
    datasets = get_meta_datasets()
    print(f"Running Meta-Training on {len(datasets)} strictly disjoint datasets:")
    for n, X, y in datasets:
        print(f"  - {n:<18} (n={len(y):>4}, d={X.shape[1]:>2}, K={len(np.unique(y)):>2})")

    results = []
    total_runs = len(ALPHAS) * len(BETAS)
    current_run = 1
    
    start_time = time.time()

    for alpha in ALPHAS:
        for beta in BETAS:
            print(f"[{current_run}/{total_runs}] alpha={alpha:<4.1f}, beta={beta:<3.1f} ...", end=" ", flush=True)
            res = eval_combination(alpha, beta, datasets, n_splits, seed)
            results.append(res)
            
            elapsed = time.time() - start_time
            avg_time = elapsed / current_run
            eta = avg_time * (total_runs - current_run)
            
            print(f"Mean Acc: {res['mean_acc']:.4f} | ETA: {eta/60:.1f}m")
            current_run += 1

    return pd.DataFrame(results)

def plot_heatmap(df):
    pivot = df.pivot(index="alpha", columns="beta", values="mean_acc")
    
    best_idx = np.unravel_index(np.argmax(pivot.values, axis=None), pivot.shape)
    best_alpha = pivot.index[best_idx[0]]
    best_beta = pivot.columns[best_idx[1]]
    best_acc = pivot.values[best_idx]

    plt.figure(figsize=(10, 7))
    sns.heatmap(pivot, annot=True, fmt=".4f", cmap="YlGnBu", 
                cbar_kws={'label': 'Mean Accuracy on Meta-Training Set'})
    
    rect = plt.Rectangle((best_idx[1], best_idx[0]), 1, 1, fill=False, edgecolor='red', lw=3)
    plt.gca().add_patch(rect)
    
    plt.title(f"Meta-Training Grid Search (Disjoint Set)\nBest: alpha={best_alpha}, beta={best_beta} (Acc={best_acc:.4f})", fontsize=14)
    plt.xlabel("Beta (Depth Decay)", fontsize=12)
    plt.ylabel("Alpha (Width Scale)", fontsize=12)
    
    plt.tight_layout()
    plt.savefig("meta_training_heatmap.png", dpi=300)
    plt.show()
    
    print("\n" + "="*50)
    print(f"WINNER: Alpha={best_alpha}, Beta={best_beta}")
    print(f"Accuracy: {best_acc:.4f}")
    print("="*50)
    
    df.to_csv("meta_training_grid_results.csv", index=False)

if __name__ == "__main__":
    df = run_grid_search()
    plot_heatmap(df)