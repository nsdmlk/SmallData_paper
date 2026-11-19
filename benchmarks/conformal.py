#!/usr/bin/env python3
"""
Section 4.2 Benchmark: Distribution-Free Uncertainty Quantification (Equations 8 & 9).
Tests GP Type-II ML breakdown at d ~ n vs. Split Conformal SmallMLP vs.
Locally Weighted Conformal SmallMLP on penultimate embeddings h(x).
"""

import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler
from smallmlp import SmallMLPRegressor

# ---------------- Configuration ----------------
N_TOTAL = 150
CAL_FRAC = 0.25          # Single calibration split: n_fit=112, n_cal=38
N_TEST = 2000
ALPHA = 0.10             # Target marginal coverage = 0.90
DIMS = (10, 100)         # Low-d (d=10) and High-d (d=100, d ~ n)
SEEDS = tuple(range(101, 121))  # 20 independent seeds
MAX_EPOCHS = 500
PATIENCE = 30

OUT_DIR = Path("results/conformal_uq") / datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
OUT_DIR.mkdir(parents=True, exist_ok=True)
ROOT_DIR = Path(__file__).resolve().parent.parent


# ---------------- Data Generator ----------------
def make_heteroscedastic_data(d: int, seed: int):
    streams = np.random.SeedSequence([20261119, d, seed]).spawn(2)

    def draw(n, stream):
        rng = np.random.default_rng(stream)
        X = rng.normal(size=(n, d))
        f_x = 1.5 * X[:, 0] - 1.0 * X[:, 1] + 0.8 * X[:, 2] * X[:, 3] + np.sin(X[:, 4])
        # Sharp 6x heteroscedastic noise regime based on X[:, 0]
        sigma_x = np.where(X[:, 0] <= 0.0, 0.25, 1.50)
        y = f_x + sigma_x * rng.normal(size=n)
        return np.ascontiguousarray(X, dtype=np.float64), y, sigma_x

    X, y, sig = draw(N_TOTAL, streams[0])
    Xt, yt, sig_t = draw(N_TEST, streams[1])
    return X, y, sig, Xt, yt, sig_t


# ---------------- Embedding & Conformal Helpers ----------------
def extract_penultimate_embedding(model: SmallMLPRegressor, X: np.ndarray) -> np.ndarray:
    """Extracts penultimate layer embedding h(x) from SmallMLPRegressor."""
    if hasattr(model, "_scaler") and model._scaler is not None:
        X_in = model._scaler.transform(X)
    else:
        X_in = X
    model._model.eval()
    with torch.no_grad():
        t = torch.as_tensor(X_in, dtype=torch.float32)
        h = model._model(t).cpu().numpy()
    return h


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Computes the q-quantile of `values` with non-negative `weights`."""
    sorter = np.argsort(values)
    v_sorted = values[sorter]
    w_sorted = weights[sorter]
    cdf = np.cumsum(w_sorted) / np.sum(w_sorted)
    idx = np.searchsorted(cdf, q, side="left")
    return float(v_sorted[min(idx, len(v_sorted) - 1)])


def locally_weighted_conformal_radii(
    h_cal: np.ndarray, r_cal: np.ndarray, h_test: np.ndarray, alpha: float
) -> np.ndarray:
    """
    Locally weighted split conformal prediction (Eq. 9):
    h_bw = median(||h(x_i) - h(x_j)||) over calibration set.
    """
    # Pairwise Euclidean distances on calibration set for deterministic median heuristic
    diff_cal = h_cal[:, None, :] - h_cal[None, :, :]
    dist_cal = np.linalg.norm(diff_cal, axis=-1)
    triu = dist_cal[np.triu_indices(len(h_cal), k=1)]
    h_bw = float(np.median(triu))
    if h_bw < 1e-8:
        h_bw = 1.0

    # Distances from test points to calibration points
    diff_test = h_test[:, None, :] - h_cal[None, :, :]
    dist_test = np.linalg.norm(diff_test, axis=-1)
    W = np.exp(-0.5 * (dist_test / h_bw) ** 2)

    # Include finite-sample weight 1.0 for the test point (Tibshirani et al. weighted conformal)
    radii = np.empty(len(h_test), dtype=np.float64)
    r_aug = np.append(r_cal, np.inf)
    for i in range(len(h_test)):
        w_aug = np.append(W[i], 1.0)
        radii[i] = weighted_quantile(r_aug, w_aug, 1.0 - alpha)
    # Cap any inf fallback (if all cal weights underflow) at max calibration residual
    radii = np.where(np.isfinite(radii), radii, float(np.max(r_cal)))
    return radii


# ---------------- Single Trial ----------------
def run_trial(d: int, seed: int):
    X, y, _, Xt, yt, sig_t = make_heteroscedastic_data(d, seed)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N_TOTAL)
    n_cal = int(round(CAL_FRAC * N_TOTAL))
    cal_idx, fit_idx = perm[:n_cal], perm[n_cal:]
    X_fit, y_fit = X[fit_idx], y[fit_idx]
    X_cal, y_cal = X[cal_idx], y[cal_idx]

    low_mask = sig_t < 0.5
    high_mask = ~low_mask
    results = []

    def record(method: str, pred: np.ndarray, lo: np.ndarray, hi: np.ndarray):
        covered = (yt >= lo) & (yt <= hi)
        widths = hi - lo
        corr = float(np.corrcoef(widths, sig_t)[0, 1]) if np.std(widths) > 1e-12 else 0.0
        results.append({
            "d": d,
            "seed": seed,
            "method": method,
            "marginal_coverage": float(np.mean(covered)),
            "low_noise_coverage": float(np.mean(covered[low_mask])),
            "high_noise_coverage": float(np.mean(covered[high_mask])),
            "mean_width": float(np.mean(widths)),
            "low_noise_width": float(np.mean(widths[low_mask])),
            "high_noise_width": float(np.mean(widths[high_mask])),
            "width_sigma_corr": corr,
        })

    # 1. Gaussian Process (RBF + WhiteKernel, Type-II Marginal Likelihood) on full N_TOTAL
    scaler = StandardScaler()
    X_gp = scaler.fit_transform(X)
    Xt_gp = scaler.transform(Xt)
    kernel = ConstantKernel(1.0) * RBF(length_scale=np.ones(d)) + WhiteKernel(noise_level=1.0)
    gp = GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=2, random_state=seed, normalize_y=True
    )
    gp.fit(X_gp, y)
    mu_gp, std_gp = gp.predict(Xt_gp, return_std=True)
    z = norm.ppf(1.0 - ALPHA / 2.0)
    record("GP (Type-II ML)", mu_gp, mu_gp - z * std_gp, mu_gp + z * std_gp)

    # 2. SmallMLPRegressor fitted on fit_idx, calibrated on cal_idx
    mlp = SmallMLPRegressor(random_state=seed, max_epochs=MAX_EPOCHS, patience=PATIENCE)
    mlp.fit(X_fit, y_fit)
    pred_cal = mlp.predict(X_cal)
    pred_test = mlp.predict(Xt)
    r_cal = np.abs(y_cal - pred_cal)

    # 2a. Unweighted Split Conformal (Eq. 8)
    q_level = np.ceil((len(r_cal) + 1) * (1.0 - ALPHA)) / len(r_cal)
    q_unweighted = float(np.quantile(r_cal, min(q_level, 1.0), method="higher"))
    record(
        "SmallMLP (Split Conformal)",
        pred_test,
        pred_test - q_unweighted,
        pred_test + q_unweighted,
    )

    # 2b. Locally Weighted Conformal on h(x) (Eq. 9)
    h_cal = extract_penultimate_embedding(mlp, X_cal)
    h_test = extract_penultimate_embedding(mlp, Xt)
    radii_lw = locally_weighted_conformal_radii(h_cal, r_cal, h_test, ALPHA)
    record(
        "SmallMLP (Locally Weighted)",
        pred_test,
        pred_test - radii_lw,
        pred_test + radii_lw,
    )

    return results


# ---------------- Plotting ----------------
def plot_results(rows):
    import pandas as pd

    df = pd.DataFrame(rows)
    methods = [
        "GP (Type-II ML)",
        "SmallMLP (Split Conformal)",
        "SmallMLP (Locally Weighted)",
    ]
    colors = {
        "GP (Type-II ML)": "#b8472a",
        "SmallMLP (Split Conformal)": "#6758a6",
        "SmallMLP (Locally Weighted)": "#00796b",
    }

    plt.rcParams.update({
        "font.size": 11.5,
        "axes.labelsize": 12.5,
        "axes.titlesize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11.5,
    })

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1), dpi=300)
    x = np.arange(len(DIMS))
    bar_w = 0.24

    # Panel 1: Marginal Coverage
    for i, m in enumerate(methods):
        sub = df[df["method"] == m]
        means = [sub[sub["d"] == d]["marginal_coverage"].mean() for d in DIMS]
        ses = [sub[sub["d"] == d]["marginal_coverage"].std(ddof=1) / np.sqrt(len(SEEDS)) for d in DIMS]
        axes[0].bar(x + (i - 1) * bar_w, means, width=bar_w, yerr=1.96 * np.array(ses),
                    capsize=4, color=colors[m], label=m, alpha=0.9)
    axes[0].axhline(1.0 - ALPHA, color="#111111", linestyle="--", linewidth=1.6, label=r"Target ($1-\alpha=0.90$)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"d={d}\n(n={N_TOTAL})" for d in DIMS])
    axes[0].set_ylim(0.50, 1.02)
    axes[0].set_ylabel("Marginal coverage")
    axes[0].set_title("Marginal coverage (Eq. 8)", fontweight="bold", pad=8)
    axes[0].grid(axis="y", linestyle="--", alpha=0.3)

    # Panel 2: High-Noise Conditional Coverage
    for i, m in enumerate(methods):
        sub = df[df["method"] == m]
        means = [sub[sub["d"] == d]["high_noise_coverage"].mean() for d in DIMS]
        ses = [sub[sub["d"] == d]["high_noise_coverage"].std(ddof=1) / np.sqrt(len(SEEDS)) for d in DIMS]
        axes[1].bar(x + (i - 1) * bar_w, means, width=bar_w, yerr=1.96 * np.array(ses),
                    capsize=4, color=colors[m], alpha=0.9)
    axes[1].axhline(1.0 - ALPHA, color="#111111", linestyle="--", linewidth=1.6)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"d={d}\n(n={N_TOTAL})" for d in DIMS])
    axes[1].set_ylim(0.50, 1.02)
    axes[1].set_ylabel("High-noise conditional coverage")
    axes[1].set_title("High-noise tail coverage", fontweight="bold", pad=8)
    axes[1].grid(axis="y", linestyle="--", alpha=0.3)

    # Panel 3: Low-Noise vs High-Noise Width Ratio (Adaptivity)
    for i, m in enumerate(methods):
        sub = df[df["method"] == m]
        ratios = []
        ses = []
        for d in DIMS:
            sd = sub[sub["d"] == d]
            r = sd["high_noise_width"] / sd["low_noise_width"]
            ratios.append(r.mean())
            ses.append(r.std(ddof=1) / np.sqrt(len(SEEDS)))
        axes[2].bar(x + (i - 1) * bar_w, ratios, width=bar_w, yerr=1.96 * np.array(ses),
                    capsize=4, color=colors[m], alpha=0.9)
    axes[2].axhline(1.0, color="#555555", linestyle=":", linewidth=1.6, label="Constant width (1.0x)")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([f"d={d}\n(n={N_TOTAL})" for d in DIMS])
    axes[2].set_ylabel("Width ratio (High-noise / Low-noise)")
    axes[2].set_title("Heteroscedastic adaptivity (Eq. 9)", fontweight="bold", pad=8)
    axes[2].grid(axis="y", linestyle="--", alpha=0.3)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.07))
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    for p in [OUT_DIR / "conformal_uq.pdf", ROOT_DIR / "conformal_uq.pdf"]:
        fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved tightly cropped figure to {OUT_DIR / 'conformal_uq.pdf'}")


def main():
    t0 = time.perf_counter()
    all_rows = []
    for d in DIMS:
        for seed in SEEDS:
            all_rows.extend(run_trial(d, seed))
        print(f"Finished d={d} ({len(SEEDS)} seeds)")

    # Save runs.csv
    with (OUT_DIR / "runs.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    # Print and save summary
    import pandas as pd
    df = pd.DataFrame(all_rows)
    summary = df.groupby(["d", "method"]).agg(
        marg_cov=("marginal_coverage", "mean"),
        low_cov=("low_noise_coverage", "mean"),
        high_cov=("high_noise_coverage", "mean"),
        width=("mean_width", "mean"),
        low_w=("low_noise_width", "mean"),
        high_w=("high_noise_width", "mean"),
        corr=("width_sigma_corr", "mean"),
    ).reset_index()
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    print("\n=== Summary ===")
    print(summary.to_string(index=False))

    protocol = {
        "n_total": N_TOTAL,
        "cal_frac": CAL_FRAC,
        "n_test": N_TEST,
        "alpha": ALPHA,
        "dims": DIMS,
        "seeds": SEEDS,
        "wall_seconds": round(time.perf_counter() - t0, 2),
    }
    (OUT_DIR / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    plot_results(all_rows)


if __name__ == "__main__":
    main()