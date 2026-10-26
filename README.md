# Learning from Small Tabular Data: Robust Gradient Boosting and Adaptive Neural Networks without Hyperparameter Tuning

**Status:** Work in Progress (Preprint preparation)
**Target:** arXiv
**Author:** Ilya Emelyanov (Beijing Institute of Technology)

## 📌 Overview

This repository contains the manuscript, LaTeX source, and supplementary materials for the paper **"Learning from Small Tabular Data: Robust Gradient Boosting and Adaptive Neural Networks without Hyperparameter Tuning."**

The paper addresses the critical challenges of machine learning on small tabular datasets (\(n < 1000\)), where standard algorithms (like XGBoost, LightGBM, and standard MLPs) often fail due to overfitting, high variance, and the impossibility of proper hyperparameter tuning. We propose a unified framework comprising two tuning-free methods: **SmallGBM** (robust gradient boosting) and **SmallMLP** (adaptive neural networks with conformal prediction).

## 🔬 Scope of the Paper

1. **Problem Statement:** Mathematical and practical bottlenecks of small-sample tabular learning (outliers, variance, lack of validation data).
2. **SmallGBM:** A gradient boosting library with robust leaf weight estimation (median + shrinkage) and stochastic split selection.
3. **SmallMLP:** An MLP framework with architecture derived from dataset parameters \((n, d, K)\) and weighted conformal intervals for calibrated uncertainty.
4. **Honest Limitations:** Extensive negative results (what did not work) and clear boundaries of where these methods should not be used.
5. **Benchmarks:** Comprehensive evaluation on 20+ synthetic and real-world small datasets.

## 🔗 Related Repositories (Code)

The core implementations of the methods discussed in this paper are open-source and available here:

* 🚀 **[SmallGBM](https://github.com/nsdmlk/SmallGBM)** — Gradient Boosting for small tabular data.
* 🧠 **[SmallMLP](https://github.com/nsdmlk/SmallMLP)** — Adaptive Neural Networks with Conformal Intervals for small data.

## 📂 Repository Structure

* `paper_outline.md` — Detailed structure, ideas, and section-by-section breakdown.
* `paper.tex` — Main LaTeX manuscript (in progress).
* `references.bib` — Bibliography database.
* `figures/` — Plots, diagrams, and benchmark visualizations.

## 🤝 Acknowledgements

This research is conducted independently during undergraduate studies at **Beijing Institute of Technology (BIT)**.
Special thanks to **Mohammad Hamza** (Postdoc, BIT) for valuable discussions and guidance on the publication process.

## 📄 License

The text and figures in this repository are licensed under MIT. The referenced code (SmallGBM, SmallMLP) is licensed under MIT.
