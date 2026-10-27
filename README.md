# Small Data Machine Learning: Fundamental Challenges and Targeted Solutions

- **Status:** Work in Progress (Preprint preparation)
- **Target:** arXiv
- **Author:** Ilya Emelyanov (Beijing Institute of Technology)

## 📌 Overview

This repository contains the manuscript and supplementary materials for the paper evaluating the critical bottlenecks of machine learning on small tabular datasets ($n < 1000$). Instead of proposing a universal framework, we conceptualize the mathematical challenges (high variance, outlier vulnerability, hyperparameter tuning failure) and present two targeted solutions as case studies: **SmallGBM** (for robust classification) and **SmallMLP** (for nonlinear regression with uncertainty).

## 🔬 Scope of the Paper

1. **Fundamental Bottlenecks:** Analysis of why standard models (XGBoost, standard MLPs) fail on small sample sizes.
2. **Targeted Solutions:**
   - **SmallGBM:** Robust gradient boosting via median leaf weights and stochastic splits.
   - **SmallMLP:** Adaptive neural networks with parameter-derived architecture and conformal intervals.
3. **Honest Limitations:** Clear boundaries of applicability (e.g., $n$-scaling curves showing where standard models take over) and an extensive summary of negative results to guide future research.
4. **Benchmarks:** Zero-tuning evaluation on 20+ synthetic and real-world small datasets.

## 🔗 Related Repositories (Code)

The core implementations discussed in this paper are open-source and available here:

* 🚀 **[SmallGBM](https://github.com/nsdmlk/SmallGBM)** — Robust Gradient Boosting for $n < 1000$.
* 🧠 **[SmallMLP](https://github.com/nsdmlk/SmallMLP)** — Adaptive Neural Networks with Conformal Intervals.

## 📂 Repository Structure

* `paper_outline.md` — Detailed structure, ideas, and section-by-section breakdown.
* `paper.tex` — Main LaTeX manuscript.
* `references.bib` — Bibliography database.
* `figures/` — Benchmark visualizations, including the critical $n$-scaling plots.

## 🤝 Acknowledgements

This research is conducted independently during undergraduate studies at **Beijing Institute of Technology (BIT)**.
Special thanks to **Mohammad Hamza** (Postdoc, BIT) for valuable discussions and guidance on the publication process.

## 📄 License

The text and figures in this repository are licensed under MIT. The referenced code (SmallGBM, SmallMLP) is licensed under MIT.
