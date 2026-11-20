![Small Data Machine Learning](Header.png)

> Machine learning on small tabular datasets ($n < 1000$) is governed by the **Trilemma of Small Data Learning**: the inability to simultaneously achieve low variance, low bias, and calibrated uncertainty without depleting scarce training signals via validation splits.

## Abstract

This work formalizes the mathematical bottlenecks of low-sample tabular learning and evaluates two targeted, zero-tuning algorithmic baselines as case studies:

1. **SmallGBM** (Robust Classification): Combines size-adaptive median leaf aggregation, MAD empirical Bayes shrinkage, and stochastic split selection.
2. **SmallMLP** (Nonlinear Regression and UQ): A two-layer bottleneck network governed by deterministic capacity scaling rules $(n, d, K)$ and locally weighted split conformal prediction.

## Key Contributions

*   **The Trilemma of Small Data.** Formalizing why standard algorithms fail without validation tuning (high variance, unbounded influence functions, signal depletion).
*   **Targeted Solutions.** Zero-tuning algorithmic designs that mathematically bound variance and outlier influence.
*   **Strict Performance Boundaries.** Empirical scaling curves demonstrating exactly where standard models (XGBoost, AutoML, TabPFN) overtake our methods, identifying the $n \approx 800 - 1000$ phase transition.
*   **True Negative Results.** A transparent documentation of structural limitations (e.g., the asymptotic $\pi/2$ Gaussian efficiency penalty of median leaves, and $L_2$ sensitivity to extreme Cauchy outliers).

## Empirical Findings

*   **The Validation Trap.** At $n=100$, standard 5-fold cross-validation hallucinates $+0.063$ ROC-AUC on pure null noise, while failing to match zero-tuning SmallGBM.
*   **Robustness to Corruption.** Under $20\%$ label corruption, SmallGBM reduces median AUC degradation by $3.7\times$ compared to XGBoost.
*   **The TabPFN Dimensionality Wall.** While TabPFN is dominant for $d \le 100$, it suffers accuracy collapse when forced to compress higher dimensions (PCA/Truncation). SmallMLP scales gracefully to $d=300$ natively.
*   **Heteroscedasticity and Coverage.** Under $6\times$ heteroscedasticity, conformal SmallMLP maintains valid $\ge 90\%$ finite-sample marginal coverage where parametric Gaussian Processes break down.

## Citation

If you use this paper or the associated code in your research, please cite it as follows:

```bibtex
@misc{emelyanov2026smalldata,
  title={Small Data Machine Learning: The Trilemma of Low-Sample Tabular Regimes},
  author={Emelyanov, Ilya and Dobrydin, Dmitrii and Hamza, Muhammad},
  year={2026}
}
```

## Repositories

The core implementations discussed in this paper are open-source, alongside the baseline frameworks used for rigorous zero-tuning benchmarking.

**Proposed Methods:**
*   [SmallGBM](https://github.com/nsdmlk/SmallGBM) — Robust Gradient Boosting for $n < 1000$.
*   [SmallMLP](https://github.com/nsdmlk/SmallMLP) — Adaptive Neural Networks with Conformal Intervals.

**Baselines & Evaluated Frameworks:**
*   [XGBoost](https://github.com/dmlc/xgboost) — Scalable Gradient Boosting.
*   [LightGBM](https://github.com/microsoft/LightGBM) — Fast, distributed gradient boosting.
*   [scikit-learn](https://github.com/scikit-learn/scikit-learn) — Machine Learning in Python.
*   [TabPFN](https://github.com/automl/TabPFN) — Prior-Data Fitted Networks.
*   [AutoGluon](https://github.com/autogluon/autogluon) — Automated machine learning.
*   [MAPIE](https://github.com/scikit-learn-contrib/MAPIE) — Conformal prediction framework.

## Acknowledgements

This research was conducted independently during undergraduate studies at Beijing Institute of Technology (BIT).

## License

*   **Manuscript and Figures:** This work is licensed under a [Creative Commons Attribution 4.0 International License (CC-BY-4.0)](https://creativecommons.org/licenses/by/4.0/).
*   **Source Code (SmallGBM, SmallMLP):** The referenced code repositories are licensed under the [MIT License](https://opensource.org/licenses/MIT).
