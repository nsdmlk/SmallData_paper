
# Summary of Current Paper State (Sections 1–5)

**Status:** Draft (Sections 1–5 formalized)
**Title:** Machine Learning on Small Tabular Data: Fundamental Challenges and Targeted Solutions

## 1. Introduction: The Small Data Paradigm

* **Problem:** The "big data" paradigm (XGBoost, LightGBM, MLPs) fails on small tabular datasets (\(n < 1000\)). This is a mathematical inevitability, not an engineering bug.
* **Root cause:** Standard Empirical Risk Minimization (ERM) overfits noise because the generalization gap depends heavily on sample size \(n\) and hypothesis class complexity \(\mathcal{F}\).
* **Core Thesis:** We need targeted solutions for specific failure modes (robustness, adaptive complexity, conformal uncertainty), not a universal framework.

## 2. Real-World Applications and Constraints (Mathematical Pathologies)

We mapped each domain to a specific mathematical bottleneck:

* **Clinical Medicine (Rare Diseases):** \(d \approx n\) regime. Induces ill-conditioned covariance matrices and destroys validation splits. Need tuning-free algorithms.
* **Materials Science (Catalysis):** Strong heteroscedasticity (input-dependent noise \(\sigma^2(x)\)). Standard models miscalibrate uncertainty. Need locally adaptive intervals.
* **High-Energy Physics / Astronomy:** Heavy-tailed noise and outliers. Standard mean-based estimators have a breakdown point of zero. Need robust statistics (median-based).

## 3. Fundamental Bottlenecks of Small Data Learning

We formalized four mathematical failure modes:

1. **High Variance & Curse of Dimensionality:** Sample density \(n^{1/d}\) is too low. Concentration of measure forces erratic interpolation.
2. **Outlier Vulnerability:** Second-order boosting weights \(w^* = - \frac{\sum g_i}{\sum h_i + \lambda}\) have unbounded influence functions (breakdown point = 0). One outlier ruins the leaf.
3. **The Hyperparameter Trap:** Validation splits deplete scarce training data. Multiplicity penalty \(\mathcal{O}(\sqrt{\log|\Lambda| / n_{\text{val}}})\) mathematically proves that grid search overfits the validation noise.
4. **Uncertainty Failure:** Standard models lack calibrated intervals. GPs suffer from kernel degeneracy and Type-II ML overfitting. We need distribution-free conformal prediction.

* **Synthesis:** The "Trilemma" — on small data, you cannot simultaneously have low bias, low variance, and calibrated uncertainty using standard algorithms.

## 4. Mathematical Principles for Robust Solutions

We defined three principles to directly counteract the bottlenecks:

1. **Robust Leaf Estimation (SmallGBM):** Replace mean with median (for \(|I| \le 30\)) and use a spatially regularized weighted mean for larger leaves. Added empirical Bayes shrinkage \(\gamma = \frac{\hat{\sigma}_{\text{leaf}}^2 / |I|}{\sigma_{\text{prior}}^2 + \hat{\sigma}_{\text{leaf}}^2 / |I|}\) (with MAD variance estimation) to shrink noisy leaves toward the parent.
2. **Adaptive Complexity (SmallMLP):** Derive architecture deterministically. Width \(w_1 = \max(2K, \min(\lfloor \alpha \sqrt{n K \sqrt{d}} \rfloor, 4n))\). Decay by \(\beta^{l-1}\). Weight decay = \(1/n\). Dropout clamped to \([0.1, 0.5]\). No tuning.
3. **Conformal Prediction (SmallMLP):** Use weighted quantiles of conformity scores with Gaussian kernel weights on embeddings \(h(x)\). Guarantees coverage \(\ge 1-\alpha\).
   * *Resolved Paradox:* Tuning the bandwidth \(h_{\text{cal}}\) over a 1D grid does not violate the hyperparameter trap because the search space is scalar (\(|\Lambda| \approx 10\)), not high-dimensional (\(|\Lambda| \approx 10^3-10^5\)), and coverage is guaranteed regardless of \(h_{\text{cal}}\).

## 5. Standard Baselines vs. Small Data Realities

We mapped baselines to their specific mathematical failures:

* **XGBoost / LightGBM:** Unbounded influence functions; fail without tuning.
* **Random Forest:** Piecewise constant; high bias on smooth regression; no native uncertainty.
* **Standard MLP:** Architecture trap; severe validation overfitting.
* **Gaussian Process:** (\mathcal{O}(n^3))	 scaling; kernel degeneracy at \(d \approx n\); Type-II ML overfitting.
* **TabPFN (In-Context Learning):** Tuning-free, but restricted to classification (\(d \le 100\), \(K \le 10\)); no native regression intervals; no finite-sample coverage guarantees under prior mismatch.
* **AutoML (AutoGluon):** Brute-force search contradicts the mathematical constraints of small data (triggers the hyperparameter trap).
