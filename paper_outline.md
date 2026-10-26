
# Small Data Machine Learning: Robust Boosting and Adaptive Neural Networks

**Status:** Draft / Outline
**Author:** Ilya Emelyanov (Beijing Institute of Technology)
**Target:** arXiv preprint

## 1. Small Data: Definition and Context

* **Definition:** What constitutes "small data" in tabular ML (e.g., \(n < 1000\)).
* **The Shift:** Moving from "big data" paradigm (scale-dependent) to low-sample regimes.
* **Challenges:** High variance, vulnerability to outliers, inability to use standard hyperparameter tuning (no data for validation splits).

## 2. Applications of Small Data

* **Where it matters:** Medical cohorts (rare diseases), chemistry/catalysis (expensive experiments), particle detectors, astronomy (rare events).
* **Constraint:** In these domains, getting more data is physically or financially impossible. Algorithms must adapt to the data, not vice versa.

## 3. Problems of Small Data (The "Why it fails" section)

* **Overfitting:** Complex models memorize noise.
* **Outlier Sensitivity:** A single anomaly destroys standard mean-based estimators (e.g., gradient boosting leaf weights).
* **Hyperparameter Tuning:** Standard Bayesian optimization or grid search fails due to lack of validation data.
* **Uncertainty:** Point estimates are not enough; we need calibrated intervals for decision-making.

## 4. Mathematical Principles for Small Data

* **Robust Statistics:** Replacing mean with median (for leaves \(n \le 30\)) and inverse-distance weighted means.
* **Shrinkage:** Signal-adaptive regularization toward the parent node.
* **Adaptive Complexity:** Deriving architecture from dataset parameters \((n, d, K)\) rather than searching.
* **Conformal Prediction:** Distribution-free uncertainty quantification under exchangeability.

## 5. Baseline Algorithms (Pros & Cons)

* **XGBoost / LightGBM:** Great for scale, but mean-based leaf weights are fragile; hyperparameter tuning is required (fails on small data).
* **Random Forests:** Robust, but poor at extrapolating and capturing smooth boundaries.
* **Standard MLPs:** Overfit heavily, require manual architecture search and heavy regularization.
* **Gaussian Processes:** Good uncertainty, but \(O(n^3)\) complexity and sensitive to kernel choice.

## 6. SmallGBM: Ideas, Math, Code, Limitations

* **Idea:** Gradient boosting tailored for \(n < 1000\).
* **Math:** Robust leaf weights (median + shrinkage), stochastic split selection (5 random thresholds per feature).
* **Code:** Native C core (2-3x faster), scikit-learn compatible.
* **Results:** +0.65% AUC over XGBoost, no tuning required.
* **Limitations:** Regression unsupported (classification only); degrades on \(n > 1000\).

## 7. SmallMLP: Ideas, Math, Code, Limitations

* **Idea:** Adaptive neural network + weighted conformal prediction.
* **Math:** Width/regularization derived from \(n, d, K\). Weighted conformal intervals for tighter bounds.
* **Code:** PyTorch-based, conformal calibration API.
* **Results:** 25/45 wins in regression (no tuning), 19% narrower intervals vs. split conformal.
* **Limitations:** Fails on linear problems (LogReg is better); undercovers on ~15% of highly heteroscedastic datasets.

## 8. Experiments and Benchmarks

* [benchmarks and experiments]

## 9. Application of Tools

* **SmallGBM:** Best for small tabular classification (e.g., medical diagnostics, sensor arrays).
* **SmallMLP:** Best for nonlinear regression with uncertainty requirements (e.g., chemistry, physics experiments).

## 10. Conclusion

* Summarize: Tuning-free, robust methods are necessary for small data.
* Negative results are valuable (we tried adaptive depth, PCA width, etc., and they failed).
* Future work: Extending to semi-supervised learning, multi-modal small data.

## 11. Acknowledgements

* Affiliation: Beijing Institute of Technology (BIT).
* Special thanks: Mohammad Hamza (Postdoc, BIT) for valuable discussions and help with the publication process.

## 12. References

* [To be filled: XGBoost, LightGBM, Conformal Prediction, etc.]
