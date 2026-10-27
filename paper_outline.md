# Small Data Machine Learning: Fundamental Challenges and Targeted Solutions

**Status:** Draft / Outline
**Author:** Ilya Emelyanov (Beijing Institute of Technology)
**Target:** arXiv preprint

## 1. Introduction: The Small Data Paradigm

* **Definition:** What constitutes "small data" in tabular ML (e.g., $n < 1000$) and why standard models fail.
* **The Core Thesis:** Conceptualizing the mathematical bottlenecks of small tabular data and presenting targeted algorithmic solutions (SmallGBM, SmallMLP) to address specific regimes, rather than a universal "one-size-fits-all" framework.

## 2. Real-World Applications and Constraints

* **Domains:** Medical cohorts, rare chemistry/catalysis experiments, particle detectors.
* **The Constraint:** Data acquisition is physically or financially bottlenecked; algorithms must adapt to the data limits, not vice versa.

## 3. Fundamental Bottlenecks of Small Data Learning

* **High Variance & Overfitting:** Complex models memorize noise instead of extracting signal.
* **Vulnerability to Outliers:** Standard mean-based estimators fail catastrophically on single anomalies.
* **The Hyperparameter Trap:** Proper validation splits deplete training data, rendering grid/Bayesian search useless.
* **Uncertainty Quantification:** Point estimates are dangerous and insufficient for high-stakes decisions.

## 4. Mathematical Principles for Robust Solutions

* **Robust Statistics:** Replacing the mean with median estimators and inverse-distance weighting.
* **Adaptive Complexity:** Deriving network regularization and architecture directly from dataset parameters $(n, d, K)$.
* **Conformal Prediction:** Distribution-free uncertainty bounds under finite-sample exchangeability.

## 5. Standard Baselines vs. Small Data Realities

* **XGBoost / LightGBM:** Fragile mean-based leaves; depend entirely on impossible hyperparameter tuning.
* **Random Forests:** Robust, but struggle to capture smooth boundaries and extrapolate.
* **MLPs & GPs:** Standard MLPs overfit; GPs scale poorly and are overly sensitive to kernel choice.

## 6. Targeted Solutions (Case Studies)

* **6.1. SmallGBM: Robust Classification ($n < 1000$)**
  * *Concept:* Robust leaf weights (median + shrinkage) and stochastic split selection.
  * *Performance:* Outperforms XGBoost (+0.65% AUC) with zero tuning.
  * *Honest Limits:* Not suitable for regression tasks or large-scale data ($n > 1000$).
* **6.2. SmallMLP: Nonlinear Regression and Conformal Uncertainty**
  * *Concept:* Parameter-derived architecture with weighted conformal prediction.
  * *Performance:* Best mean rank (1.82) in regression; 19% narrower valid conformal intervals.
  * *Honest Limits:* Underperforms on strictly linear problems; heteroscedastic datasets can cause undercoverage.

## 7. Experimental Evaluation & Benchmarks

* **Setup:** Strict zero-tuning evaluation on 20+ synthetic and real-world small datasets.
* **Results:** Performance tables, Wilcoxon signed-rank tests for statistical significance.
* **The Scaling Curve:** The critical "AUC/MAE vs $n$" plot demonstrating exactly where standard models (XGBoost/MLP) overtake the proposed methods.

## 8. Conclusion and Future Directions

* **Summary:** Effective small data learning requires shifting from data-hungry optimization to mathematically constrained, tuning-free algorithms.
* **Negative Results:** A transparent summary of failed hypotheses (e.g., adaptive depth, PCA width) to guide future research.

## 9. Acknowledgements & References

* **Affiliation:** Beijing Institute of Technology (BIT).
