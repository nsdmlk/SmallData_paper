# Experiment & Benchmark Plan (Pre-print)

## 1. What graphs/tables we need

* **Table 1:** SmallGBM vs XGBoost, LightGBM, RF, LogReg (Mean AUC, Training time, Inference time).
* **Table 2:** SmallMLP vs MLP-128, RF, KNN, GP (Mean MAE, Mean Rank, Wins).
* **Table 3:** Conformal Intervals (Coverage vs Width) for SmallMLP vs Split Conformal.
* **Figure 1:** Bar chart — Time vs Accuracy tradeoff (SmallGBM vs XGBoost).
* **Figure 2:** Line plot — AUC/MAE as \(n\) grows (from 50 to 10,000) to show the exact breaking point where standard models overtake your methods.
* **Figure 3:** Scatter plot — Set size vs Dataset difficulty (multiclass) for conformal prediction.

## 2. The "n vs Performance" curve (Crucial)

Reviewers will ask: *"Where exactly does your model stop working?"*

* **Action:** Run a synthetic experiment where \(n\) varies (\(50, 100, 200, 500, 1000, 2000, 5000, 10000\)).
* **What to show:** SmallGBM is best at \(n<500\), XGBoost takes over at \(n>1000\). This honest limitation proves academic maturity.

## 3. Statistical Significance

* We need **Wilcoxon signed-rank tests** for all baselines (you already have this for SmallMLP vs others).
* **Action:** Generate a table with p-values to show that improvements are not just random noise.

## 4. Ablation Studies (For the paper)

* **SmallGBM:** What happens if we remove Median leaf weights? (Use standard mean instead).
* **SmallGBM:** What happens if we use 100 random thresholds instead of 5?
* **SmallMLP:** What happens if we fix width to 128 instead of using the formula?

## 5. To-Do before writing

- [ ] Generate Figure 2 (the \(n\)-scaling plot) — this is the most important visual.
- [ ] Consolidate all results into a single `.csv` file.
- [ ] Draw the architecture diagram for SmallMLP (using draw.io or matplotlib).
