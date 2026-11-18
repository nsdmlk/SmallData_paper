import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import t

# 1. Находим последнюю папку с результатами
results_dir = sorted(Path("benchmarks/results/group13/20261116-103909-422844").glob("*/"))[-1]
print(f"Loading data from: {results_dir}")

runs = pd.read_csv(results_dir / "runs.csv")
diagnostics = pd.read_csv(results_dir / "summary.csv")

tasks = ["Gaussian mixture", "Nonlinear interactions", "Null signal"]
methods = ["XGBoost default", "XGBoost tuned", "SmallGBM"]

# 2. Перерисовываем Рисунок 1 (Test AUC vs Methods)
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
fig.suptitle('Default vs. 5-Fold Grid-Tuned XGBoost vs. Zero-Tuning SmallGBM at \(n = 100\)', fontsize=14)

for i, task in enumerate(tasks):
    ax = axes[i]
    task_data = runs[runs['task'] == task]
    
    # Считаем среднее и 95% CI
    means = []
    errors = []
    for m in methods:
        m_data = task_data[task_data['method'] == m]['auc']
        means.append(m_data.mean())
        # 95% CI по t-распределению
        ci = t.ppf(0.975, len(m_data)-1) * m_data.std() / np.sqrt(len(m_data))
        errors.append(ci)
    
    # Рисуем точки с усами
    x_pos = np.arange(len(methods))
    ax.errorbar(x_pos, means, yerr=errors, fmt='o', capsize=5, color='black', ecolor='black', label='Mean ± 95% CI')
    
    # Добавляем серые линии для каждого сида (spaghetti)
    for seed in task_data['seed'].unique():
        seed_data = task_data[task_data['seed'] == seed]
        seed_means = [seed_data[seed_data['method'] == m]['auc'].values[0] for m in methods]
        ax.plot(x_pos, seed_means, color='gray', alpha=0.2, linewidth=0.5)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, rotation=15, ha='right')
    ax.set_title(task)
    ax.set_ylabel('Test ROC-AUC' if i == 0 else '')
    ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("fig1_updated.pdf", bbox_inches='tight')
plt.savefig("fig1_updated.png", dpi=300, bbox_inches='tight')
plt.show()

# 3. Перерисовываем Рисунок 2 (Diagnostic intervals)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Левая панель: Разница на тесте после рефита
ax = axes[0]
for i, task in enumerate(tasks):
    task_diag = diagnostics[diagnostics['task'] == task]
    diffs = -task_diag['tuning_penalty'] # default - tuned
    mean_diff = diffs.mean()
    ci = t.ppf(0.975, len(diffs)-1) * diffs.std() / np.sqrt(len(diffs))
    
    ax.errorbar(i, mean_diff, yerr=ci, fmt='o', capsize=5, color='black')
    # Добавляем точки отдельных сидов
    ax.scatter(np.full(len(diffs), i) + np.random.normal(0, 0.05, len(diffs)), diffs, color='purple', alpha=0.4, s=15)

ax.axhline(0, color='black', linestyle='--')
ax.set_xticks(range(3))
ax.set_xticklabels(tasks)
ax.set_ylabel('Test AUC Difference (Default - Tuned)')
ax.set_title('Refitted Test Difference (Default - Tuned)')
ax.grid(True, linestyle='--', alpha=0.5)

# Правая панель: CV Optimism Bias
ax = axes[1]
for i, task in enumerate(tasks):
    task_diag = diagnostics[diagnostics['task'] == task]
    optimisms = task_diag['differential_cv_optimism']
    mean_opt = optimisms.mean()
    ci = t.ppf(0.975, len(optimisms)-1) * optimisms.std() / np.sqrt(len(optimisms))
    
    ax.errorbar(i, mean_opt, yerr=ci, fmt='o', capsize=5, color='black')
    ax.scatter(np.full(len(optimisms), i) + np.random.normal(0, 0.05, len(optimisms)), optimisms, color='purple', alpha=0.4, s=15)

ax.axhline(0, color='black', linestyle='--')
ax.set_xticks(range(3))
ax.set_xticklabels(tasks)
ax.set_ylabel('CV Optimism Bias (CV Gain - True Gain)')
ax.set_title('5-Fold CV Optimism Bias (CV Gain - True Gain)')
ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("fig2_updated.pdf", bbox_inches='tight')
plt.savefig("fig2_updated.png", dpi=300, bbox_inches='tight')
plt.show()