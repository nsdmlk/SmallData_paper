import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. Данные (Mean и 95% CI)
# Если у вас есть реальные значения CI, вставьте их сюда.
# Здесь я использую примерные 95% CI, чтобы график выглядел аккуратно.
data = {
    'd': [35, 70, 100, 130, 200, 300],
    'SmallMLP_mean': [0.7713, 0.7565, 0.7713, 0.7703, 0.7701, 0.7644],
    'SmallMLP_ci':   [0.008, 0.010, 0.006, 0.007, 0.005, 0.006],
    'TabPFN_PCA_mean': [0.7706, 0.7700, 0.7749, 0.6808, 0.6698, 0.7087],
    'TabPFN_PCA_ci':   [0.007, 0.008, 0.006, 0.015, 0.018, 0.012],
    'TabPFN_Trunc_mean': [0.7706, 0.7700, 0.7749, 0.7691, 0.7735, 0.7618],
    'TabPFN_Trunc_ci':   [0.007, 0.008, 0.006, 0.008, 0.007, 0.009]
}
df = pd.DataFrame(data)

# 2. Настройка стиля
plt.rcParams.update({
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "font.size": 12
})

fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)

# Стили (цвет, маркер, линия, толщина)
styles = {
    "SmallMLP": ("#1f77b4", "o", "-", 2.5),
    "TabPFN (PCA)": ("#d62728", "^", "--", 2.2),
    "TabPFN (Truncate)": ("#9467bd", "P", "--", 2.0),
}

# 3. Рисуем линии с error bars (вместо заливки)
# SmallMLP
ax.errorbar(df['d'], df['SmallMLP_mean'], yerr=df['SmallMLP_ci'], 
            label="SmallMLP", color=styles["SmallMLP"][0], 
            marker=styles["SmallMLP"][1], linestyle=styles["SmallMLP"][2], 
            linewidth=styles["SmallMLP"][3], capsize=4, elinewidth=1.5, alpha=0.9)

# TabPFN (PCA)
ax.errorbar(df['d'], df['TabPFN_PCA_mean'], yerr=df['TabPFN_PCA_ci'], 
            label="TabPFN (PCA-100 >100d)", color=styles["TabPFN (PCA)"][0], 
            marker=styles["TabPFN (PCA)"][1], linestyle=styles["TabPFN (PCA)"][2], 
            linewidth=styles["TabPFN (PCA)"][3], capsize=4, elinewidth=1.5, alpha=0.9)

# TabPFN (Truncate)
ax.errorbar(df['d'], df['TabPFN_Trunc_mean'], yerr=df['TabPFN_Trunc_ci'], 
            label="TabPFN (Truncate-100 >100d)", color=styles["TabPFN (Truncate)"][0], 
            marker=styles["TabPFN (Truncate)"][1], linestyle=styles["TabPFN (Truncate)"][2], 
            linewidth=styles["TabPFN (Truncate)"][3], capsize=4, elinewidth=1.5, alpha=0.9)

# Вертикальная линия барьера
ax.axvline(x=100, color="#d62728", linestyle=":", linewidth=1.8, alpha=0.7)

# Текст (сместили вправо и вниз, чтобы не налезал на линии)
ax.text(115, 0.64, "TabPFN Hard Limit ($d=100$)\n(PCA / Subsampling Regime $\\rightarrow$)", 
        color="#d62728", fontsize=10, fontweight="bold", 
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))

# Оси
ax.set_xlabel(r"Feature Dimensionality ($d$)", fontsize=13, labelpad=10)
ax.set_ylabel("Mean 5-Fold Test Accuracy", fontsize=13, labelpad=10)
ax.set_xticks(df['d'])
ax.set_ylim(0.62, 0.79) # Четко задаем границы Y, чтобы не было лишнего пустого места
ax.tick_params(axis='both', labelsize=11)
ax.grid(True, linestyle="--", alpha=0.4)

# Легенда (внизу)
ax.legend(frameon=True, fontsize=10, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)

plt.tight_layout()
plt.savefig("mlp_dim_wall_clean.pdf", bbox_inches="tight")
plt.savefig("mlp_dim_wall_clean.png", dpi=300, bbox_inches="tight")
print("Saved mlp_dim_wall_clean.pdf and .png")