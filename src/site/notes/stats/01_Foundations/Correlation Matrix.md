---
{"dg-publish":true,"permalink":"/stats/01-foundations/correlation-matrix/","tags":["EDA","Visualization","Statistics"]}
---


## Definition

> [!abstract] Overview
> A **Correlation Matrix** is a table showing correlation coefficients between variables. Each cell in the table shows the correlation between two variables. It is the first step in detecting relationships and **Multicollinearity**.

- **Range:** -1 to +1.
- **Identity:** The diagonal is always 1 (Correlation of X with itself).

---

## 1. Types of Correlation

1.  **Pearson ($r$):** Measures **Linear** relationship. Sensitive to outliers.
2.  **Spearman ($\rho$):** Measures **Monotonic** relationship (Rank-based). Robust to outliers and catches non-linear (but monotonic) trends.
3.  **Kendall's $\tau$:** Similar to Spearman, better for small samples.

---

## 2. Red Flags

- **High Correlation (> 0.9) between Predictors:** Indicates [[Multicollinearity\|Multicollinearity]]. You should drop one or use dimensionality reduction ([[PCA\|PCA]]).
- **Low Correlation with Target:** Variable might be useless features (or relationship is non-linear).

---

## 3. Python Implementation

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load Data
df = sns.load_dataset('iris')
df_numeric = df.select_dtypes(include='number') # Correlation needs numbers only

# 1. Calculate Matrix
# method='pearson' (default), 'spearman', 'kendall'
corr_matrix = df_numeric.corr(method='pearson')

print(corr_matrix)

# 2. Visualize with Heatmap
plt.figure(figsize=(8, 6))
# mask=True to hide upper triangle (redundant)
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', mask=mask)
plt.show()
```

---

## Related Concepts

- [[stats/02_Hypothesis_Testing/Pearson Correlation\|Pearson Correlation]]
- [[stats/02_Hypothesis_Testing/Spearman's Rank Correlation\|Spearman's Rank Correlation]]
- [[Multicollinearity\|Multicollinearity]]
- [[stats/08_Visualization/Heatmap\|Heatmap]]
