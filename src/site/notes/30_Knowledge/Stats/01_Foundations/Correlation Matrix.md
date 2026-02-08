---
{"dg-publish":true,"permalink":"/30-knowledge/stats/01-foundations/correlation-matrix/","tags":["probability","foundations"]}
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

- **High Correlation (> 0.9) between Predictors:** Indicates [[30_Knowledge/Stats/03_Regression_Analysis/VIF (Variance Inflation Factor)\|Multicollinearity]]. You should drop one or use dimensionality reduction ([[30_Knowledge/Stats/05_Unsupervised_Learning/PCA (Principal Component Analysis)\|PCA]]).
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

- [[30_Knowledge/Stats/02_Statistical_Inference/Pearson Correlation\|Pearson Correlation]]
- [[30_Knowledge/Stats/02_Statistical_Inference/Spearman's Rank Correlation\|Spearman's Rank Correlation]]
- [[30_Knowledge/Stats/03_Regression_Analysis/VIF (Variance Inflation Factor)\|Multicollinearity]]
- [[30_Knowledge/Stats/09_EDA_and_Visualization/Heatmap\|Heatmap]]

---

## When to Use

> [!success] Use Correlation Matrix When...
> - Refer to standard documentation
> - Refer to standard documentation

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Assumptions are violated
> - Alternative methods are more appropriate

---

## Python Implementation

```python
import numpy as np
import pandas as pd

# Example implementation of Correlation Matrix
# See documentation for details

data = np.random.randn(100)
print(f"Mean: {np.mean(data):.3f}")
print(f"Std: {np.std(data):.3f}")
```

---

## R Implementation

```r
# Correlation Matrix in R
set.seed(42)

# Example implementation
data <- rnorm(100)
summary(data)
```

---

## References

- **Book:** Rodgers, J. L., & Nicewander, W. A. (1988). Thirteen ways to look at the correlation coefficient. *The American Statistician*, 42(1), 59-66. [Taylor & Francis](https://www.tandfonline.com/doi/abs/10.1080/00031305.1988.10475524)
- **Book:** Cohen, J., et al. (2013). *Applied Multiple Regression/Correlation Analysis for the Behavioral Sciences* (3rd ed.). Routledge. [Routledge Link](https://www.routledge.com/Applied-Multiple-RegressionCorrelation-Analysis-for-the-Behavioral-Sciences/Cohen-Cohen-West-Aiken/p/book/9780805822236)
- **Book:** Wickham, H., & Grolemund, G. (2016). *R for Data Science*. O'Reilly Media. [O'Reilly Link](https://www.oreilly.com/library/view/r-for-data/9781491910382/)
