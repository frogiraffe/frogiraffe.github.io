---
{"dg-publish":true,"permalink":"/stats/01-foundations/correlation-analysis/","tags":["Statistics","Descriptive-Statistics","Association"]}
---


## Definition

> [!abstract] Core Statement
> **Correlation Analysis** quantifies the ==strength and direction of the linear relationship== between two variables. The correlation coefficient (r or ρ) ranges from -1 to +1.

---

## Types

| Type | Use | Formula |
|------|-----|---------|
| **Pearson (r)** | Linear, continuous | $\frac{\sum(x_i-\bar{x})(y_i-\bar{y})}{\sqrt{\sum(x_i-\bar{x})^2\sum(y_i-\bar{y})^2}}$ |
| **Spearman (ρ)** | Monotonic, ordinal | Pearson on ranks |
| **Kendall (τ)** | Ordinal, robust | Based on concordant/discordant pairs |

---

## Interpretation

| r Value | Interpretation |
|---------|----------------|
| 0.9 - 1.0 | Very strong positive |
| 0.7 - 0.9 | Strong positive |
| 0.5 - 0.7 | Moderate positive |
| 0.3 - 0.5 | Weak positive |
| 0.0 - 0.3 | Negligible |
| < 0 | Negative (same scale) |

---

## Python Implementation

```python
import numpy as np
from scipy import stats
import pandas as pd

x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

# Pearson
r, p = stats.pearsonr(x, y)
print(f"Pearson r = {r:.3f}, p = {p:.4f}")

# Spearman
rho, p = stats.spearmanr(x, y)
print(f"Spearman ρ = {rho:.3f}")

# Kendall
tau, p = stats.kendalltau(x, y)
print(f"Kendall τ = {tau:.3f}")

# Correlation matrix
df = pd.DataFrame({'A': x, 'B': y, 'C': [1, 3, 2, 5, 4]})
print(df.corr())  # Pearson by default
print(df.corr(method='spearman'))
```

---

## R Implementation

```r
cor(x, y)                    # Pearson
cor(x, y, method = "spearman")
cor(x, y, method = "kendall")

cor.test(x, y)               # With p-value
```

---

## Common Pitfalls

> [!warning] Traps
>
> **1. Correlation ≠ Causation**
> - Ice cream sales correlate with drownings (both increase in summer)
>
> **2. Non-linear Relationships**
> - Pearson r = 0 doesn't mean no relationship
>
> **3. Outliers**
> - A single outlier can flip the sign of r

---

## Related Concepts

- [[stats/02_Statistical_Inference/Pearson Correlation\|Pearson Correlation]] — Most common
- [[stats/01_Foundations/Spearman Rank Correlation\|Spearman Rank Correlation]] — For ordinal/non-linear
- [[stats/01_Foundations/Covariance\|Covariance]] — Unstandardized version

---

## References

- **Book:** Cohen, J., et al. (2003). *Applied Multiple Regression/Correlation Analysis*. Lawrence Erlbaum.
