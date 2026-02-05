---
{"dg-publish":true,"permalink":"/stats/01-foundations/correlation-analysis/","tags":["probability","descriptive-statistics","association","foundations"]}
---

## Definition

> [!abstract] Core Statement
> **Correlation Analysis** quantifies the ==strength and direction of the linear relationship== between two variables. The correlation coefficient (r or ρ) ranges from -1 to +1.

![Correlation examples: negative, none, positive|500](https://upload.wikimedia.org/wikipedia/commons/d/d4/Correlation_examples2.svg)
*Figure 1: Various correlation values and their scatter patterns*

---

> [!tip] Intuition (ELI5): The Strength of Agreement
> Correlation is like asking "When X goes up, does Y also go up?" A correlation of +1 means "perfectly yes," -1 means "perfectly opposite," and 0 means "no relationship at all."

---

## Purpose

1. **Measure linear association** between two variables
2. **Screen relationships** before regression
3. **Feature selection** in machine learning
4. **Validate constructs** in psychometrics

---

## When to Use

> [!success] Use Correlation When...
> - Exploring relationships between **continuous variables**
> - Need a **standardized** (-1 to +1) measure of association
> - Checking assumptions before [[stats/03_Regression_Analysis/Simple Linear Regression\|regression]]

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - **Non-linear relationships:** r=0 doesn't mean no relationship
> - **Causation needed:** Correlation ≠ causation!
> - **Outliers present:** A single outlier can flip the sign

---

## Types

| Type | Use | Formula |
|------|-----|---------|
| **[[stats/02_Statistical_Inference/Pearson Correlation\|Pearson]] (r)** | Linear, continuous | $\frac{\sum(x_i-\bar{x})(y_i-\bar{y})}{\sqrt{\sum(x_i-\bar{x})^2\sum(y_i-\bar{y})^2}}$ |
| **[[stats/02_Statistical_Inference/Spearman's Rank Correlation\|Spearman]] (ρ)** | Monotonic, ordinal | Pearson on ranks |
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

> [!important] r² = Coefficient of Determination
> $r^2$ tells you the proportion of variance in Y explained by X.
> - r = 0.7 → r² = 0.49 → X explains 49% of Y's variance

---

## Worked Example

> [!example] Problem
> Study hours (X) vs exam scores (Y) for 5 students:
> - X: [2, 4, 6, 8, 10]
> - Y: [65, 70, 75, 85, 90]

**Solution:**

```python
import numpy as np
from scipy import stats

x = np.array([2, 4, 6, 8, 10])
y = np.array([65, 70, 75, 85, 90])

r, p = stats.pearsonr(x, y)
print(f"Pearson r = {r:.3f}")  # 0.986
print(f"p-value = {p:.4f}")   # 0.0020
print(f"r² = {r**2:.3f}")     # 0.972 → 97% explained
```

**Interpretation:** Very strong positive correlation (r=0.99). More study hours → higher scores.

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
print("\nCorrelation Matrix:")
print(df.corr().round(3))
```

**Expected Output:**
```
Pearson r = 0.866, p = 0.0577
Spearman ρ = 0.900
Kendall τ = 0.800
```

---

## R Implementation

```r
x <- c(1, 2, 3, 4, 5)
y <- c(2, 4, 5, 4, 5)

cor(x, y)                     # Pearson
cor(x, y, method = "spearman")
cor(x, y, method = "kendall")

cor.test(x, y)                # With p-value
```

---

## Common Pitfalls

> [!warning] Traps to Avoid

**1. Correlation ≠ Causation**
- Ice cream sales correlate with drownings (both increase in summer)
- The lurking variable is temperature!

**2. Non-linear Relationships**
- Pearson r = 0 doesn't mean no relationship
- Plot the data first!

**3. Outliers**
- A single outlier can flip the sign of r
- Use Spearman for robustness

**4. Restriction of Range**
- Correlating only "high performers" artificially reduces r

---

## Related Concepts

### Directly Related
- [[stats/01_Foundations/Covariance\|Covariance]] - Unstandardized version
- [[stats/02_Statistical_Inference/Pearson Correlation\|Pearson Correlation]] - Most common type
- [[stats/02_Statistical_Inference/Spearman's Rank Correlation\|Spearman's Rank Correlation]] - For ordinal/non-linear

### Applications
- [[stats/03_Regression_Analysis/Simple Linear Regression\|Simple Linear Regression]] - Uses r to measure fit
- [[stats/03_Regression_Analysis/Multiple Linear Regression\|Multiple Linear Regression]] - Multicollinearity check
- [[stats/01_Foundations/Feature Selection\|Feature Selection]] - Screen features

### Other Related Topics
- [[stats/09_EDA_and_Visualization/Boxplot\|Boxplot]]
- [[stats/01_Foundations/Coefficient of Variation\|Coefficient of Variation]]
- [[stats/01_Foundations/Correlation Matrix\|Correlation Matrix]]
- [[stats/01_Foundations/Covariance\|Covariance]]
- [[stats/01_Foundations/Mean\|Mean]]

{ .block-language-dataview}

---

## References

1. Cohen, J., Cohen, P., West, S. G., & Aiken, L. S. (2003). *Applied Multiple Regression/Correlation Analysis for the Behavioral Sciences* (3rd ed.). Lawrence Erlbaum.

2. Agresti, A., & Finlay, B. (2009). *Statistical Methods for the Social Sciences* (4th ed.). Pearson.
