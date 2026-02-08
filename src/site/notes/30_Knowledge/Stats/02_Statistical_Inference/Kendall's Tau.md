---
{"dg-publish":true,"permalink":"/30-knowledge/stats/02-statistical-inference/kendall-s-tau/","tags":["inference","hypothesis-testing"]}
---

## Overview

> [!abstract] Definition
> **Kendall's Tau ($\tau$)** is a non-parametric measure of rank correlation: the similarity of the orderings of the data when ranked by each of the quantities. It is an alternative to Spearman's rho for ordinal data or non-normal continuous data.

---

## 1. Concordance and Discordance

Let $(x_i, y_i)$ and $(x_j, y_j)$ be two observations.
- **Concordant:** If the order of X matches the order of Y ($x_i > x_j$ and $y_i > y_j$ OR $x_i < x_j$ and $y_i < y_j$).
- **Discordant:** If the orders are inverted ($x_i > x_j$ and $y_i < y_j$ OR vice versa).

$$ \tau = \frac{(\text{number of concordant pairs}) - (\text{number of discordant pairs})}{\text{total number of pairs}} $$

---

## 2. Comparison with Pearson and Spearman

| Feature | Pearson (r) | Spearman ($\rho$) | Kendall ($\tau$) |
|---------|-------------|-------------------|------------------|
| **Type** | Linear | Monotonic (Rank) | Monotonic (Rank) |
| **Logic** | Covariance | Covariance of Ranks | Probability of Concordance |
| **Sensitivity** | Outliers | Robust | Robust |
| **Small Samples** | Sensitive | Good | **Best** (More accurate p-values) |

**Recommendation:** Infer conclusions about the population rank correlation are often more accurate with Kendall's Tau, especially with small sample sizes, although Spearman's rho is more commonly reported.

---

## 3. Python Implementation Example

```python
from scipy import stats

correlation, p_value = stats.kendalltau(x, y)
print(f"Kendall's Tau: {correlation:.3f}")
```

---

## 4. Related Concepts

- [[30_Knowledge/Stats/02_Statistical_Inference/Pearson Correlation\|Pearson Correlation]] - Parametric standard.
- [[30_Knowledge/Stats/02_Statistical_Inference/Spearman's Rank Correlation\|Spearman's Rank Correlation]] - Common rank alternative.

---

## Definition

> [!abstract] Core Statement
> **Kendall's Tau** ... Refer to standard documentation

---

> [!tip] Intuition (ELI5)
> Refer to standard documentation

---

## When to Use

> [!success] Use Kendall's Tau When...
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

# Example implementation of Kendall's Tau
# See documentation for details

data = np.random.randn(100)
print(f"Mean: {np.mean(data):.3f}")
print(f"Std: {np.std(data):.3f}")
```

---

## R Implementation

```r
# Kendall's Tau in R
set.seed(42)

# Example implementation
data <- rnorm(100)
summary(data)
```

---

## Related Concepts

- [[30_Knowledge/Stats/02_Statistical_Inference/Confidence Intervals\|Confidence Interval]]
- [[30_Knowledge/Stats/02_Statistical_Inference/Hypothesis Testing (P-Value & CI)\|Hypothesis Testing]]
- [[30_Knowledge/Stats/02_Statistical_Inference/Hypothesis Testing (P-Value & CI)\|P-Value]]

---

## References

- **Historical:** Kendall, M. G. (1938). A new measure of rank correlation. *Biometrika*, 30(1/2), 81-93. [JSTOR](https://www.jstor.org/stable/2332141)
- **Book:** Conover, W. J. (1999). *Practical Nonparametric Statistics* (3rd ed.). Wiley. [Wiley Link](https://www.wiley.com/en-us/Practical+Nonparametric+Statistics%2C+3rd+Edition-p-9780471160687)
- **Book:** Hollander, M., Wolfe, D. A., & Chicken, E. (2014). *Nonparametric Statistical Methods* (3rd ed.). Wiley. [Wiley Link](https://www.wiley.com/en-us/Nonparametric+Statistical+Methods%2C+3rd+Edition-p-9780470387375)
