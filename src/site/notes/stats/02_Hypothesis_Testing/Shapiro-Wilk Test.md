---
{"dg-publish":true,"permalink":"/stats/02-hypothesis-testing/shapiro-wilk-test/","tags":["Diagnostics","Normality","Assumptions"]}
---


# Shapiro-Wilk Test

## Overview

> [!abstract] Definition
> **Shapiro-Wilk Test** checks if data comes from a **Normal Distribution**. It is one of the most powerful normality tests.
> *   $H_0$: Data is Normal.
> *   $H_1$: Data is Not Normal.

> [!tip] Sample Size
> Best for $n < 50$. For large samples, it is too sensitive (flags minor deviations). Use Q-Q Plots for $n > 50$.

---

## 1. Python Implementation

```python
from scipy import stats
stat, p = stats.shapiro(data)

if p < 0.05:
    print("Not Normal (Reject H0)")
else:
    print("Normal (Fail to Reject H0)")
```

---

## 2. R Implementation

```r
# Built-in function
shapiro.test(data)

# If p < 0.05, consider non-parametric tests like Wilcoxon.
```

---

## 3. Related Concepts

- [[stats/08_Visualization/Q-Q Plot\|Q-Q Plot]]
- [[stats/01_Foundations/Normal Distribution\|Normal Distribution]]
