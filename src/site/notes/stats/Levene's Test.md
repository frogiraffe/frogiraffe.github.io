---
{"dg-publish":true,"permalink":"/stats/levene-s-test/","tags":["Statistics","Diagnostics","Assumptions"]}
---


# Levene's Test

## Overview

> [!abstract] Definition
> **Levene's Test** assesses the assumption of **Homogeneity of Variance** (Equal Variances) required by T-Tests and ANOVA.
> *   $H_0$: Variances are equal.
> *   $H_1$: Variances are not equal.

> [!warning] Interpretation
> *   **Significant (p < 0.05):** Bad news. Variances are unequal. Switch to **Welch's T-Test**.
> *   **Not Significant (p > 0.05):** Good news. Assumption met.

---

## 1. Python Implementation

```python
from scipy import stats
# Compare variances of Group A and Group B
stat, p = stats.levene(group_A, group_B)

if p < 0.05:
    print("Variances Unequal (Use Welch)")
```

---

## 2. R Implementation

```r
library(car)

# leveneTest(Outcome ~ Group)
leveneTest(Score ~ Group, data = df)

# Note: The car package version is robust (uses deviations from median).
```

---

## 3. Related Concepts

- [[stats/Welch's T-Test\|Welch's T-Test]]
- [[stats/One-Way ANOVA\|One-Way ANOVA]]