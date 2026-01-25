---
{"dg-publish":true,"permalink":"/stats/02-hypothesis-testing/levene-s-test/","tags":["Diagnostics","Assumptions"]}
---

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

- [[stats/02_Hypothesis_Testing/Welch's T-Test\|Welch's T-Test]]
- [[stats/02_Hypothesis_Testing/One-Way ANOVA\|One-Way ANOVA]]