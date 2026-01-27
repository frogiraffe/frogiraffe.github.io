---
{"dg-publish":true,"permalink":"/stats/02-statistical-inference/levene-s-test/","tags":["Diagnostics","Assumptions"]}
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

- [[stats/02_Statistical_Inference/Welch's T-Test\|Welch's T-Test]]
- [[stats/02_Statistical_Inference/One-Way ANOVA\|One-Way ANOVA]]

---

## References

- **Article:** Levene, H. (1960). Robust tests for equality of variances. In I. Olkin et al. (Eds.), *Contributions to Probability and Statistics* (pp. 278-292). Stanford University Press. [WorldCat](https://www.worldcat.org/title/contributions-to-probability-and-statistics-essays-in-honor-of-harold-hotelling/oclc/411786)
- **Article:** Brown, M. B., & Forsythe, A. B. (1974). Robust tests for the equality of variances. *Journal of the American Statistical Association*, 69(346), 364-367. [JSTOR Link](http://www.jstor.org/stable/2285141)
- **Book:** Field, A. (2018). *Discovering Statistics Using IBM SPSS Statistics* (5th ed.). Sage. [SAGE Link](https://us.sagepub.com/en-us/nam/discovering-statistics-using-ibm-spss-statistics/book249648)