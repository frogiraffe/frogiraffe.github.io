---
{"dg-publish":true,"permalink":"/30-knowledge/stats/02-statistical-inference/levene-s-test/","tags":["inference","hypothesis-testing"]}
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

- [[30_Knowledge/Stats/02_Statistical_Inference/Welch's T-Test\|Welch's T-Test]]
- [[30_Knowledge/Stats/02_Statistical_Inference/One-Way ANOVA\|One-Way ANOVA]]

---

## Definition

> [!abstract] Core Statement
> **Levene's Test** ... Refer to standard documentation

---

> [!tip] Intuition (ELI5)
> Refer to standard documentation

---

## When to Use

> [!success] Use Levene's Test When...
> - Refer to standard documentation
> - Refer to standard documentation

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Assumptions of the test are violated
> - Sample size doesn't meet minimum requirements

---

## Python Implementation

```python
from scipy import stats
import numpy as np

# Sample data
group1 = np.random.normal(10, 2, 30)
group2 = np.random.normal(12, 2, 30)

# Perform test
statistic, pvalue = stats.ttest_ind(group1, group2)

print(f"Test Statistic: {statistic:.4f}")
print(f"P-value: {pvalue:.4f}")
print(f"Significant at α=0.05: {pvalue < 0.05}")
```

---

## R Implementation

```r
# Levene's Test in R
set.seed(42)

# Sample data
group1 <- rnorm(30, mean = 10, sd = 2)
group2 <- rnorm(30, mean = 12, sd = 2)

# Perform test
result <- t.test(group1, group2)
print(result)
```

---

## Related Concepts

- [[30_Knowledge/Stats/02_Statistical_Inference/Confidence Intervals\|Confidence Interval]]
- [[30_Knowledge/Stats/02_Statistical_Inference/Hypothesis Testing (P-Value & CI)\|Hypothesis Testing]]
- [[30_Knowledge/Stats/02_Statistical_Inference/Hypothesis Testing (P-Value & CI)\|P-Value]]

---

## References

- **Article:** Levene, H. (1960). Robust tests for equality of variances. In I. Olkin et al. (Eds.), *Contributions to Probability and Statistics* (pp. 278-292). Stanford University Press. [WorldCat](https://www.worldcat.org/title/contributions-to-probability-and-statistics-essays-in-honor-of-harold-hotelling/oclc/411786)
- **Article:** Brown, M. B., & Forsythe, A. B. (1974). Robust tests for the equality of variances. *Journal of the American Statistical Association*, 69(346), 364-367. [JSTOR Link](http://www.jstor.org/stable/2285141)
- **Book:** Field, A. (2018). *Discovering Statistics Using IBM SPSS Statistics* (5th ed.). Sage. [SAGE Link](https://us.sagepub.com/en-us/nam/discovering-statistics-using-ibm-spss-statistics/book249648)