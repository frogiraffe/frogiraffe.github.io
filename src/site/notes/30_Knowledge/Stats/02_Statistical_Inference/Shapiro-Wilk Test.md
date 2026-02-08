---
{"dg-publish":true,"permalink":"/30-knowledge/stats/02-statistical-inference/shapiro-wilk-test/","tags":["inference","hypothesis-testing"]}
---

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

- [[30_Knowledge/Stats/09_EDA_and_Visualization/Q-Q Plot\|Q-Q Plot]]
- [[30_Knowledge/Stats/01_Foundations/Normal Distribution\|Normal Distribution]]

---

## Definition

> [!abstract] Core Statement
> **Shapiro-Wilk Test** ... Refer to standard documentation

---

> [!tip] Intuition (ELI5)
> Refer to standard documentation

---

## When to Use

> [!success] Use Shapiro-Wilk Test When...
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
# Shapiro-Wilk Test in R
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

- **Historical:** Shapiro, S. S., & Wilk, M. B. (1965). An analysis of variance test for normality (complete samples). *Biometrika*, 52(3-4), 591-611. [JSTOR Link](http://www.jstor.org/stable/2333709)
- **Article:** Razali, N. M., & Wah, Y. B. (2011). Power comparisons of Shapiro-Wilk, Kolmogorov-Smirnov, Lilliefors and Anderson-Darling tests. *Journal of Statistical Modeling and Analytics*, 2(1), 21-33. [Full Text](https://www.nrc.gov/docs/ML1409/ML14093A468.pdf)
- **Book:** Thode, H. C. (2002). *Testing for Normality*. Marcel Dekker. [CRC Press](https://www.routledge.com/Testing-for-Normality/Thode/p/book/9780824796136)
