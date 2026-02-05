---
{"dg-publish":true,"permalink":"/stats/01-foundations/standard-error/","tags":["probability","inference","sampling","foundations"]}
---

## Definition

> [!abstract] Core Statement
> **Standard Error (SE)** is the ==standard deviation of a sampling distribution==. It quantifies the **precision** of a sample statistic (usually the mean) as an estimate of the population parameter. Smaller SE means more precise estimates.

![Sampling Distribution of the Mean: Convergence and Standard Error](https://upload.wikimedia.org/wikipedia/commons/7/71/Sampling_distribution.png)

---

> [!tip] Intuition (ELI5): The Blur Filter
> Imagine you are taking a photo (the sample mean) of an object (the population mean).
> - **Standard Deviation** is how much the background naturally varies.
> - **Standard Error** is how "blurry" your focus is.
> The larger your "lens" (sample size), the less blur (Standard Error) you have, and the sharper your estimate of the truth becomes.

---

## Purpose

1. Quantify **uncertainty** in sample estimates.
2. Construct [[stats/02_Statistical_Inference/Confidence Intervals\|Confidence Intervals]].
3. Calculate test statistics (t-statistic, z-statistic).
4. Distinguish between **variability in data (SD)** and **variability in estimates (SE)**.

---

## When to Use

> [!success] Use Standard Error When...
> - Reporting the **precision** of a sample mean.
> - Constructing confidence intervals or hypothesis tests.
> - Comparing the reliability of different samples.

---

## Theoretical Background

### Formula (Standard Error of the Mean)

$$
SE = \frac{s}{\sqrt{n}} = \frac{\sigma}{\sqrt{n}}
$$

where:
- $s$ = sample [[stats/01_Foundations/Standard Deviation\|Standard Deviation]]
- $n$ = sample size
- $\sigma$ = population SD (usually unknown)

### Key Insight

> [!important] SE Decreases with Sample Size
> - Doubling precision requires **4 times** the sample size.
> - SE captures the [[stats/01_Foundations/Central Limit Theorem (CLT)\|Central Limit Theorem (CLT)]]: as $n$ increases, the sampling distribution narrows.

### SD vs SE

| Measure | What It Describes | Changes With $n$? |
|---------|-------------------|-------------------|
| **Standard Deviation (SD)** | Variability in the **data** (individual observations). | No. More data doesn't change population spread. |
| **Standard Error (SE)** | Variability in the **sample mean** (precision of estimate). | Yes. SE $\propto 1/\sqrt{n}$. |

> [!warning] Common Confusion
> Researchers sometimes report SE when they should report SD (to describe data), or vice versa. Be clear about what you're measuring.

---

## Assumptions

- [ ] **Random Sampling:** Sample is representative of the population.
- [ ] **Independence:** Observations are independent.
- [ ] For small $n$: **Normality** (or use [[stats/01_Foundations/T-Distribution\|T-Distribution]]).

---

## Limitations

> [!warning] Pitfalls
> 1. **SE alone is not informative:** Always report it with the mean and sample size.
2. **Assumes IID data:** Clustered or dependent data violates assumptions; use robust SE or multilevel models.

---

## Python Implementation

```python
import numpy as np
from scipy import stats

data = np.array([23, 25, 27, 22, 24, 26, 28, 21])

# Mean
mean = np.mean(data)

# Standard Error
se = stats.sem(data)  # Equivalent to: np.std(data, ddof=1) / np.sqrt(len(data))

print(f"Mean: {mean:.2f}")
print(f"SE: {se:.2f}")

# 95% Confidence Interval using SE
ci = stats.t.interval(confidence=0.95, df=len(data)-1, loc=mean, scale=se)
print(f"95% CI: [{ci[0]:.2f}, {ci[1]:.2f}]")
```

---

## R Implementation

```r
data <- c(23, 25, 27, 22, 24, 26, 28, 21)

# Mean
mean(data)

# Standard Error (manual)
se <- sd(data) / sqrt(length(data))
print(se)

# Alternatively, use plotrix package
library(plotrix)
std.error(data)

# 95% CI
t.test(data)$conf.int
```

---

## Interpretation Guide

| Scenario | Interpretation |
|----------|----------------|
| Mean = 50, SE = 2 | The true population mean is likely within ±4 of 50 (roughly 2 SE). |
| SE = 0.5 with n=100 vs SE = 2 with n=25 | Larger sample gives more precise estimate. |
| SE decreases by half | Precision has doubled (requires 4x sample size). |

---

## Related Concepts

- [[stats/01_Foundations/Standard Deviation\|Standard Deviation]] - Variability in data.
- [[stats/01_Foundations/Central Limit Theorem (CLT)\|Central Limit Theorem (CLT)]] - Why SE exists.
- [[stats/02_Statistical_Inference/Confidence Intervals\|Confidence Intervals]] - Built from SE.
- [[stats/01_Foundations/T-Distribution\|T-Distribution]] - Used with SE when $\sigma$ unknown.

---

## References

- **Book:** Wasserman, L. (2004). *All of Statistics*. Springer. [Springer Link](https://link.springer.com/book/10.1007/978-0-387-21736-9)
- **Book:** Rice, J. A. (2007). *Mathematical Statistics and Data Analysis* (3rd ed.). Duxbury. [Cengage](https://www.cengage.com/c/mathematical-statistics-and-data-analysis-3e-rice/9780534399429/)
- **Book:** Hogg, R. V., McKean, J. W., & Craig, A. T. (2019). *Introduction to Mathematical Statistics* (8th ed.). Pearson. [Pearson](https://www.pearson.com/en-us/subject-catalog/p/introduction-to-mathematical-statistics/P200000006263/)
