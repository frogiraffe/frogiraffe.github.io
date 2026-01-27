---
{"dg-publish":true,"permalink":"/stats/02-statistical-inference/confidence-intervals/","tags":["Foundations","Inference","Estimation"]}
---

## Definition

> [!abstract] Core Statement
> A **Confidence Interval (CI)** is a range of values, derived from sample data, that is likely to contain the ==true population parameter== with a specified level of confidence (e.g., 95%). It quantifies the **uncertainty** in an estimate.

---

> [!tip] Intuition (ELI5): The "Safety Zone"
> You have a giant jar of jelly beans. You scoop some and see 10% are red. Instead of saying "Exactly 10% of the whole jar is red," you say: "I'm 95% sure the real number is between **8% and 12%**." That range is your safety zone. The wider it is, the safer you are, but the less specific you are.

> [!example] Real-Life Example: Political Polling
> A news agency says: "Candidate X has **52% support (±3%)**." This is a CI of [49%, 55%]. Since the interval includes numbers below 50%, it's a "statistical toss-up"—Candidate X might actually be losing, even though the sample average is winning.

---

## Purpose

1.  Provide a range estimate instead of a single point estimate.
2.  Communicate the **precision** of an estimate.
3.  Support hypothesis testing (if CI excludes a null value, reject $H_0$).

---

## When to Use

> [!success] Use Confidence Intervals When...
> - Reporting estimates of means, proportions, or regression coefficients.
> - Communicating uncertainty to stakeholders.
> - Comparing groups (CIs for difference or ratio).

---

## Theoretical Background

### General Formula (For a Mean)

$$
CI = \bar{X} \pm Z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}
$$

When $\sigma$ is unknown (almost always), use the [[stats/01_Foundations/T-Distribution\|T-Distribution]]:
$$
CI = \bar{X} \pm t_{\alpha/2, n-1} \cdot \frac{s}{\sqrt{n}}
$$

### Interpretation

> [!important] Correct Interpretation
> "If we repeated this experiment many times, 95% of the calculated confidence intervals would contain the true population parameter."

> [!warning] Common Misinterpretation
> ~~"There is a 95% probability the true parameter is within this interval."~~ (That's a Bayesian credible interval).

### CI Width Factors

| Factor | Effect on CI Width |
|--------|-------------------|
| Larger $n$ | Narrower (more precise). |
| Larger Confidence Level (99% vs 95%) | Wider. |
| Larger Variability ($s$) | Wider. |

---

## CI and Hypothesis Testing

- 95% CI for a **difference** (e.g., $\mu_1 - \mu_2$): If CI **excludes 0**, the difference is significant at $\alpha = 0.05$.
- 95% CI for an **Odds Ratio**: If CI **excludes 1**, the effect is significant.

---

## Python Implementation

```python
from scipy import stats
import numpy as np

data = np.array([12, 15, 14, 10, 13, 11, 16, 14])
n = len(data)
mean = np.mean(data)
se = stats.sem(data)  # Standard Error

# 95% CI
ci = stats.t.interval(confidence=0.95, df=n-1, loc=mean, scale=se)
print(f"Mean: {mean:.2f}")
print(f"95% CI: [{ci[0]:.2f}, {ci[1]:.2f}]")
```

---

## R Implementation

```r
data <- c(12, 15, 14, 10, 13, 11, 16, 14)

# t.test provides mean and 95% CI
t.test(data)$conf.int

# For regression coefficients
model <- lm(Y ~ X, data = df)
confint(model)
```

---

## Interpretation Guide

| Output | Interpretation |
|--------|----------------|
| 95% CI: [50, 70] | We are 95% confident the true mean is between 50 and 70. |
| CI for OR: [1.2, 3.5] | The odds ratio is significantly > 1. Effect is real. |
| CI for Diff: [-2, 5] | CI includes 0; difference is NOT significant. |

---

## Related Concepts

- [[stats/02_Statistical_Inference/Hypothesis Testing (P-Value & CI)\|Hypothesis Testing (P-Value & CI)]]
- [[stats/01_Foundations/Standard Error\|Standard Error]]
- [[stats/01_Foundations/T-Distribution\|T-Distribution]]
- [[stats/01_Foundations/Bayesian Statistics\|Bayesian Statistics]] - Credible Intervals.

---

## References

- **Book:** Wasserman, L. (2004). *All of Statistics*. Springer. [Springer Link](https://link.springer.com/book/10.1007/978-0-387-21736-9)
- **Book:** Rice, J. A. (2007). *Mathematical Statistics and Data Analysis* (3rd ed.). Duxbury. [Cengage Link](https://www.cengage.com/c/mathematical-statistics-and-data-analysis-3e-rice/9780534399429)
- **Article:** Cumming, G. (2014). The new statistics: Why and how. *Psychological Science*, 25(1), 7-29. [SAGE Link](https://journals.sagepub.com/doi/full/10.1177/0956797613504966)
