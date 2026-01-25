---
{"dg-publish":true,"permalink":"/stats/02-hypothesis-testing/z-test/","tags":["Hypothesis-Testing","Parametric-Tests","Foundations"]}
---

## Definition

> [!abstract] Core Statement
> A **Z-Test** is a statistical test used to determine whether a sample mean differs significantly from a population mean when the **Population Standard Deviation ($\sigma$) is Known** and/or the sample size is **Large** ($n \ge 30$).

$$ Z = \frac{\bar{x} - \mu_0}{\sigma / \sqrt{n}} $$

---

## Z-Test vs T-Test

| Feature | Z-Test | T-Test |
|---------|--------|--------|
| **Standard Deviation** | **Known ($\sigma$)** | Unknown (use sample $s$) |
| **Sample Size** | Large ($n \ge 30$) | Any (esp. Small $n < 30$) |
| **Distribution** | Standard Normal ($Z$) | Student's $t$ (fatter tails) |

> [!note] Practical Reality
> In real life, we almost *never* know the true population $\sigma$. Therefore, **T-tests are almost always preferred**, even for large samples (where t approaches Z anyway). Z-tests are mostly taught for theoretical introduction.

---

## One-Sample Z-Test Example

> [!example] IQ Test
> National Average IQ = 100, $\sigma = 15$. (We know this from census).
> A school samples **50 students** with Mean IQ = 105.
> 
> **Calculation:**
> $$ SE = 15 / \sqrt{50} \approx 2.12 $$
> $$ Z = \frac{105 - 100}{2.12} = \frac{5}{2.12} \approx 2.36 $$
> 
> **Decision:**
> -   Critical Z (95% confidence) = 1.96.
> -   $2.36 > 1.96$. **Reject Null.**
> -   This school is significantly smarter than average.

---

## Two-Proportion Z-Test

This is the **most common real-world use case** for Z-tests (A/B Testing).
Used for comparing two conversion rates ($p_1$ vs $p_2$).

$$ Z = \frac{\hat{p}_1 - \hat{p}_2}{\sqrt{\hat{p}(1-\hat{p})(\frac{1}{n_1} + \frac{1}{n_2})}} $$

---

## Python Implementation

```python
from statsmodels.stats.weightstats import ztest

# 1. One Sample Mean
# (Requires raw data, treats sample std as population std if n is large)
stat, p = ztest(data, value=100)

# 2. Proportions (A/B Test) - The more common use
from statsmodels.stats.proportion import proportions_ztest
count = [30, 45]     # Successes
nobs = [1000, 1000]  # Total observations
z_stat, p_val = proportions_ztest(count, nobs)
```

---

## Related Concepts

- [[stats/02_Hypothesis_Testing/One-Sample t-test\|One-Sample t-test]] - The alternative for unknown $\sigma$.
- [[stats/01_Foundations/Central Limit Theorem (CLT)\|Central Limit Theorem (CLT)]] - Justifies the Normal approx.
- [[A/B Testing\|A/B Testing]] - Uses Two-Proportion Z-test.
