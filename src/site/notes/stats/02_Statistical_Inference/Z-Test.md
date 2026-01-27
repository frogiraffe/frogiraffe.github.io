---
{"dg-publish":true,"permalink":"/stats/02-statistical-inference/z-test/","tags":["Hypothesis-Testing","Parametric-Tests","Foundations"]}
---

## Definition

> [!abstract] Core Statement
> A **Z-Test** is a statistical test used to determine whether a sample mean differs significantly from a population mean when the **Population Standard Deviation ($\sigma$) is Known** and/or the sample size is **Large** ($n \ge 30$).

![Z-Test Rejection Region](https://upload.wikimedia.org/wikipedia/commons/0/0b/Null-hypothesis-region-eng.png)

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

## R Implementation

```r
# One-sample Z-test (Manual, as base R doesn't have it by default)
z_test <- function(sample_mean, pop_mean, pop_std, n) {
  z_score <- (sample_mean - pop_mean) / (pop_std / sqrt(n))
  p_value <- 2 * pnorm(-abs(z_score)) # Two-tailed
  return(list(z = z_score, p = p_value))
}

# Run
res <- z_test(sample_mean = 105, pop_mean = 100, pop_std = 15, n = 50)
print(res)

# Or using BSDA package
# install.packages("BSDA")
# library(BSDA)
# z.test(data_vector, mu=100, sigma.x=15)
```

---

## Related Concepts

- [[stats/02_Statistical_Inference/One-Sample t-test\|One-Sample t-test]] - The alternative for unknown $\sigma$.
- [[stats/01_Foundations/Central Limit Theorem (CLT)\|Central Limit Theorem (CLT)]] - Justifies the Normal approx.
- [[stats/02_Statistical_Inference/A-B Testing\|A/B Testing]] - Uses Two-Proportion Z-test.

---

## References

- **Book:** Sprinthall, R. C. (2011). *Basic Statistical Analysis* (9th ed.). Pearson. [Pearson Link](https://www.pearson.com/en-gb/subject-catalog/p/basic-statistical-analysis-pearson-new-international-edition/P100000109765)
- **Book:** Daniel, W. W., & Cross, C. L. (2013). *Biostatistics: A Foundation for Analysis in the Health Sciences* (10th ed.). Wiley. [Wiley Link](https://www.wiley.com/en-us/Biostatistics%3A+A+Foundation+for+Analysis+in+the+Health+Sciences%2C+10th+Edition-p-9781118302798)
- **Book:** Freedman, D., Pisani, R., & Purves, R. (2007). *Statistics* (4th ed.). W. W. Norton & Company. [W.W. Norton Link](https://wwnorton.com/books/9780393929720)
