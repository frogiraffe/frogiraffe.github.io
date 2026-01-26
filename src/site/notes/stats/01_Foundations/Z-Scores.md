---
{"dg-publish":true,"permalink":"/stats/01-foundations/z-scores/","tags":["Foundations","Descriptive-Statistics","Normalization"]}
---


## Definition

> [!abstract] Core Statement
> A **Z-Score** measures how many **standard deviations** a data point is from the mean. It standardizes values from any normal distribution to the **standard normal distribution** (μ=0, σ=1).

$$
z = \frac{x - \mu}{\sigma}
$$

**Intuition (ELI5):** Imagine test scores from two classes — one averages 70 (SD=10), the other 85 (SD=5). If you scored 80 in the first class (z=+1) and 90 in the second (z=+1), you performed equally well relative to each class, even though the raw scores differ.

**Key Properties:**
- Z-score = 0 → Value equals the mean
- Z-score > 0 → Value is above the mean
- Z-score < 0 → Value is below the mean

---

## When to Use

> [!success] Use Z-Scores When...
> - Comparing values from **different distributions** (different scales/units).
> - Identifying **outliers** (|z| > 2 or 3 is unusual).
> - Calculating **probabilities** using the standard normal table.
> - **Standardizing features** for machine learning.
> - Testing **hypotheses** about population means.

> [!failure] Do NOT Use Z-Scores When...
> - Data is **not normally distributed** (percentiles are better).
> - Comparing within the **same distribution** (raw values suffice).
> - Sample size is **very small** (z-distribution unreliable).

---

## Theoretical Background

### Standard Normal Distribution

Converting to z-scores transforms any normal distribution $N(\mu, \sigma^2)$ to the **standard normal** $N(0, 1)$:

| Original Value | Z-Score | Interpretation |
|----------------|---------|----------------|
| $\mu - 2\sigma$ | $z = -2$ | 2 SDs below mean |
| $\mu - \sigma$ | $z = -1$ | 1 SD below mean |
| $\mu$ | $z = 0$ | At the mean |
| $\mu + \sigma$ | $z = +1$ | 1 SD above mean |
| $\mu + 2\sigma$ | $z = +2$ | 2 SDs above mean |

### The Empirical Rule (68-95-99.7)

| Z-Score Range | Percentage of Data |
|---------------|--------------------|
| $-1 \leq z \leq 1$ | ~68% |
| $-2 \leq z \leq 2$ | ~95% |
| $-3 \leq z \leq 3$ | ~99.7% |

### Common Z-Score Thresholds

| Z-Score | Probability (one-tail) | Use |
|---------|------------------------|-----|
| 1.645 | 0.05 | 90% one-sided CI |
| 1.96 | 0.025 | 95% two-sided CI |
| 2.576 | 0.005 | 99% two-sided CI |

---

## Implementation

### Python

```python
import numpy as np
from scipy import stats

# Sample data
data = [72, 85, 90, 68, 95, 88, 76, 82, 79, 91]

# ========== CALCULATE Z-SCORES ==========
mean = np.mean(data)
std = np.std(data, ddof=0)  # Population std
z_scores = (data - mean) / std

print("Data:", data)
print(f"Mean: {mean:.2f}, Std: {std:.2f}")
print("Z-Scores:", np.round(z_scores, 2))

# ========== USING SCIPY ==========
z_scores_scipy = stats.zscore(data)
print("Z-Scores (scipy):", np.round(z_scores_scipy, 2))

# ========== FIND OUTLIERS ==========
outliers = [x for x, z in zip(data, z_scores) if abs(z) > 2]
print(f"Outliers (|z| > 2): {outliers}")

# ========== Z-SCORE TO PROBABILITY ==========
z = 1.96
prob_below = stats.norm.cdf(z)  # P(Z < 1.96)
prob_above = 1 - prob_below      # P(Z > 1.96)
print(f"\nP(Z < {z}) = {prob_below:.4f}")
print(f"P(Z > {z}) = {prob_above:.4f}")

# ========== PROBABILITY TO Z-SCORE ==========
prob = 0.95
z_value = stats.norm.ppf(prob)  # Inverse CDF
print(f"Z-score for P={prob}: {z_value:.3f}")
```

### R

```r
# Sample data
data <- c(72, 85, 90, 68, 95, 88, 76, 82, 79, 91)

# ========== CALCULATE Z-SCORES ==========
z_scores <- scale(data)
print("Z-Scores:")
print(round(z_scores, 2))

# ========== FIND OUTLIERS ==========
outliers <- data[abs(z_scores) > 2]
cat("Outliers (|z| > 2):", outliers, "\n")

# ========== Z-SCORE TO PROBABILITY ==========
z <- 1.96
prob_below <- pnorm(z)  # P(Z < 1.96)
cat("\nP(Z <", z, ") =", prob_below, "\n")

# ========== PROBABILITY TO Z-SCORE ==========
prob <- 0.95
z_value <- qnorm(prob)  # Inverse CDF
cat("Z-score for P =", prob, ":", z_value, "\n")
```

---

## Interpretation Guide

| Z-Score | Interpretation | Action |
|---------|----------------|--------|
| $z = 0$ | Exactly average | No concern |
| $\|z\| < 1$ | Within 1 SD — typical value | Normal range |
| $1 < \|z\| < 2$ | Somewhat unusual | Worth noting |
| $\|z\| > 2$ | Unusual (~5% of data) | Investigate as potential outlier |
| $\|z\| > 3$ | Very rare (~0.3% of data) | Likely outlier or data error |

---

## Common Pitfalls

> [!warning] Traps to Avoid
>
> **1. Using with Non-Normal Data**
> - Z-scores assume normality; skewed data gives misleading z-scores.
> - Solution: Use percentile ranks for non-normal data.
>
> **2. Confusing Population vs Sample SD**
> - Population: $\sigma = \sqrt{\frac{\sum(x-\mu)^2}{n}}$
> - Sample: $s = \sqrt{\frac{\sum(x-\bar{x})^2}{n-1}}$
> - Solution: Use `ddof=1` in numpy for sample SD.
>
> **3. Interpreting Z-Score as Percentile**
> - $z = 1$ is NOT the 100th percentile — it's the ~84th.
> - Solution: Use `norm.cdf(z)` to convert z to percentile.

---

## Worked Example

> [!example] Comparing SAT and ACT Scores
> **Problem:** Alice scored 1400 on SAT. Bob scored 32 on ACT. Who did better?
>
> **Given:**
> - SAT: μ = 1060, σ = 217
> - ACT: μ = 21, σ = 5.8
>
> **Step 1: Calculate Z-Scores**
> $$z_{Alice} = \frac{1400 - 1060}{217} = 1.57$$
> $$z_{Bob} = \frac{32 - 21}{5.8} = 1.90$$
>
> **Conclusion:** Bob performed better relative to test-takers (1.90 > 1.57 SDs above mean).

---

## Related Concepts

- [[stats/01_Foundations/Standard Deviation\|Standard Deviation]] — Base measure for z-score
- [[stats/01_Foundations/Normal Distribution\|Normal Distribution]] — Z-scores assume normality
- [[stats/01_Foundations/Feature Scaling\|Feature Scaling]] — Z-scores for standardization
- [[stats/01_Foundations/T-Distribution\|T-Distribution]] — Used instead of z when σ unknown and n small
