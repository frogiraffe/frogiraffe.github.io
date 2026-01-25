---
{"dg-publish":true,"permalink":"/stats/01-foundations/standard-deviation/","tags":["Foundations","Variability",{"Descriptivealiases":null},"SD","Sample Variance"]}
---

## Definition

> [!abstract] Core Statement
> **Standard Deviation (SD or $\sigma$)** quantifies the ==amount of variation or dispersion== in a dataset. It measures how far individual observations typically deviate from the mean. A low SD indicates data points cluster close to the mean; a high SD indicates they are spread out.

---

## Purpose

1. Describe the **spread** of a distribution.
2. Identify **outliers** (values > 2-3 SD from mean are unusual).
3. Standardize scores (Z-scores = deviations in SD units).
4. Foundation for [[stats/01_Foundations/Standard Error\|Standard Error]], [[stats/02_Hypothesis_Testing/Confidence Intervals\|Confidence Intervals]], and inference.

---

## When to Use

> [!success] Use Standard Deviation When...
> - Describing the variability of **continuous data**.
> - Data is approximately **normally distributed**.
> - You need a measure of spread in the **same units** as the data.

> [!failure] Alternatives
> - **Non-normal data:** Use **Interquartile Range (IQR)** or **Median Absolute Deviation (MAD)**.
> - **Ordinal data:** Use range or IQR.

---

## Theoretical Background

### Population vs Sample

| Type | Formula | When to Use |
|------|---------|-------------|
| **Population SD ($\sigma$)** | $\sigma = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(X_i - \mu)^2}$ | You have data for the **entire population**. |
| **Sample SD ($s$)** | $s = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(X_i - \bar{X})^2}$ | You have a **sample** from a population. |

> [!important] Why $n-1$ (Bessel's Correction)?
> Using $n$ would **underestimate** the population variance. The $n-1$ correction makes the sample variance an **unbiased estimator** of the population variance.

### Variance ($s^2$ or $\sigma^2$)

Variance is the squared standard deviation. SD is preferred for interpretation because it's in the **original units**.

### Properties

- **Always non-negative:** $s \ge 0$.
- **Zero only if all values are identical.**
- **Affected by outliers:** One extreme value can inflate SD dramatically.

---

## Assumptions

- [ ] **Continuous or discrete numeric data.**
- [ ] For interpretation as "typical deviation," data should be **unimodal and roughly symmetric**.

---

## Limitations

> [!warning] Pitfalls
> 1. **Sensitive to Outliers:** A single extreme value inflates SD. Use MAD for robustness.
> 2. **Assumes Symmetry:** For skewed distributions, SD can be misleading. Report median and IQR instead.
> 3. **Not Comparable Across Different Scales:** SD of income (\$) vs SD of age (years) are meaningless to compare. Use **Coefficient of Variation (CV = SD/Mean)**.

---

## Python Implementation

```python
import numpy as np

data = np.array([10, 12, 23, 23, 16, 23, 21, 16])

# Sample Standard Deviation (default: ddof=1)
sd_sample = np.std(data, ddof=1)
print(f"Sample SD: {sd_sample:.2f}")

# Population Standard Deviation (ddof=0)
sd_pop = np.std(data, ddof=0)
print(f"Population SD: {sd_pop:.2f}")

# Variance
var_sample = np.var(data, ddof=1)
print(f"Sample Variance: {var_sample:.2f}")
```

---

## R Implementation

```r
data <- c(10, 12, 23, 23, 16, 23, 21, 16)

# Sample Standard Deviation (default in R)
sd(data)

# Variance
var(data)

# Population SD (manual)
sqrt(sum((data - mean(data))^2) / length(data))
```

---

## Interpretation Guide

| Scenario | Interpretation |
|----------|----------------|
| SD = 5, Mean = 50 | Typical value is within ±5 of 50. |
| SD = 0 | All values are identical. |
| SD very large relative to mean | High variability; data is spread out. |
| CV = SD/Mean = 0.1 | Low relative variability (10%). |

---

## Related Concepts

- [[stats/01_Foundations/Standard Error\|Standard Error]] - SD of a sampling distribution.
- [[stats/01_Foundations/Normal Distribution\|Normal Distribution]] - 68-95-99.7 rule uses SD.
- [[stats/01_Foundations/Variance\|Variance]]
- [[stats/01_Foundations/Coefficient of Variation\|Coefficient of Variation]]
- [[stats/01_Foundations/Z-Scores\|Z-Scores]]
