---
{"dg-publish":true,"permalink":"/stats/01-foundations/mean/","tags":["probability","descriptive-statistics","central-tendency","foundations"]}
---


## Definition

> [!abstract] Core Statement
> The **Mean** (arithmetic mean) is the ==sum of all values divided by the count==. It's the most common measure of central tendency but is sensitive to outliers.

$$
\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i
$$

---

## When NOT to Use

> [!danger] Do NOT Use Mean When...
> - **Skewed data:** Use median (income, housing prices)
> - **Outliers present:** One extreme value distorts the mean
> - **Ordinal data:** Mean of ratings may be meaningless

---

## Types of Means

| Type | Formula | Use Case |
|------|---------|----------|
| **Arithmetic** | $\frac{\sum x}{n}$ | General average |
| **Geometric** | $\sqrt[n]{\prod x}$ | Growth rates, ratios |
| **Harmonic** | $\frac{n}{\sum \frac{1}{x}}$ | Rates (speed, returns) |
| **Weighted** | $\frac{\sum w_i x_i}{\sum w_i}$ | When observations differ in importance |

---

## Python Implementation

```python
import numpy as np
from scipy import stats

data = np.array([10, 20, 30, 40, 100])

# Arithmetic mean
mean = np.mean(data)
print(f"Arithmetic Mean: {mean}")

# Geometric mean (all values must be positive)
geo_mean = stats.gmean(data)
print(f"Geometric Mean: {geo_mean}")

# Harmonic mean (values must be positive)
harm_mean = stats.hmean(data)
print(f"Harmonic Mean: {harm_mean}")

# Trimmed mean (robust to outliers)
trimmed = stats.trim_mean(data, proportiontocut=0.1)
print(f"Trimmed Mean (10%): {trimmed}")

# Weighted mean
weights = np.array([1, 1, 1, 1, 0.5])  # Downweight outlier
weighted = np.average(data, weights=weights)
print(f"Weighted Mean: {weighted}")
```

---

## R Implementation

```r
data <- c(10, 20, 30, 40, 100)

mean(data)              # Arithmetic mean
exp(mean(log(data)))    # Geometric mean
1/mean(1/data)          # Harmonic mean
mean(data, trim = 0.1)  # Trimmed mean
weighted.mean(data, w)  # Weighted mean
```

---

## Properties

| Property | Description |
|----------|-------------|
| **Uniqueness** | Only one mean exists for any dataset |
| **Sum of deviations = 0** | $\sum(x_i - \bar{x}) = 0$ |
| **Minimizes squared error** | Mean minimizes $\sum(x_i - c)^2$ |
| **Sensitive to outliers** | Single extreme value can shift mean significantly |

---

## Mean vs Median

| Scenario | Better Measure |
|----------|----------------|
| Symmetric, no outliers | Mean |
| Skewed distribution | Median |
| Income data | Median |
| Test scores (normal) | Mean |

> [!tip] Rule of Thumb
> If **Mean ≠ Median** significantly, the distribution is skewed.
> - Mean > Median → Right-skewed (positive skew)
> - Mean < Median → Left-skewed (negative skew)

---

## Related Concepts

- [[stats/01_Foundations/Median\|Median]] — Robust alternative
- [[stats/01_Foundations/Sample Variance\|Sample Variance]] — Spread around the mean
- [[stats/01_Foundations/Central Limit Theorem (CLT)\|Central Limit Theorem (CLT)]] — Means follow normal distribution
- [[stats/01_Foundations/Skewness\|Skewness]] — Mean vs median comparison

---

## References

- **Book:** Wackerly, D., Mendenhall, W., & Scheaffer, R. (2014). *Mathematical Statistics with Applications* (7th ed.).
