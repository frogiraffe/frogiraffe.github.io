---
{"dg-publish":true,"permalink":"/30-knowledge/stats/01-foundations/normal-distribution/","tags":["probability","foundations"]}
---

## Definition

> [!abstract] Core Statement
> The **Normal Distribution** (Gaussian Distribution) is a continuous probability distribution that is ==symmetric about the mean==, with the property that data points near the mean are more frequent than those far from the mean. It is fully characterized by two parameters: **mean ($\mu$)** and **standard deviation ($\sigma$)**.

![Normal Distribution showing PDF for different parameters|500](https://upload.wikimedia.org/wikipedia/commons/7/74/Normal_Distribution_PDF.svg)
*Figure 1: Normal distribution PDF for various combinations of μ and σ. Notice how μ shifts the center and σ controls the spread.*

---

> [!tip] Intuition (ELI5): The Crowd's Height
> If you measure the height of 1,000 strangers, most will be "average" (the middle of the bell). A few will be very tall and a few will be very short (the tails). The Normal Distribution is the "default" shape nature often picks for things that vary randomly around an average.

---

## Purpose

The Normal Distribution is central to statistics because:
1.  Many natural phenomena approximate normality (height, IQ, measurement error).
2.  The [[30_Knowledge/Stats/01_Foundations/Central Limit Theorem (CLT)\|Central Limit Theorem (CLT)]] guarantees the sampling distribution of means is normal.
3.  Most parametric tests (T-tests, ANOVA, Regression) assume normality of residuals.

---

## When to Use

> [!success] Assume Normality When...
> - Modeling continuous variables known to be approximately normal
> - Working with sample means of large samples (via CLT)
> - Residuals of a regression model have been verified to be normal

---

## When NOT to Use

> [!danger] Do NOT Assume Normality When...
> - **Data is discrete:** Use [[30_Knowledge/Stats/01_Foundations/Binomial Distribution\|Binomial Distribution]], [[30_Knowledge/Stats/01_Foundations/Poisson Distribution\|Poisson Distribution]], etc.
> - **Data is bounded:** E.g., percentages (0-100) often follow [[30_Knowledge/Stats/01_Foundations/Beta Distribution\|Beta Distribution]].
> - **Data has heavy tails:** Extreme outliers suggest [[30_Knowledge/Stats/01_Foundations/T-Distribution\|T-Distribution]] or robust methods.
> - **Data is skewed:** Consider [[30_Knowledge/Stats/01_Foundations/Log Transformation\|Log Transformation]] or [[30_Knowledge/Stats/01_Foundations/Box-Cox Transformation\|Box-Cox Transformation]].
> - **Small samples without verification:** Always test normality first.

---

## Theoretical Background

### Notation

$$
X \sim N(\mu, \sigma^2)
$$

where:
- $\mu$ = mean (location parameter)
- $\sigma^2$ = variance (scale parameter)
- $\sigma$ = standard deviation

### Probability Density Function (PDF)

$$
f(x | \mu, \sigma) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2\right)
$$

**Understanding the Formula:**
- $\frac{1}{\sigma\sqrt{2\pi}}$: Normalization constant (ensures area = 1)
- $\exp(-\frac{1}{2}z^2)$: The bell shape, where $z = \frac{x-\mu}{\sigma}$
- Maximum value occurs at $x = \mu$

### Properties

| Property | Value |
|----------|-------|
| Mean | $\mu$ |
| Median | $\mu$ |
| Mode | $\mu$ |
| Variance | $\sigma^2$ |
| Skewness | 0 (perfectly symmetric) |
| Kurtosis | 3 (mesokurtic) |

### The Empirical Rule (68-95-99.7)

> [!important] Memorize This
> - **68.27%** of data falls within $\mu \pm 1\sigma$
> - **95.45%** of data falls within $\mu \pm 2\sigma$
> - **99.73%** of data falls within $\mu \pm 3\sigma$

### The Standard Normal Distribution ($Z$)

Any normal variable $X \sim N(\mu, \sigma^2)$ can be **standardized** to $Z \sim N(0, 1)$:

$$
Z = \frac{X - \mu}{\sigma}
$$

**Interpretation:** $Z$ represents how many standard deviations $X$ is away from the mean.

---

## Worked Example: IQ Scores

> [!example] Problem
> IQ scores are normally distributed with $\mu = 100$ and $\sigma = 15$.
> 
> **Questions:**
> 1. What proportion of people have IQ < 115?
> 2. What IQ score is at the 90th percentile?

**Solution:**

**1. P(X < 115):**
$$ Z = \frac{115 - 100}{15} = 1.0 $$
Using Z-table: $P(Z < 1.0) = 0.8413$
**Result:** ~84.13% of people have IQ below 115.

**2. 90th Percentile:**
From Z-table: $Z_{0.90} = 1.28$
$$ X = \mu + Z \cdot \sigma = 100 + 1.28 \times 15 = 119.2 $$
**Result:** IQ of ~119 is at the 90th percentile.

**Verification with Code:**
```python
from scipy import stats

mu, sigma = 100, 15

# P(X < 115)
prob = stats.norm.cdf(115, loc=mu, scale=sigma)
print(f"P(X < 115): {prob:.4f}")  # 0.8413

# 90th percentile
x_90 = stats.norm.ppf(0.90, loc=mu, scale=sigma)
print(f"90th percentile: {x_90:.2f}")  # 119.22
```

---

## Assumptions

- [ ] **Continuous Data:** The variable must be measured on a continuous (interval or ratio) scale.
  - *Example:* Height in cm ✓ vs Number of children ✗
  
- [ ] **Independence:** Observations are independent.
  - *Example:* Random sample ✓ vs Time series data ✗
  
- [ ] **Unbounded Range:** Theoretically, normal data can range from $-\infty$ to $+\infty$.
  - *Example:* Temperature ✓ vs Percentage (0-100) ✗

---

## Limitations

> [!warning] Pitfalls
> 1.  **Normality is often assumed, not verified.** Always run diagnostics.
> 2.  **Sensitivity to Outliers:** Outliers can make data appear non-normal.
> 3.  **Misuse with Proportions/Counts:** Never use normal distribution for inherently non-normal data types.

---

## Assessing Normality

### Visual Methods
1.  **Histogram:** Should appear symmetric and bell-shaped.
2.  **[[30_Knowledge/Stats/09_EDA_and_Visualization/Q-Q Plot\|Q-Q Plot]]:** Points should fall on the 45-degree diagonal reference line.

### Statistical Tests
| Test | Best For | Null Hypothesis |
|------|----------|-----------------|
| [[30_Knowledge/Stats/02_Statistical_Inference/Shapiro-Wilk Test\|Shapiro-Wilk Test]] | $n < 50$ | Data is Normal |
| Kolmogorov-Smirnov | $n > 50$ | Data is Normal |
| D'Agostino-Pearson | Based on Skew/Kurtosis | Data is Normal |
| Anderson-Darling | General, weighted tails | Data is Normal |
| [[30_Knowledge/Stats/02_Statistical_Inference/Hotelling's T-Squared\|Hotelling's T-Squared]] | Multivariate normality | Mean vector = μ₀ |

---

## Python Implementation

```python
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

# --- Properties ---
mu, sigma = 100, 15  # IQ example

# P(X < 115) = ?
prob = stats.norm.cdf(115, loc=mu, scale=sigma)
print(f"P(X < 115): {prob:.4f}")

# What value is at the 90th percentile?
x_90 = stats.norm.ppf(0.90, loc=mu, scale=sigma)
print(f"90th Percentile: {x_90:.2f}")

# --- Generate and Test ---
data = np.random.normal(mu, sigma, 200)

# Shapiro-Wilk
stat, p = stats.shapiro(data)
print(f"Shapiro-Wilk p-value: {p:.4f}")

# Visualize
x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
plt.plot(x, stats.norm.pdf(x, mu, sigma), 'b-', lw=2)
plt.fill_between(x, stats.norm.pdf(x, mu, sigma), alpha=0.3)
plt.xlabel('X')
plt.ylabel('Density')
plt.title(f'Normal Distribution (μ={mu}, σ={sigma})')
plt.grid(alpha=0.3)
plt.show()
```

**Expected Output:**
```
P(X < 115): 0.8413
90th Percentile: 119.22
Shapiro-Wilk p-value: ~0.5-0.9 (varies)
```

---

## R Implementation

```r
# --- Properties ---
mu <- 100
sigma <- 15

# P(X < 115)
pnorm(115, mean = mu, sd = sigma)

# 90th Percentile
qnorm(0.90, mean = mu, sd = sigma)

# --- Generate and Test ---
data <- rnorm(200, mean = mu, sd = sigma)

# Shapiro-Wilk Test
shapiro.test(data)

# Q-Q Plot
qqnorm(data, main = "Q-Q Plot for Normality")
qqline(data, col = "red", lwd = 2)
```

---

## Interpretation Guide

| Scenario | Interpretation |
|----------|----------------|
| $Z = 0$ | Observation is exactly at the mean. |
| $Z = 2$ | Observation is 2 std. deviations above the mean (top ~2.5%). |
| $Z = -1.5$ | Observation is 1.5 std. deviations below the mean. |
| Shapiro $p > 0.05$ | Fail to reject $H_0$; data is consistent with normality. |
| Q-Q plot curves at tails | Data has heavier/lighter tails than normal (kurtosis issue). |

---

## Related Concepts

### Directly Related
- [[30_Knowledge/Stats/01_Foundations/Central Limit Theorem (CLT)\|Central Limit Theorem (CLT)]] - Why sample means become normal
- [[30_Knowledge/Stats/01_Foundations/T-Distribution\|T-Distribution]] - Normal's cousin for small samples
- [[30_Knowledge/Stats/01_Foundations/Z-Scores\|Z-Scores]] - Standardized normal values
- [[30_Knowledge/Stats/01_Foundations/Standard Deviation\|Standard Deviation]] - The σ parameter

### Diagnostics
- [[30_Knowledge/Stats/09_EDA_and_Visualization/Q-Q Plot\|Q-Q Plot]] - Visual diagnostic
- [[30_Knowledge/Stats/02_Statistical_Inference/Shapiro-Wilk Test\|Shapiro-Wilk Test]] - Statistical diagnostic
- [[30_Knowledge/Stats/01_Foundations/Log Transformation\|Log Transformation]] - A cure for non-normality

### Other Distributions
- [[30_Knowledge/Stats/01_Foundations/Binomial Distribution\|Binomial Distribution]] - Approximated by Normal for large n
- [[30_Knowledge/Stats/01_Foundations/Chebyshev's Inequality\|Chebyshev's Inequality]] - Distribution-free alternative to 68-95-99.7 rule

### Other Related Topics

{ .block-language-dataview}

---

## References

1. Gauss, C. F. (1809). *Theoria Motus Corporum Coelestium*. [Available online](https://books.google.com/books?id=f_8_AAAAYAAJ)

2. Rice, J. A. (2007). *Mathematical Statistics and Data Analysis* (3rd ed.). Duxbury. Chapter 3: Random Variables. [Available online](https://www.cengage.com/c/mathematical-statistics-and-data-analysis-3e-rice/)

3. DeGroot, M. H., & Schervish, M. J. (2012). *Probability and Statistics* (4th ed.). Pearson. Chapter 5: Special Distributions. [Available online](https://www.pearson.com/en-us/subject-catalog/p/probability-and-statistics/P200000006277/)

### Additional Resources
- [Normal Distribution Interactive Visualization](https://seeing-theory.brown.edu/probability-distributions/index.html#section2)
- [Khan Academy: Normal Distribution](https://www.khanacademy.org/math/statistics-probability/modeling-distributions-of-data/normal-distributions-library)
