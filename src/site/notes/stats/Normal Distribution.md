---
{"dg-publish":true,"permalink":"/stats/normal-distribution/","tags":["Statistics","Probability-Theory","Distributions","Foundations"]}
---


# Normal Distribution

## Definition

> [!abstract] Core Statement
> The **Normal Distribution** (Gaussian Distribution) is a continuous probability distribution that is ==symmetric about the mean==, with the property that data points near the mean are more frequent than those far from the mean. It is fully characterized by two parameters: **mean ($\mu$)** and **standard deviation ($\sigma$)**.

---

## Purpose

The Normal Distribution is central to statistics because:
1.  Many natural phenomena approximate normality (height, IQ, measurement error).
2.  The [[stats/Central Limit Theorem (CLT)\|Central Limit Theorem (CLT)]] guarantees the sampling distribution of means is normal.
3.  Most parametric tests (T-tests, ANOVA, Regression) assume normality of residuals.

---

## When to Use

> [!success] Assume Normality When...
> - Modeling continuous variables known to be approximately normal.
> - Working with sample means of large samples (via CLT).
> - Residuals of a regression model have been verified to be normal.

> [!failure] Do NOT Assume Normality When...
> - Data is discrete (use Binomial, Poisson).
> - Data is bounded (e.g., percentages 0-100 often have Beta distribution).
> - Data has heavy tails or extreme outliers.

---

## Theoretical Background

### Probability Density Function (PDF)

$$
f(x | \mu, \sigma) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2\right)
$$

### Properties

| Property | Value |
|----------|-------|
| Mean | $\mu$ |
| Median | $\mu$ |
| Mode | $\mu$ |
| Variance | $\sigma^2$ |
| Skewness | 0 (Perfectly Symmetric) |
| Kurtosis | 3 (Mesokurtic) |

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

## Assumptions

- [ ] **Continuous Data:** The variable must be measured on a continuous (interval or ratio) scale.
- [ ] **Independence:** Observations are independent.
- [ ] **Unbounded Range:** Theoretically, normal data can range from $-\infty$ to $+\infty$. (Practically, bounded data may be approximately normal if not near bounds).

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
2.  **[[stats/Q-Q Plot\|Q-Q Plot]]:** Points should fall on the 45-degree diagonal reference line.

### Statistical Tests
| Test | Best For | Null Hypothesis |
|------|----------|-----------------|
| [[stats/Shapiro-Wilk Test\|Shapiro-Wilk Test]] | $n < 50$ | Data is Normal |
| Kolmogorov-Smirnov | $n > 50$ | Data is Normal |
| D'Agostino-Pearson | Based on Skew/Kurtosis | Data is Normal |

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

# Q-Q Plot
import statsmodels.api as sm
sm.qqplot(data, line='45')
plt.title("Q-Q Plot")
plt.show()
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
| $Z = 2$ | Observation is 2 std. deviations above the mean (Top ~2.5%). |
| $Z = -1.5$ | Observation is 1.5 std. deviations below the mean. |
| Shapiro $p > 0.05$ | Fail to reject $H_0$; data is consistent with normality. |
| Q-Q plot curves at tails | Data has heavier/lighter tails than normal (Kurtosis issue). |

---

## Related Concepts

- [[stats/Central Limit Theorem (CLT)\|Central Limit Theorem (CLT)]] - Why sample means become normal.
- [[stats/T-Distribution\|T-Distribution]] - Normal's cousin for small samples.
- [[stats/Q-Q Plot\|Q-Q Plot]] - Visual diagnostic.
- [[stats/Shapiro-Wilk Test\|Shapiro-Wilk Test]] - Statistical diagnostic.
- [[stats/Log Transformations\|Log Transformations]] - A cure for non-normality.
