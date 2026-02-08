---
{"dg-publish":true,"permalink":"/30-knowledge/stats/01-foundations/skewness/","tags":["probability","foundations"]}
---


## Definition

> [!abstract] Core Statement
> **Skewness** measures the ==asymmetry== of a distribution. Positive skewness means a long right tail; negative skewness means a long left tail; zero indicates symmetry.

$$
\text{Skewness} = \frac{E[(X - \mu)^3]}{\sigma^3} = \frac{\frac{1}{n}\sum(x_i - \bar{x})^3}{s^3}
$$

---

## Visual Interpretation

```
Negative Skew          Symmetric           Positive Skew
(Left-skewed)                              (Right-skewed)

    ▄▄▄▄▄▄▄▄              ▄▄▄▄▄▄              ▄▄▄▄▄▄▄▄
  ▄▄▄▄▄▄▄▄▄▄▄▄▄        ▄▄▄▄▄▄▄▄▄▄▄▄        ▄▄▄▄▄▄▄▄▄▄▄▄▄
▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄    ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄    ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄

Median > Mean           Mean ≈ Median        Mean > Median
```

---

## Interpretation Guide

| Skewness | Description | Example |
|----------|-------------|---------|
| **< -1** | Highly left-skewed | Age at death |
| **-1 to -0.5** | Moderately left-skewed | — |
| **-0.5 to 0.5** | Approximately symmetric | Heights |
| **0.5 to 1** | Moderately right-skewed | — |
| **> 1** | Highly right-skewed | Income, wealth |

---

## Python Implementation

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# ========== CALCULATE SKEWNESS ==========
data_symmetric = np.random.normal(0, 1, 1000)
data_right_skew = np.random.exponential(1, 1000)
data_left_skew = -np.random.exponential(1, 1000)

print(f"Symmetric: Skewness = {stats.skew(data_symmetric):.3f}")
print(f"Right-skewed: Skewness = {stats.skew(data_right_skew):.3f}")
print(f"Left-skewed: Skewness = {stats.skew(data_left_skew):.3f}")

# ========== VISUALIZE ==========
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, data, title in zip(axes, 
                            [data_left_skew, data_symmetric, data_right_skew],
                            ['Left-Skewed', 'Symmetric', 'Right-Skewed']):
    ax.hist(data, bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(data), color='red', label=f'Mean: {np.mean(data):.2f}')
    ax.axvline(np.median(data), color='blue', linestyle='--', 
               label=f'Median: {np.median(data):.2f}')
    ax.set_title(f'{title}\nSkewness: {stats.skew(data):.2f}')
    ax.legend()

plt.tight_layout()
plt.show()

# ========== NORMALITY TEST ==========
stat, pvalue = stats.skewtest(data_right_skew)
print(f"D'Agostino skewness test p-value: {pvalue:.4f}")
```

---

## R Implementation

```r
library(e1071)

# Skewness
skewness(data)

# Or base R (moments package)
library(moments)
skewness(data)

# Visualization
hist(data, probability = TRUE)
lines(density(data), col = "red")
abline(v = mean(data), col = "blue", lwd = 2)
abline(v = median(data), col = "green", lwd = 2)
```

---

## Transformations for Skewness

| Skewness | Transformation |
|----------|----------------|
| **Right (+)** | Log, Square root, Box-Cox |
| **Left (-)** | Square, Cube, Reflect then log |

```python
import numpy as np

# Right-skewed data
original = np.random.exponential(1, 1000)
log_transformed = np.log1p(original)  # log(1+x) handles zeros
sqrt_transformed = np.sqrt(original)

print(f"Original skewness: {stats.skew(original):.3f}")
print(f"Log-transformed skewness: {stats.skew(log_transformed):.3f}")
print(f"Sqrt-transformed skewness: {stats.skew(sqrt_transformed):.3f}")
```

---

## Related Concepts

- [[30_Knowledge/Stats/01_Foundations/Mean\|Mean]] — Affected by skewness
- [[30_Knowledge/Stats/01_Foundations/Median\|Median]] — Robust to skewness
- [[30_Knowledge/Stats/01_Foundations/Log Transformation\|Log Transformation]] — Fixes right skew
- [[30_Knowledge/Stats/01_Foundations/Box-Cox Transformation\|Box-Cox Transformation]] — Optimal transformation
- [[30_Knowledge/Stats/01_Foundations/Normal Distribution\|Normal Distribution]] — Zero skewness

---

## When to Use

> [!success] Use Skewness When...
> - Refer to standard documentation
> - Refer to standard documentation

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Assumptions are violated
> - Alternative methods are more appropriate

---

## References

- **Book:** Joanes, D. N., & Gill, C. A. (1998). Comparing measures of sample skewness and kurtosis. *Journal of the Royal Statistical Society*, 47(1), 183-189.
