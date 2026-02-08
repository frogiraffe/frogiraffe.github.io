---
{"dg-publish":true,"permalink":"/30-knowledge/stats/01-foundations/log-transformation/","tags":["probability","foundations"]}
---


## Definition

> [!abstract] Core Statement
> **Log Transformation** applies the natural logarithm (or log base 10) to data to reduce **right skewness**, stabilize **variance**, and make **multiplicative relationships** additive.

$$
y' = \log(y) \quad \text{or} \quad y' = \log_{10}(y)
$$

**Intuition (ELI5):** Income data: Most people earn \$50K, some earn \$1M. On a regular scale, the millionaires squash everyone else. Log transformation spreads out the low values and compresses the high ones, making the distribution more even.

**Key Uses:**
- Make skewed data more **normal**
- Stabilize **variance** (reduce [[30_Knowledge/Stats/03_Regression_Analysis/Heteroscedasticity\|Heteroscedasticity]])
- Linearize **exponential** relationships

---

## When to Use

> [!success] Use Log Transformation When...
> - Data is **right-skewed** (long tail to the right).
> - Variance **increases with the mean** (fan-shaped residuals).
> - The relationship is **multiplicative** (e.g., % growth).
> - Data spans **multiple orders of magnitude** (e.g., 10 to 10,000,000).
> - Data represents **ratios, rates, or counts**.

> [!failure] Do NOT Use Log Transformation When...
> - Data is **already symmetric** or left-skewed.
> - Data contains **zeros or negatives** — log(0) = undefined.
> - There's **no theoretical reason** for the transformation.
> - Interpretability is critical — log scale is harder to explain.

---

## Theoretical Background

### Why Log Works for Right-Skewed Data

The logarithm is a **concave function**:
- Compresses large values more than small values
- Spreads out small values
- Result: Right tail gets pulled in, left side spreads out

### Handling Zeros

| Method | When to Use |
|--------|-------------|
| $\log(y + 1)$ | Count data with zeros |
| $\log(y + c)$ | Add small constant $c$ |
| Use different transformation | If many zeros, consider [[30_Knowledge/Stats/01_Foundations/Box-Cox Transformation\|Box-Cox Transformation]] |

### Interpretation of Log-Transformed Variables

| Model | Interpretation of β |
|-------|---------------------|
| $\log(Y) = \beta X$ | 1 unit ↑ in X → $e^\beta$ multiplicative change in Y |
| $Y = \beta \log(X)$ | 1% ↑ in X → $\frac{\beta}{100}$ unit change in Y |
| $\log(Y) = \beta \log(X)$ | 1% ↑ in X → β% change in Y (**elasticity**) |

---

## Implementation

### Python

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Right-skewed data (e.g., income)
np.random.seed(42)
income = np.random.lognormal(mean=10.5, sigma=1.0, size=1000)

# ========== VISUALIZE BEFORE/AFTER ==========
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(income, bins=50, edgecolor='black')
axes[0].set_title(f'Original (Skewness: {stats.skew(income):.2f})')
axes[0].set_xlabel('Income')

log_income = np.log(income)
axes[1].hist(log_income, bins=50, edgecolor='black')
axes[1].set_title(f'Log-Transformed (Skewness: {stats.skew(log_income):.2f})')
axes[1].set_xlabel('Log(Income)')

plt.tight_layout()
plt.show()

# ========== NORMALITY TEST ==========
print("Shapiro-Wilk Test (p > 0.05 = normal):")
print(f"  Original: p = {stats.shapiro(income[:5000])[1]:.6f}")
print(f"  Log-transformed: p = {stats.shapiro(log_income[:5000])[1]:.4f}")

# ========== HANDLING ZEROS ==========
data_with_zeros = np.array([0, 1, 5, 10, 100, 1000])
log_data = np.log1p(data_with_zeros)  # log(1 + x), handles zeros
print(f"\nlog1p([0, 1, 5, ...]): {log_data}")
```

### R

```r
# Right-skewed data
set.seed(42)
income <- rlnorm(1000, meanlog = 10.5, sdlog = 1.0)

# ========== VISUALIZE ==========
par(mfrow = c(1, 2))
hist(income, main = paste("Original\nSkewness:", round(e1071::skewness(income), 2)),
     xlab = "Income", breaks = 50)
hist(log(income), main = paste("Log-Transformed\nSkewness:", 
                                round(e1071::skewness(log(income)), 2)),
     xlab = "Log(Income)", breaks = 50)

# ========== NORMALITY TEST ==========
shapiro.test(income[1:5000])
shapiro.test(log(income)[1:5000])

# ========== HANDLING ZEROS ==========
data_with_zeros <- c(0, 1, 5, 10, 100, 1000)
log1p(data_with_zeros)  # log(1 + x)
```

---

## Interpretation Guide

| Scenario | Effect of Log |
|----------|---------------|
| Original: [1, 10, 100, 1000] | Log: [0, 1, 2, 3] — Equal spacing |
| High skewness (>2) | Usually becomes near-normal |
| Fan-shaped residuals | Stabilizes variance |
| β = 0.05 in log(Y) ~ X | 1 unit ↑ in X → 5.1% ↑ in Y |

### Back-Transformation

To interpret in original units:
$$
\text{Median}(Y) = e^{\bar{\log(Y)}}
$$

**Warning:** $e^{\bar{\log(Y)}} \neq \bar{Y}$ (geometric mean ≠ arithmetic mean)

---

## Common Pitfalls

> [!warning] Traps to Avoid
>
> **1. Applying Log to Data with Zeros**
> - log(0) = -∞, undefined
> - Solution: Use log(y+1) or log1p()
>
> **2. Wrong Back-Transformation**
> - Mean of log values ≠ log of mean
> - For predictions, exponentiate, then add adjustment for bias
>
> **3. Over-Transforming**
> - If data is already normal, log makes it left-skewed
> - Solution: Check skewness before transforming
>
> **4. Forgetting Interpretation Changes**
> - Coefficients in log-linear model have multiplicative interpretation
> - Document that you're using log scale

---

## Worked Example

> [!example] Housing Prices — Before and After Log
> **Problem:** Model house prices. Distribution is heavily right-skewed.
>
> **Original Data:**
> - Mean: \$450,000, Median: \$320,000
> - Skewness: 2.8
> - Residuals: Fan-shaped (heteroscedastic)
>
> **After Log Transformation:**
> - Log(Mean): 12.8, Log(Median): 12.7
> - Skewness: 0.2
> - Residuals: Homoscedastic
>
> **Regression Result:**
> $$\log(\text{Price}) = 10.5 + 0.05 \times \text{Bedrooms}$$
>
> **Interpretation:**
> Each additional bedroom increases price by $e^{0.05} - 1 = 5.1\%$

---

## Related Concepts

- [[30_Knowledge/Stats/01_Foundations/Box-Cox Transformation\|Box-Cox Transformation]] — Generalized power transformation
- [[30_Knowledge/Stats/03_Regression_Analysis/Heteroscedasticity\|Heteroscedasticity]] — Variance stabilization
- [[30_Knowledge/Stats/01_Foundations/Normal Distribution\|Normal Distribution]] — Target for many tests
- [[30_Knowledge/Stats/01_Foundations/Skewness\|Skewness]] — Measure of asymmetry

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Assumptions are violated
> - Alternative methods are more appropriate

---

## References

- **Book:** Tukey, J. W. (1977). *Exploratory Data Analysis*. Addison-Wesley. [Pearson Link](https://www.pearson.com/us/higher-education/program/Tukey-Exploratory-Data-Analysis/PGM159512.html)
- **Book:** Fox, J. (2016). *Applied Regression Analysis and Generalized Linear Models* (3rd ed.). Sage. [SAGE Link](https://us.sagepub.com/en-us/nam/applied-regression-analysis-and-generalized-linear-models/book237254)
- **Article:** Manning, W. G. (1998). The logged dependent variable, heteroscedasticity, and the retransformation problem. *Journal of Health Economics*, 17(3), 283-295. [ScienceDirect](https://doi.org/10.1016/S0167-6296(98)00025-3)
