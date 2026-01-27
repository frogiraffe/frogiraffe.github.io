---
{"dg-publish":true,"permalink":"/stats/01-foundations/variance/","tags":["Foundations","Descriptive-Statistics","Dispersion"]}
---


## Definition

> [!abstract] Core Statement
> **Variance** measures the **average squared deviation** from the mean. It quantifies how spread out data points are around the center.

**Population Variance:**
$$
\sigma^2 = \frac{1}{N}\sum_{i=1}^{N}(x_i - \mu)^2
$$

**Sample Variance:**
$$
s^2 = \frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2
$$

**Intuition (ELI5):** Imagine you and your friends mark where you're standing on a number line. Variance measures how scattered everyone is. If everyone clusters around the center, variance is small. If people spread across the whole line, variance is large.

---

## When to Use

> [!success] Use Variance When...
> - Measuring **spread/dispersion** of data.
> - Comparing **consistency** between groups.
> - Calculating [[stats/01_Foundations/Standard Deviation\|Standard Deviation]] ($\sigma = \sqrt{\sigma^2}$).
> - Building statistical models ([[stats/02_Hypothesis_Testing/One-Way ANOVA\|ANOVA]], [[stats/03_Regression_Analysis/Multiple Linear Regression\|Linear Regression]]).
> - Understanding [[stats/01_Foundations/Bias-Variance Trade-off\|Bias-Variance Trade-off]] in ML.

> [!failure] Variance Limitations
> - Units are **squared** (hard to interpret — use SD instead).
> - Sensitive to **outliers** (squared deviations amplify extremes).
> - Not meaningful for **non-numeric** data.

---

## Theoretical Background

### Why Square Deviations?

| Method | Problem |
|--------|---------|
| Sum of deviations: $\sum(x_i - \mu)$ | Always equals 0 |
| Sum of absolute deviations: $\sum|x_i - \mu|$ | Not differentiable at 0 |
| Sum of squared deviations: $\sum(x_i - \mu)^2$ | ✓ Always positive, mathematically tractable |

### Why n-1 for Sample Variance? (Bessel's Correction)

The sample mean $\bar{x}$ is estimated from the data, so we lose one degree of freedom. Without dividing by $n-1$:
- Sample variance **underestimates** population variance
- $n-1$ provides an **unbiased estimator**

### Variance Properties

| Property | Formula |
|----------|---------|
| Non-negative | $\text{Var}(X) \geq 0$ |
| Constant shift | $\text{Var}(X + c) = \text{Var}(X)$ |
| Scaling | $\text{Var}(cX) = c^2 \cdot \text{Var}(X)$ |
| Sum (independent) | $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)$ |

---

## Implementation

### Python

```python
import numpy as np
from scipy import stats

data = [4, 8, 6, 5, 3, 2, 8, 9, 2, 5]

# ========== POPULATION VARIANCE ==========
var_pop = np.var(data, ddof=0)
print(f"Population Variance (σ²): {var_pop:.2f}")

# ========== SAMPLE VARIANCE ==========
var_sample = np.var(data, ddof=1)
print(f"Sample Variance (s²): {var_sample:.2f}")

# ========== STANDARD DEVIATION ==========
std_sample = np.std(data, ddof=1)
print(f"Sample Std Dev (s): {std_sample:.2f}")

# ========== MANUAL CALCULATION ==========
mean = np.mean(data)
squared_devs = [(x - mean)**2 for x in data]
manual_var = sum(squared_devs) / (len(data) - 1)
print(f"Manual Sample Variance: {manual_var:.2f}")
```

### R

```r
data <- c(4, 8, 6, 5, 3, 2, 8, 9, 2, 5)

# ========== SAMPLE VARIANCE (default) ==========
var_sample <- var(data)
cat("Sample Variance (s²):", var_sample, "\n")

# ========== POPULATION VARIANCE ==========
n <- length(data)
var_pop <- var(data) * (n - 1) / n
cat("Population Variance (σ²):", var_pop, "\n")

# ========== STANDARD DEVIATION ==========
sd_sample <- sd(data)
cat("Sample Std Dev (s):", sd_sample, "\n")
```

---

## Interpretation Guide

| Variance | Interpretation |
|----------|----------------|
| $\sigma^2 = 0$ | All values are identical |
| Low variance | Data tightly clustered around mean |
| High variance | Data widely spread |
| Variance >> Mean² | Extreme spread relative to magnitude |

### Comparing Datasets

| Dataset A | Dataset B |
|-----------|-----------|
| [5, 5, 5, 5, 5] | [1, 3, 5, 7, 9] |
| Mean = 5 | Mean = 5 |
| Variance = 0 | Variance = 10 |
| → No spread | → High spread |

---

## Common Pitfalls

> [!warning] Traps to Avoid
>
> **1. Forgetting ddof for Samples**
> - Default in numpy is `ddof=0` (population)
> - For samples, use `ddof=1`
>
> **2. Interpreting Squared Units**
> - If data is in meters, variance is in meters²
> - Use standard deviation for interpretable units
>
> **3. Ignoring Outlier Sensitivity**
> - One extreme value drastically inflates variance
> - Consider IQR or MAD for robust alternatives

---

## Worked Example

> [!example] Calculating Variance Step by Step
> **Data:** [2, 4, 4, 4, 5, 5, 7, 9]
>
> **Step 1: Calculate Mean**
> $$\bar{x} = \frac{2+4+4+4+5+5+7+9}{8} = 5$$
>
> **Step 2: Calculate Squared Deviations**
> | $x_i$ | $x_i - \bar{x}$ | $(x_i - \bar{x})^2$ |
> |-------|-----------------|---------------------|
> | 2 | -3 | 9 |
> | 4 | -1 | 1 |
> | 4 | -1 | 1 |
> | 4 | -1 | 1 |
> | 5 | 0 | 0 |
> | 5 | 0 | 0 |
> | 7 | 2 | 4 |
> | 9 | 4 | 16 |
>
> **Step 3: Sum Squared Deviations**
> $$\sum(x_i - \bar{x})^2 = 9+1+1+1+0+0+4+16 = 32$$
>
> **Step 4: Divide by n-1 (sample variance)**
> $$s^2 = \frac{32}{8-1} = \frac{32}{7} = 4.57$$
>
> **Step 5: Standard Deviation**
> $$s = \sqrt{4.57} = 2.14$$

---

## Related Concepts

- [[stats/01_Foundations/Standard Deviation\|Standard Deviation]] — Square root of variance
- [[stats/01_Foundations/Covariance Matrix\|Covariance]] — Variance between two variables
- [[stats/01_Foundations/Coefficient of Variation\|Coefficient of Variation]] — Variance relative to mean
- [[stats/01_Foundations/Bias-Variance Trade-off\|Bias-Variance Trade-off]] — Variance in ML context

---

## References

- **Book:** Wackerly, D., Mendenhall, W., & Scheaffer, R. L. (2008). *Mathematical Statistics with Applications* (7th ed.). Thomson Brooks/Cole. [Cengage](https://www.cengage.com/c/mathematical-statistics-with-applications-7e-wackerly/9780495110811/)
- **Book:** Freedman, D., Pisani, R., & Purves, R. (2007). *Statistics* (4th ed.). W. W. Norton & Company. [W.W. Norton](https://wwnorton.com/books/9780393929720)
- **Book:** Casella, G., & Berger, R. L. (2002). *Statistical Inference* (2nd ed.). Duxbury. [Cengage](https://www.cengage.com/c/statistical-inference-2e-casella/9780534243128/)
