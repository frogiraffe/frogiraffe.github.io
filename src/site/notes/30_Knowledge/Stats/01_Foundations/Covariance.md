---
{"dg-publish":true,"permalink":"/30-knowledge/stats/01-foundations/covariance/","tags":["probability","foundations"]}
---

## Definition

> [!abstract] Core Statement
> **Covariance** measures the ==joint variability== of two random variables. Positive covariance indicates they move together; negative means they move oppositely.

![Covariance showing positive and negative relationships|500](https://upload.wikimedia.org/wikipedia/commons/a/a0/Covariance_trends.svg)
*Figure 1: Positive covariance (left) vs negative covariance (right)*

$$
\text{Cov}(X, Y) = E[(X - \mu_X)(Y - \mu_Y)] = E[XY] - E[X]E[Y]
$$

**Sample Covariance:**
$$
s_{xy} = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})
$$

---

> [!tip] Intuition (ELI5): The Dance Partners
> Imagine two dancers. If when one steps forward (above their average), the other also steps forward—they have **positive covariance**. If one steps forward while the other steps back, that's **negative covariance**. Zero covariance means their movements are unrelated.

---

## Purpose

1. Measure **linear relationship** between variables
2. Build [[30_Knowledge/Stats/01_Foundations/Covariance Matrix\|covariance matrices]] for multivariate analysis
3. Calculate [[30_Knowledge/Stats/01_Foundations/Correlation Analysis\|correlation]] (covariance standardized by SDs)
4. Understanding [[30_Knowledge/Stats/01_Foundations/Multivariate Normal Distribution\|MVN]] and [[30_Knowledge/Stats/05_Unsupervised_Learning/PCA (Principal Component Analysis)\|PCA]]

---

## When to Use

> [!success] Use Covariance When...
> - Building covariance/correlation matrices
> - Calculating portfolio variance (finance)
> - Understanding variable relationships before correlation

---

## When NOT to Use

> [!danger] Limitations
> - **Scale-dependent:** Hard to interpret magnitude (use correlation instead)
> - **Linear only:** Misses non-linear relationships
> - **Outlier sensitive:** Extreme values distort covariance

---

## Theoretical Background

### Properties

| Property | Formula |
|----------|---------|
| $\text{Cov}(X, X)$ | $\text{Var}(X)$ |
| $\text{Cov}(X, Y)$ | $\text{Cov}(Y, X)$ (symmetric) |
| $\text{Cov}(aX, Y)$ | $a \cdot \text{Cov}(X, Y)$ |
| $\text{Cov}(X + c, Y)$ | $\text{Cov}(X, Y)$ (adding constant doesn't change cov) |
| $\text{Var}(X + Y)$ | $\text{Var}(X) + \text{Var}(Y) + 2\text{Cov}(X,Y)$ |

### Covariance vs Correlation

| Measure | Range | Interpretation |
|---------|-------|----------------|
| **Covariance** | $(-\infty, +\infty)$ | Scale-dependent |
| **Correlation** | $[-1, +1]$ | Standardized, interpretable |

$$
r = \frac{\text{Cov}(X, Y)}{\sigma_X \cdot \sigma_Y}
$$

---

## Worked Example

> [!example] Problem
> Heights (X) and weights (Y) of 5 people:
> - X: [160, 170, 175, 180, 190] cm
> - Y: [55, 65, 70, 72, 85] kg
> 
> Calculate the sample covariance.

**Solution:**

$\bar{x}$ = (160+170+175+180+190)/5 = 175
$\bar{y}$ = (55+65+70+72+85)/5 = 69.4

| $x_i$ | $y_i$ | $(x_i - \bar{x})$ | $(y_i - \bar{y})$ | Product |
|-------|-------|-------------------|-------------------|---------|
| 160 | 55 | -15 | -14.4 | 216 |
| 170 | 65 | -5 | -4.4 | 22 |
| 175 | 70 | 0 | 0.6 | 0 |
| 180 | 72 | 5 | 2.6 | 13 |
| 190 | 85 | 15 | 15.6 | 234 |

$$s_{xy} = \frac{216 + 22 + 0 + 13 + 234}{5-1} = \frac{485}{4} = 121.25$$

**Interpretation:** Positive covariance → taller people tend to weigh more.

**Verification:**
```python
import numpy as np

x = np.array([160, 170, 175, 180, 190])
y = np.array([55, 65, 70, 72, 85])

cov = np.cov(x, y)[0, 1]
print(f"Covariance: {cov:.2f}")  # 121.25
```

---

## Python Implementation

```python
import numpy as np
import pandas as pd

x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

# Sample covariance (np.cov returns matrix)
cov = np.cov(x, y)[0, 1]
print(f"Covariance: {cov:.3f}")

# Covariance matrix
cov_matrix = np.cov(x, y)
print("Covariance Matrix:")
print(cov_matrix)

# Using pandas
df = pd.DataFrame({'X': x, 'Y': y})
print(df.cov())
```

**Expected Output:**
```
Covariance: 1.500
Covariance Matrix:
[[2.5 1.5]
 [1.5 1.3]]
```

---

## R Implementation

```r
x <- c(1, 2, 3, 4, 5)
y <- c(2, 4, 5, 4, 5)

# Sample covariance
cov(x, y)

# Covariance matrix
cov(cbind(x, y))
```

---

## Related Concepts

### Directly Related
- [[30_Knowledge/Stats/01_Foundations/Correlation Analysis\|Correlation Analysis]] - Standardized version
- [[30_Knowledge/Stats/01_Foundations/Variance\|Variance]] - Covariance with itself
- [[30_Knowledge/Stats/01_Foundations/Covariance Matrix\|Covariance Matrix]] - Multiple variables

### Applications
- [[30_Knowledge/Stats/01_Foundations/Multivariate Normal Distribution\|Multivariate Normal Distribution]] - Uses covariance matrix
- [[30_Knowledge/Stats/05_Unsupervised_Learning/PCA (Principal Component Analysis)\|Principal Component Analysis (PCA)]] - Eigendecomposition of covariance
- Portfolio Theory - Covariance between asset returns

### Other Related Topics

{ .block-language-dataview}

---

## References

1. Casella, G., & Berger, R. L. (2002). *Statistical Inference* (2nd ed.). Duxbury. [Available online](https://www.routledge.com/Statistical-Inference/Casella-Berger/p/book/9781032593036)

2. Johnson, R. A., & Wichern, D. W. (2007). *Applied Multivariate Statistical Analysis* (6th ed.). Pearson.
