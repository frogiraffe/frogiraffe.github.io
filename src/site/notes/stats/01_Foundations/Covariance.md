---
{"dg-publish":true,"permalink":"/stats/01-foundations/covariance/","tags":["Statistics","Descriptive-Statistics","Association"]}
---


## Definition

> [!abstract] Core Statement
> **Covariance** measures the ==joint variability== of two random variables. Positive covariance indicates they move together; negative means they move oppositely.

$$
\text{Cov}(X, Y) = E[(X - \mu_X)(Y - \mu_Y)] = E[XY] - E[X]E[Y]
$$

**Sample Covariance:**
$$
s_{xy} = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})
$$

---

## Properties

| Property | Formula |
|----------|---------|
| $\text{Cov}(X, X)$ | $\text{Var}(X)$ |
| $\text{Cov}(X, Y)$ | $\text{Cov}(Y, X)$ |
| $\text{Cov}(aX, Y)$ | $a \cdot \text{Cov}(X, Y)$ |
| $\text{Var}(X + Y)$ | $\text{Var}(X) + \text{Var}(Y) + 2\text{Cov}(X,Y)$ |

---

## Python Implementation

```python
import numpy as np

x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

# Sample covariance
cov = np.cov(x, y)[0, 1]
print(f"Covariance: {cov:.3f}")

# Covariance matrix
cov_matrix = np.cov(x, y)
print(cov_matrix)
```

---

## Covariance vs Correlation

| Measure | Range | Interpretation |
|---------|-------|----------------|
| **Covariance** | $(-\infty, +\infty)$ | Scale-dependent |
| **Correlation** | $[-1, +1]$ | Standardized |

$$
r = \frac{\text{Cov}(X, Y)}{\sigma_X \cdot \sigma_Y}
$$

---

## Related Concepts

- [[stats/01_Foundations/Correlation Analysis\|Correlation Analysis]] — Standardized version
- [[stats/01_Foundations/Sample Variance\|Sample Variance]] — Covariance with itself
- [[stats/01_Foundations/Multivariate Normal Distribution\|Multivariate Normal Distribution]] — Covariance matrix

---

## References

- **Book:** Casella, G., & Berger, R. L. (2002). *Statistical Inference*. Cengage.
