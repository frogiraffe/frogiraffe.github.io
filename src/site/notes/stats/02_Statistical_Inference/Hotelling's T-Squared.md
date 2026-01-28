---
{"dg-publish":true,"permalink":"/stats/02-statistical-inference/hotelling-s-t-squared/","tags":["Multivariate","Hypothesis-Testing","Statistics"]}
---


## Definition

> [!abstract] Core Statement
> **Hotelling's T²** is the multivariate generalization of the t-test, testing whether the ==mean vector of a multivariate sample== differs from a hypothesized value or between groups.

---

## One-Sample Test

$$
T^2 = n(\bar{\mathbf{x}} - \boldsymbol{\mu}_0)^T \mathbf{S}^{-1} (\bar{\mathbf{x}} - \boldsymbol{\mu}_0)
$$

Where:
- $\bar{\mathbf{x}}$ = sample mean vector
- $\boldsymbol{\mu}_0$ = hypothesized mean vector
- $\mathbf{S}$ = sample covariance matrix

---

## Conversion to F

$$
F = \frac{n - p}{p(n-1)} T^2 \sim F_{p, n-p}
$$

Where $p$ = number of variables.

---

## Python Implementation

```python
import numpy as np
from scipy import stats

# ========== ONE-SAMPLE TEST ==========
def hotellings_t2_one_sample(X, mu0):
    n, p = X.shape
    x_bar = X.mean(axis=0)
    S = np.cov(X.T)
    
    diff = x_bar - mu0
    T2 = n * diff @ np.linalg.inv(S) @ diff
    
    # Convert to F
    F = (n - p) / (p * (n - 1)) * T2
    p_value = 1 - stats.f.cdf(F, p, n - p)
    
    return T2, F, p_value

# Example
X = np.random.multivariate_normal([1, 2], [[1, 0.5], [0.5, 1]], 50)
mu0 = np.array([0, 0])
T2, F, p = hotellings_t2_one_sample(X, mu0)
print(f"T² = {T2:.3f}, F = {F:.3f}, p = {p:.4f}")

# ========== TWO-SAMPLE TEST ==========
def hotellings_t2_two_sample(X1, X2):
    n1, p = X1.shape
    n2 = X2.shape[0]
    
    x1_bar = X1.mean(axis=0)
    x2_bar = X2.mean(axis=0)
    
    S1 = np.cov(X1.T)
    S2 = np.cov(X2.T)
    S_pooled = ((n1-1)*S1 + (n2-1)*S2) / (n1 + n2 - 2)
    
    diff = x1_bar - x2_bar
    T2 = (n1*n2)/(n1+n2) * diff @ np.linalg.inv(S_pooled) @ diff
    
    F = (n1 + n2 - p - 1) / (p * (n1 + n2 - 2)) * T2
    p_value = 1 - stats.f.cdf(F, p, n1 + n2 - p - 1)
    
    return T2, F, p_value
```

---

## R Implementation

```r
library(DescTools)

# One-sample
HotellingsT2Test(X, mu = c(0, 0))

# Two-sample
HotellingsT2Test(X1, X2)
```

---

## Assumptions

- Multivariate normality
- Homogeneous covariance matrices (two-sample)
- Independent observations

---

## Related Concepts

- [[stats/02_Statistical_Inference/Student's T-Test\|Student's T-Test]] — Univariate version
- [[stats/02_Statistical_Inference/MANOVA\|MANOVA]] — Extension for multiple groups
- [[stats/01_Foundations/Multivariate Normal Distribution\|Multivariate Normal Distribution]] — Assumed distribution

---

## References

- **Book:** Johnson, R. A., & Wichern, D. W. (2007). *Applied Multivariate Statistical Analysis* (6th ed.). Pearson.
