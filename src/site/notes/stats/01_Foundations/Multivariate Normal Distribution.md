---
{"dg-publish":true,"permalink":"/stats/01-foundations/multivariate-normal-distribution/","tags":["Probability","Distributions","Multivariate"]}
---


## Definition

> [!abstract] Core Statement
> The **Multivariate Normal Distribution** generalizes the normal distribution to ==multiple correlated variables==, characterized by a mean vector and covariance matrix.

$$
f(\mathbf{x}) = \frac{1}{\sqrt{(2\pi)^k|\boldsymbol{\Sigma}|}} \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)
$$

---

## Properties

| Property | Description |
|----------|-------------|
| **Marginals** | Each variable is univariate normal |
| **Conditionals** | Conditionals are also normal |
| **Linear combinations** | Any linear combo is normal |
| **Mahalanobis distance** | $(\mathbf{x}-\boldsymbol{\mu})^T\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})$ |

---

## Python Implementation

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# ========== DEFINE DISTRIBUTION ==========
mean = np.array([0, 0])
cov = np.array([[1, 0.8],
                [0.8, 1]])  # Strong positive correlation

mvn = stats.multivariate_normal(mean, cov)

# ========== SAMPLING ==========
samples = mvn.rvs(1000)
plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5)
plt.xlabel('X1'); plt.ylabel('X2')
plt.title('Bivariate Normal Samples')
plt.show()

# ========== PDF ==========
x = np.array([1, 1])
prob_density = mvn.pdf(x)
print(f"PDF at {x}: {prob_density:.4f}")

# ========== CONTOUR PLOT ==========
x, y = np.mgrid[-3:3:.01, -3:3:.01]
pos = np.dstack((x, y))
plt.contourf(x, y, mvn.pdf(pos))
plt.colorbar()
plt.title('Bivariate Normal PDF')
plt.show()
```

---

## R Implementation

```r
library(mvtnorm)

mean <- c(0, 0)
sigma <- matrix(c(1, 0.8, 0.8, 1), nrow = 2)

# Sampling
samples <- rmvnorm(1000, mean, sigma)

# PDF
dmvnorm(c(1, 1), mean, sigma)
```

---

## Conditional Distribution

If $\mathbf{X} = (X_1, X_2)^T \sim N(\boldsymbol{\mu}, \boldsymbol{\Sigma})$:

$$
X_1 | X_2 = x_2 \sim N\left(\mu_1 + \frac{\sigma_{12}}{\sigma_{22}}(x_2 - \mu_2), \sigma_{11} - \frac{\sigma_{12}^2}{\sigma_{22}}\right)
$$

---

## Applications

| Application | Use |
|-------------|-----|
| **PCA** | Data follows MVN → principal components normal |
| **Gaussian Processes** | Prior/posterior are MVN |
| **Linear Regression** | Errors assumed MVN |
| **LDA** | Class-conditional densities |

---

## Related Concepts

- [[stats/01_Foundations/Normal Distribution\|Normal Distribution]] — Univariate case
- [[stats/01_Foundations/Covariance\|Covariance]] — Off-diagonal of Σ
- [[stats/02_Statistical_Inference/Hotelling's T-Squared\|Hotelling's T-Squared]] — Hypothesis testing

---

## References

- **Book:** Johnson, R. A., & Wichern, D. W. (2007). *Applied Multivariate Statistical Analysis*. Pearson.
