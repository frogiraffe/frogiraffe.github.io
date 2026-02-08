---
{"dg-publish":true,"permalink":"/30-knowledge/stats/01-foundations/multivariate-normal-distribution/","tags":["probability","foundations"]}
---

## Definition

> [!abstract] Core Statement
> The **Multivariate Normal Distribution** generalizes the normal distribution to ==multiple correlated variables==, characterized by a mean vector $\boldsymbol{\mu}$ and covariance matrix $\boldsymbol{\Sigma}$.

![Bivariate Normal Distribution showing correlation structure|500](https://upload.wikimedia.org/wikipedia/commons/8/8e/MultivariateNormal.png)
*Figure 1: Bivariate normal distribution. Contours show constant density; ellipse orientation shows correlation.*

$$
f(\mathbf{x}) = \frac{1}{\sqrt{(2\pi)^k|\boldsymbol{\Sigma}|}} \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)
$$

---

> [!tip] Intuition (ELI5): The Stretched Blob
> A univariate Normal is a bell curve. A bivariate Normal is a "blob" viewed from above—like contour lines on a map. If variables are correlated, the blob is stretched diagonally. Uncorrelated variables make a circular blob. The covariance matrix determines the stretch direction and amount.

---

## Purpose

1. **Multivariate inference:** Joint distributions of multiple variables
2. **Machine learning:** Gaussian classifiers, GMMs, Gaussian Processes
3. **Regression:** Assumption of error structure
4. **Finance:** Portfolio risk (correlated returns)

---

## When to Use

> [!success] Use Multivariate Normal When...
> - Modeling **joint distribution** of correlated continuous variables
> - Variables are approximately **normally distributed**
> - Need to capture **linear correlations** between variables

---

## When NOT to Use

> [!danger] Do NOT Use Multivariate Normal When...
> - **Non-linear relationships:** MVN only captures linear correlation
> - **Heavy tails:** Use multivariate t-distribution
> - **Discrete outcomes:** Use appropriate discrete distribution
> - **Asymmetric/skewed data:** MVN is symmetric

---

## Theoretical Background

### Notation

$$
\mathbf{X} \sim \mathcal{N}_k(\boldsymbol{\mu}, \boldsymbol{\Sigma})
$$

where:
- $\boldsymbol{\mu}$ = mean vector ($k \times 1$)
- $\boldsymbol{\Sigma}$ = covariance matrix ($k \times k$, symmetric, [[30_Knowledge/Stats/01_Foundations/Positive Definite Matrices\|positive definite]])

### Properties

| Property | Description |
|----------|-------------|
| **Marginals** | Each variable is univariate normal: $X_i \sim N(\mu_i, \sigma_{ii})$ |
| **Conditionals** | Conditionals are also normal (see below) |
| **Linear combinations** | Any linear combo $\mathbf{a}^T\mathbf{X} \sim N(\mathbf{a}^T\boldsymbol{\mu}, \mathbf{a}^T\boldsymbol{\Sigma}\mathbf{a})$ |
| **Sum** | Independent MVNs: $\mathbf{X} + \mathbf{Y} \sim \mathcal{N}(\boldsymbol{\mu}_X + \boldsymbol{\mu}_Y, \boldsymbol{\Sigma}_X + \boldsymbol{\Sigma}_Y)$ |

### Mahalanobis Distance

The exponent in the PDF is the squared Mahalanobis Distance:
$$D^2 = (\mathbf{x}-\boldsymbol{\mu})^T\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})$$

This measures how far a point is from the mean, accounting for correlations.

### Conditional Distribution

For bivariate case $\mathbf{X} = (X_1, X_2)^T$:

$$
X_1 | X_2 = x_2 \sim N\left(\mu_1 + \frac{\sigma_{12}}{\sigma_{22}}(x_2 - \mu_2), \sigma_{11} - \frac{\sigma_{12}^2}{\sigma_{22}}\right)
$$

**Key insight:** Conditional variance is always less than marginal variance (unless $\rho = 0$).

---

## Worked Example: Bivariate Data

> [!example] Problem
> Heights (X₁) and weights (X₂) of adults are jointly normal with:
> - $\mu = (170, 70)$ (cm and kg)
> - $\Sigma = \begin{pmatrix} 100 & 30 \\ 30 & 25 \end{pmatrix}$
> 
> **Question:** What is the conditional distribution of weight given height = 180 cm?

**Solution:**

Using the conditional formula:
- $\mu_1 = 170$, $\mu_2 = 70$
- $\sigma_{11} = 100$, $\sigma_{22} = 25$, $\sigma_{12} = 30$

$$E[X_2 | X_1 = 180] = 70 + \frac{30}{100}(180 - 170) = 70 + 3 = 73 \text{ kg}$$

$$\text{Var}[X_2 | X_1 = 180] = 25 - \frac{30^2}{100} = 25 - 9 = 16$$

**Result:** $X_2 | X_1 = 180 \sim N(73, 16)$

**Verification with Code:**
```python
import numpy as np
from scipy import stats

mean = np.array([170, 70])
cov = np.array([[100, 30],
                [30, 25]])

# Conditional distribution parameters
def conditional_normal(mu, sigma, idx_given, value_given, idx_target):
    """Compute conditional distribution X_target | X_given = value"""
    mu_t, mu_g = mu[idx_target], mu[idx_given]
    sig_tt = sigma[idx_target, idx_target]
    sig_gg = sigma[idx_given, idx_given]
    sig_tg = sigma[idx_target, idx_given]
    
    mu_cond = mu_t + (sig_tg / sig_gg) * (value_given - mu_g)
    var_cond = sig_tt - (sig_tg**2 / sig_gg)
    return mu_cond, var_cond

mu_cond, var_cond = conditional_normal(mean, cov, 0, 180, 1)
print(f"E[Weight | Height=180]: {mu_cond:.1f} kg")
print(f"Var[Weight | Height=180]: {var_cond:.1f}")
print(f"SD[Weight | Height=180]: {np.sqrt(var_cond):.1f} kg")
```

---

## Assumptions

- [ ] **Normality:** Each variable is normally distributed.
  - *Check:* Q-Q plots, Shapiro-Wilk test
  
- [ ] **Linearity:** Relationships are linear.
  - *Check:* Scatter plots for curved patterns
  
- [ ] **Positive definite Σ:** Covariance matrix is invertible.
  - *Check:* All eigenvalues > 0

---

## Limitations

> [!warning] Pitfalls
> 1. **Only linear correlation:** MVN can't capture non-linear relationships.
> 2. **Sensitivity to outliers:** Outliers severely affect $\boldsymbol{\mu}$ and $\boldsymbol{\Sigma}$ estimation.
> 3. **Curse of dimensionality:** In high dimensions, requires many parameters.

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
plt.figure(figsize=(8, 8))
plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title(f'Bivariate Normal (ρ = {cov[0,1]:.1f})')
plt.axis('equal')
plt.grid(alpha=0.3)
plt.show()

# ========== PDF ==========
x = np.array([1, 1])
prob_density = mvn.pdf(x)
print(f"PDF at {x}: {prob_density:.4f}")

# ========== CONTOUR PLOT ==========
x, y = np.mgrid[-3:3:.01, -3:3:.01]
pos = np.dstack((x, y))
plt.figure(figsize=(8, 8))
plt.contourf(x, y, mvn.pdf(pos), levels=20)
plt.colorbar(label='Density')
plt.title('Bivariate Normal PDF Contours')
plt.xlabel('X1')
plt.ylabel('X2')
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
plot(samples, xlab = "X1", ylab = "X2", main = "Bivariate Normal")

# PDF at a point
dmvnorm(c(1, 1), mean, sigma)

# Correlation check
cor(samples)
```

---

## Interpretation Guide

| Feature | Interpretation |
|---------|----------------|
| **Circular contours** | Uncorrelated variables ($\rho = 0$) |
| **Tilted ellipse** | Correlated variables |
| **Elongated ellipse** | Strong correlation |
| **Eigenvalues of Σ** | Variance along principal axes |

---

## Applications

| Application | Use |
|-------------|-----|
| **[[30_Knowledge/Stats/05_Unsupervised_Learning/PCA (Principal Component Analysis)\|Principal Component Analysis (PCA)]]** | Data → principal components (MVN eigenvectors) |
| **Gaussian Processes** | Prior/posterior are MVN |
| **Linear Discriminant Analysis (LDA)** | Class-conditional densities |
| **[[30_Knowledge/Stats/03_Regression_Analysis/Multiple Linear Regression\|Multiple Linear Regression]]** | Error term $\boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \sigma^2\mathbf{I})$ |
| **Portfolio theory** | Joint return distributions |

---

## Related Concepts

### Directly Related
- [[30_Knowledge/Stats/01_Foundations/Normal Distribution\|Normal Distribution]] - Univariate case
- [[30_Knowledge/Stats/01_Foundations/Covariance Matrix\|Covariance Matrix]] - Second parameter
- Mahalanobis Distance - Distance metric for MVN

### Hypothesis Testing
- [[30_Knowledge/Stats/02_Statistical_Inference/Hotelling's T-Squared\|Hotelling's T-Squared]] - Multivariate t-test

### Other Related Topics

{ .block-language-dataview}

---

## References

1. Johnson, R. A., & Wichern, D. W. (2007). *Applied Multivariate Statistical Analysis* (6th ed.). Pearson. [Available online](https://www.pearson.com/en-us/subject-catalog/p/applied-multivariate-statistical-analysis/P200000003483/)

2. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer. Section 2.3. [Available online](https://www.springer.com/gp/book/9780387310732)

3. Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press. Chapter 4. [Available online](https://mitpress.mit.edu/9780262017091/machine-learning/)
