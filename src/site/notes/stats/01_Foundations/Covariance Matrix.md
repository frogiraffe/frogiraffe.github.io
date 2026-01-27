---
{"dg-publish":true,"permalink":"/stats/01-foundations/covariance-matrix/","tags":["Linear-Algebra","Statistics","Multivariate"]}
---


## Definition

> [!abstract] Core Statement
> The **Covariance Matrix** is a ==symmetric positive semi-definite matrix== containing all pairwise covariances between variables. For random vector $\mathbf{X} = (X_1, \dots, X_p)^T$:
> $$\Sigma_{ij} = \text{Cov}(X_i, X_j) = E[(X_i - \mu_i)(X_j - \mu_j)]$$

---

## Properties

| Property | Description |
|----------|-------------|
| **Symmetric** | $\Sigma = \Sigma^T$ |
| **Diagonal** | Variances: $\Sigma_{ii} = \text{Var}(X_i)$ |
| **PSD** | All eigenvalues ≥ 0 |

---

## Formulas

**Sample covariance:**
$$S_{ij} = \frac{1}{n-1} \sum_{k=1}^{n} (x_{ki} - \bar{x}_i)(x_{kj} - \bar{x}_j)$$

**To Correlation:**
$$R_{ij} = \frac{S_{ij}}{\sqrt{S_{ii} S_{jj}}}$$

---

## Python Implementation

```python
import numpy as np

X = np.random.randn(100, 3)  # 100 obs, 3 vars

cov_matrix = np.cov(X, rowvar=False)
corr_matrix = np.corrcoef(X, rowvar=False)

print("Covariance Matrix:\n", cov_matrix)
print("Correlation Matrix:\n", corr_matrix)
```

---

## R Implementation

```r
X <- matrix(rnorm(300), ncol = 3)

cov(X)   # Covariance matrix
cor(X)   # Correlation matrix
```

---

## Applications

- [[stats/04_Supervised_Learning/Principal Component Analysis (PCA)\|Principal Component Analysis (PCA)]] - Eigendecomposition of Σ
- [[Multivariate Normal Distribution\|Multivariate Normal Distribution]] - Parameterized by μ and Σ
- [[stats/03_Regression_Analysis/Multiple Linear Regression\|Multiple Linear Regression]] - Variance of coefficients

---

## References

- **Book:** Johnson, R. A., & Wichern, D. W. (2007). *Applied Multivariate Statistical Analysis*. Pearson. [Pearson Link](https://www.pearson.com/us/higher-education/program/Johnson-Applied-Multivariate-Statistical-Analysis-6th-Edition/PGM248880.html)
