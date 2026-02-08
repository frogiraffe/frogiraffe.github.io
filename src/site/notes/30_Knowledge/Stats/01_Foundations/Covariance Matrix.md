---
{"dg-publish":true,"permalink":"/30-knowledge/stats/01-foundations/covariance-matrix/","tags":["probability","foundations"]}
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

- [[30_Knowledge/Stats/05_Unsupervised_Learning/PCA (Principal Component Analysis)\|PCA (Principal Component Analysis)]] - Eigendecomposition of Σ
- [[30_Knowledge/Stats/01_Foundations/Multivariate Normal Distribution\|Multivariate Normal Distribution]] - Parameterized by μ and Σ
- [[30_Knowledge/Stats/03_Regression_Analysis/Multiple Linear Regression\|Multiple Linear Regression]] - Variance of coefficients
- [[30_Knowledge/Stats/01_Foundations/Positive Definite Matrices\|Positive Definite Matrices]] - Covariance matrices are always PSD

---

## When to Use

> [!success] Use Covariance Matrix When...
> - Refer to standard documentation
> - Refer to standard documentation

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Assumptions are violated
> - Alternative methods are more appropriate

---

## Related Concepts

- [[30_Knowledge/Stats/01_Foundations/Normal Distribution\|Normal Distribution]]
- [[30_Knowledge/Stats/01_Foundations/Central Limit Theorem (CLT)\|Central Limit Theorem (CLT)]]
- [[30_Knowledge/Stats/01_Foundations/Variance\|Variance]]

---

## References

- **Book:** Johnson, R. A., & Wichern, D. W. (2007). *Applied Multivariate Statistical Analysis*. Pearson. [Pearson Link](https://www.pearson.com/us/higher-education/program/Johnson-Applied-Multivariate-Statistical-Analysis-6th-Edition/PGM248880.html)
