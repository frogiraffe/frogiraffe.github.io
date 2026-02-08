---
{"dg-publish":true,"permalink":"/30-knowledge/stats/01-foundations/singular-value-decomposition-svd/","tags":["probability","foundations"]}
---


## Definition

> [!abstract] Core Statement
> **Singular Value Decomposition (SVD)** factorizes any matrix $A$ into three matrices:
> $$A = U \Sigma V^T$$
> where U, V are orthogonal and Σ is diagonal with singular values.

![Singular Value Decomposition Visualization (2x2 Matrix)](https://commons.wikimedia.org/wiki/Special:FilePath/Singular-Value-Decomposition.svg)

---

## Components

| Matrix | Size | Contains |
|--------|------|----------|
| **U** | m × m | Left singular vectors |
| **Σ** | m × n | Singular values (diagonal) |
| **V^T** | n × n | Right singular vectors |

---

## Applications

- **PCA:** Principal components from SVD
- **Dimensionality reduction:** Truncated SVD for compression
- **Recommender systems:** Collaborative filtering
- **Pseudoinverse:** Computing $(A^TA)^{-1}A^T$

---

## Python Implementation

```python
import numpy as np

A = np.random.randn(100, 50)
U, s, Vt = np.linalg.svd(A, full_matrices=False)

# Truncated SVD (keep top k)
k = 10
A_approx = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
```

---

## R Implementation

```r
A <- matrix(rnorm(5000), nrow = 100)
svd_result <- svd(A)
U <- svd_result$u
d <- svd_result$d
V <- svd_result$v
```

---

## Related Concepts

- [[30_Knowledge/Stats/05_Unsupervised_Learning/PCA (Principal Component Analysis)\|PCA (Principal Component Analysis)]] - Uses SVD internally
- [[30_Knowledge/Stats/01_Foundations/Eigenvalues & Eigenvectors\|Eigenvalue]] - Related to singular values

---

## When to Use

> [!success] Use Singular Value Decomposition (SVD) When...
> - Refer to standard documentation
> - Refer to standard documentation

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Assumptions are violated
> - Alternative methods are more appropriate

---

## References

- **Book:** Strang, G. (2019). *Linear Algebra and Learning from Data*. Wellesley-Cambridge. [MIT Link](https://math.mit.edu/~gs/learningfromdata/)
