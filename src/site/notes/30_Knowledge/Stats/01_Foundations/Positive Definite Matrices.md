---
{"dg-publish":true,"permalink":"/30-knowledge/stats/01-foundations/positive-definite-matrices/","tags":["probability","foundations"]}
---

## Definition

> [!abstract] Core Statement
> A symmetric matrix $A$ is **positive definite** if for all non-zero vectors $x$:
> $$x^T A x > 0$$
> 
> Equivalently, all eigenvalues of $A$ are strictly positive.

---

## Purpose

1.  Ensure **convexity** in optimization (Hessian must be PD at minimum).
2.  Valid **covariance matrices** must be positive semi-definite.
3.  Enable **Cholesky decomposition** for efficient computation.
4.  Guarantee **kernel validity** in SVMs and Gaussian Processes.

---

## When to Use

> [!success] Check for Positive Definiteness When...
> - Verifying a covariance matrix is valid.
> - Checking if a function is convex (via Hessian).
> - Using kernel methods (kernel matrix must be PSD).
> - Applying Cholesky decomposition.

> [!failure] Common Issues
> - Numerical precision can make matrices appear non-PD.
> - Adding small diagonal (regularization) fixes near-singular cases.

---

## Theoretical Background

### Equivalent Conditions

A symmetric matrix $A$ is positive definite **if and only if**:

| Condition | Description |
|-----------|-------------|
| $x^T A x > 0$ | Quadratic form positive for all $x \neq 0$ |
| All eigenvalues $\lambda_i > 0$ | Eigenvalue test |
| All leading principal minors $> 0$ | Sylvester's criterion |
| $A = L L^T$ exists | Cholesky decomposition |
| $A = B^T B$ for some $B$ | Gram matrix representation |

### Positive Semi-Definite (PSD)

- $x^T A x \geq 0$ (allows equality)
- All eigenvalues $\lambda_i \geq 0$
- Covariance matrices are PSD (with rank = # linearly independent variables)

### Sylvester's Criterion

For a 3×3 matrix:
$$A = \begin{pmatrix} a_{11} & a_{12} & a_{13} \\ a_{12} & a_{22} & a_{23} \\ a_{13} & a_{23} & a_{33} \end{pmatrix}$$

Check:
1. $a_{11} > 0$
2. $\det\begin{pmatrix} a_{11} & a_{12} \\ a_{12} & a_{22} \end{pmatrix} > 0$
3. $\det(A) > 0$

---

## Worked Example

> [!example] Problem
> Determine if the following matrix is positive definite:
> $$A = \begin{pmatrix} 4 & 2 \\ 2 & 5 \end{pmatrix}$$

**Solution (Three Methods):**

**Method 1: Eigenvalues**
$$\det(A - \lambda I) = (4-\lambda)(5-\lambda) - 4 = \lambda^2 - 9\lambda + 16 = 0$$
$$\lambda = \frac{9 \pm \sqrt{81 - 64}}{2} = \frac{9 \pm \sqrt{17}}{2}$$
$$\lambda_1 \approx 6.56 > 0, \quad \lambda_2 \approx 2.44 > 0$$
✅ **Positive definite**

**Method 2: Sylvester's Criterion**
- $a_{11} = 4 > 0$ ✅
- $\det(A) = 4(5) - 2(2) = 16 > 0$ ✅

**Method 3: Cholesky Decomposition**
$$A = L L^T = \begin{pmatrix} 2 & 0 \\ 1 & 2 \end{pmatrix} \begin{pmatrix} 2 & 1 \\ 0 & 2 \end{pmatrix}$$
Decomposition exists ✅ → **Positive definite**

---

## Python Implementation

```python
import numpy as np
from scipy.linalg import cholesky, eigvalsh

def is_positive_definite(A, method='eigenvalue', tol=1e-10):
    """Check if matrix A is positive definite."""
    
    # Ensure symmetric
    if not np.allclose(A, A.T):
        return False, "Matrix is not symmetric"
    
    if method == 'eigenvalue':
        eigenvalues = eigvalsh(A)  # For symmetric matrices
        is_pd = np.all(eigenvalues > tol)
        return is_pd, eigenvalues
    
    elif method == 'cholesky':
        try:
            L = cholesky(A, lower=True)
            return True, L
        except np.linalg.LinAlgError:
            return False, "Cholesky decomposition failed"

# Test cases
A_pd = np.array([[4, 2], [2, 5]])
A_psd = np.array([[1, 1], [1, 1]])  # Rank 1, PSD but not PD
A_negative = np.array([[1, 2], [2, 1]])  # Not PD

for name, mat in [("PD", A_pd), ("PSD", A_psd), ("Negative", A_negative)]:
    is_pd, info = is_positive_definite(mat)
    print(f"{name}: {is_pd}, eigenvalues: {eigvalsh(mat)}")

# Make a matrix PD by adding regularization
def make_positive_definite(A, epsilon=1e-6):
    """Add small diagonal to ensure positive definiteness."""
    min_eig = np.min(eigvalsh(A))
    if min_eig < 0:
        A = A + (-min_eig + epsilon) * np.eye(A.shape[0])
    return A
```

---

## R Implementation

```r
library(Matrix)

is_positive_definite <- function(A, tol = 1e-10) {
  # Check symmetry
  if (!isSymmetric(A)) return(list(is_pd = FALSE, reason = "Not symmetric"))
  
  # Eigenvalue method
  eigenvalues <- eigen(A, symmetric = TRUE)$values
  
  if (all(eigenvalues > tol)) {
    return(list(is_pd = TRUE, eigenvalues = eigenvalues))
  } else {
    return(list(is_pd = FALSE, eigenvalues = eigenvalues))
  }
}

# Cholesky method
is_pd_cholesky <- function(A) {
  tryCatch({
    L <- chol(A)
    return(TRUE)
  }, error = function(e) {
    return(FALSE)
  })
}

# Test
A <- matrix(c(4, 2, 2, 5), nrow = 2)
print(is_positive_definite(A))
print(is_pd_cholesky(A))
```

---

## ML Applications

| Application | Role of Positive Definiteness |
|-------------|------------------------------|
| **Covariance Matrices** | Must be PSD for valid probability distributions. |
| **Gaussian Processes** | Kernel matrix must be PSD. |
| **SVM Kernels** | Mercer's theorem: valid kernels produce PSD Gram matrices. |
| **Optimization** | Hessian PD at stationary point ⟹ local minimum. |
| **Regularization** | $X^TX + \lambda I$ is PD for $\lambda > 0$. |

---

## Interpretation Guide

| Output | Interpretation |
|--------|----------------|
| **All eigenvalues > 0** | Matrix is positive definite. |
| **Some eigenvalues = 0** | Matrix is positive semi-definite (rank-deficient). |
| **Any eigenvalue < 0** | Matrix is indefinite (not PD). |
| **Cholesky fails** | Matrix is not positive definite. |

---

## Related Concepts

- [[30_Knowledge/Stats/01_Foundations/Covariance Matrix\|Covariance Matrix]] - Must be PSD.
- [[30_Knowledge/Stats/01_Foundations/Eigenvalues & Eigenvectors\|Eigenvalues & Eigenvectors]] - PD ⟺ all eigenvalues positive.
- [[30_Knowledge/Stats/01_Foundations/Matrix Multiplication\|Matrix Multiplication]] - $A = B^TB$ is always PSD.
- [[30_Knowledge/Stats/03_Regression_Analysis/Ridge Regression\|Ridge Regression]] - Adds $\lambda I$ to ensure PD.

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Assumptions are violated
> - Alternative methods are more appropriate

---

## References

- **Book:** Horn, R. A., & Johnson, C. R. (2012). *Matrix Analysis* (2nd ed.). Cambridge University Press.
- **Book:** Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*. Cambridge University Press. Chapter 3.
