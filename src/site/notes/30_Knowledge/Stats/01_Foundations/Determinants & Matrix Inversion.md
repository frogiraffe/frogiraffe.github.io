---
{"dg-publish":true,"permalink":"/30-knowledge/stats/01-foundations/determinants-and-matrix-inversion/","tags":["probability","foundations"]}
---

## Definition

> [!abstract] Core Statements
> **Determinant:** A scalar value that encodes important properties of a square matrix—whether it's invertible, and how it scales volumes.
> 
> **Matrix Inverse:** For a square matrix $A$, the inverse $A^{-1}$ satisfies $A A^{-1} = A^{-1} A = I$.

---

## Purpose

1.  Determine if a matrix is **invertible** (det ≠ 0).
2.  Solve linear systems: $Ax = b \Rightarrow x = A^{-1}b$.
3.  Compute **volume scaling** of linear transformations.
4.  Calculate eigenvalues via $\det(A - \lambda I) = 0$.

---

## Theoretical Background

### Determinant Formulas

**2×2 Matrix:**
$$\det \begin{pmatrix} a & b \\ c & d \end{pmatrix} = ad - bc$$

**3×3 Matrix (Sarrus' Rule):**
$$\det \begin{pmatrix} a & b & c \\ d & e & f \\ g & h & i \end{pmatrix} = aei + bfg + cdh - ceg - bdi - afh$$

**General (Cofactor Expansion):**
$$\det(A) = \sum_{j=1}^{n} (-1)^{i+j} a_{ij} \det(M_{ij})$$

Where $M_{ij}$ is the minor (matrix with row $i$ and column $j$ removed).

### Key Properties

| Property | Formula |
|----------|---------|
| **Product** | $\det(AB) = \det(A) \det(B)$ |
| **Transpose** | $\det(A^T) = \det(A)$ |
| **Inverse** | $\det(A^{-1}) = \frac{1}{\det(A)}$ |
| **Scalar** | $\det(cA) = c^n \det(A)$ for $n \times n$ matrix |
| **Triangular** | $\det(A) = \prod_{i=1}^n a_{ii}$ (diagonal elements) |

### Matrix Inversion

**2×2 Inverse:**
$$A^{-1} = \frac{1}{ad-bc} \begin{pmatrix} d & -b \\ -c & a \end{pmatrix}$$

**General Formula:**
$$A^{-1} = \frac{1}{\det(A)} \text{adj}(A)$$

Where $\text{adj}(A)$ is the adjugate (transpose of cofactor matrix).

---

## Worked Example

> [!example] Problem
> Find the determinant and inverse of:
> $$A = \begin{pmatrix} 3 & 1 \\ 2 & 4 \end{pmatrix}$$

**Solution:**

1. **Determinant:**
$$\det(A) = 3(4) - 1(2) = 12 - 2 = 10$$

2. **Inverse:**
$$A^{-1} = \frac{1}{10} \begin{pmatrix} 4 & -1 \\ -2 & 3 \end{pmatrix} = \begin{pmatrix} 0.4 & -0.1 \\ -0.2 & 0.3 \end{pmatrix}$$

3. **Verify:**
$$A A^{-1} = \begin{pmatrix} 3 & 1 \\ 2 & 4 \end{pmatrix} \begin{pmatrix} 0.4 & -0.1 \\ -0.2 & 0.3 \end{pmatrix} = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$$ ✅

---

## Python Implementation

```python
import numpy as np

# Determinant
A = np.array([[3, 1], [2, 4]])
det_A = np.linalg.det(A)
print(f"Determinant: {det_A}")  # 10.0

# Inverse
A_inv = np.linalg.inv(A)
print(f"Inverse:\n{A_inv}")

# Verify
print(f"A @ A_inv:\n{A @ A_inv}")  # Identity matrix

# Manual 2x2 inverse
def inverse_2x2(A):
    a, b, c, d = A[0,0], A[0,1], A[1,0], A[1,1]
    det = a*d - b*c
    if abs(det) < 1e-10:
        raise ValueError("Matrix is singular")
    return np.array([[d, -b], [-c, a]]) / det

# Solve linear system Ax = b
b = np.array([5, 6])
x = np.linalg.solve(A, b)  # More stable than A_inv @ b
print(f"Solution x: {x}")

# Check if invertible
def is_invertible(A, tol=1e-10):
    return abs(np.linalg.det(A)) > tol

# 3x3 example
B = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
print(f"3x3 Det: {np.linalg.det(B):.4f}")
print(f"3x3 Inverse:\n{np.linalg.inv(B)}")
```

---

## R Implementation

```r
# Determinant
A <- matrix(c(3, 2, 1, 4), nrow = 2)
det_A <- det(A)
print(paste("Determinant:", det_A))

# Inverse
A_inv <- solve(A)
print("Inverse:")
print(A_inv)

# Verify
print("A %*% A_inv:")
print(A %*% A_inv)

# Solve linear system
b <- c(5, 6)
x <- solve(A, b)
print(paste("Solution:", x))

# Check singularity
is_invertible <- function(A, tol = 1e-10) {
  abs(det(A)) > tol
}
```

---

## Geometric Interpretation

| Determinant | Meaning |
|-------------|---------|
| $\det(A) > 0$ | Preserves orientation, scales volume by det. |
| $\det(A) < 0$ | Flips orientation (reflection). |
| $\det(A) = 0$ | Collapses dimension (singular). |
| $\|\det(A)\| = 1$ | Preserves volume (rotation, reflection). |

---

## ML Applications

| Application | Role |
|-------------|------|
| **Linear Regression** | $(X^TX)^{-1}$ for closed-form solution. |
| **Covariance Matrix** | det ≈ 0 means multicollinearity. |
| **Gaussian PDF** | $\frac{1}{\sqrt{(2\pi)^n |\Sigma|}} \exp(...)$ |
| **Change of Variables** | Jacobian determinant in probability. |
| **Eigenvalues** | $\det(A - \lambda I) = 0$ |

---

## Limitations

> [!warning] Pitfalls
> 1. **Large Matrices:** Direct inversion is $O(n^3)$—use iterative methods.
> 2. **Numerical Instability:** Near-singular matrices give poor inverses.
> 3. **Prefer solve():** Use `np.linalg.solve(A, b)` not `inv(A) @ b`.
> 4. **Regularization:** Add $\lambda I$ if matrix is ill-conditioned.

---

## Related Concepts

- [[30_Knowledge/Stats/01_Foundations/Matrix Multiplication\|Matrix Multiplication]] - Determinant of product = product of determinants.
- [[30_Knowledge/Stats/01_Foundations/Eigenvalues & Eigenvectors\|Eigenvalues & Eigenvectors]] - Found via characteristic polynomial.
- [[30_Knowledge/Stats/01_Foundations/Singular Value Decomposition (SVD)\|Singular Value Decomposition (SVD)]] - For non-square/singular matrices.
- [[30_Knowledge/Stats/01_Foundations/Positive Definite Matrices\|Positive Definite Matrices]] - All leading minors positive.

---

## When to Use

> [!success] Use Determinants & Matrix Inversion When...
> - Refer to standard documentation
> - Refer to standard documentation

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Assumptions are violated
> - Alternative methods are more appropriate

---

## References

- **Book:** Strang, G. (2016). *Introduction to Linear Algebra* (5th ed.). Wellesley-Cambridge Press.
- **Book:** Lay, D. C., Lay, S. R., & McDonald, J. J. (2016). *Linear Algebra and Its Applications* (5th ed.). Pearson.
