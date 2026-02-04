---
{"dg-publish":true,"permalink":"/stats/01-foundations/lu-decomposition/","tags":["Linear-Algebra","Numerical-Methods","Matrix-Factorization"]}
---

## Definition

> [!abstract] Core Statement
> **LU Decomposition** factors a square matrix $A$ into the product of a **lower triangular** matrix $L$ and an **upper triangular** matrix $U$:
> $$A = LU$$

This is the matrix form of **Gaussian elimination**.

---

## Purpose

1.  **Solve linear systems** efficiently: $Ax = b \Rightarrow LUx = b$.
2.  **Compute determinants**: $\det(A) = \det(L) \cdot \det(U) = \prod u_{ii}$.
3.  **Invert matrices** more stably than direct methods.
4.  **Reuse factorization** for multiple right-hand sides.

---

## When to Use

> [!success] Use LU Decomposition When...
> - Solving $Ax = b$ for **multiple** different $b$ vectors.
> - Computing determinants of large matrices.
> - Matrix is **square** and **non-singular**.

> [!failure] Limitations
> - Requires **pivoting** for numerical stability (→ PLU).
> - Matrix must be square.
> - Not suitable for sparse matrices (use specialized methods).

---

## Theoretical Background

### Structure

For a 3×3 matrix:
$$A = LU = \begin{pmatrix} 1 & 0 & 0 \\ l_{21} & 1 & 0 \\ l_{31} & l_{32} & 1 \end{pmatrix} \begin{pmatrix} u_{11} & u_{12} & u_{13} \\ 0 & u_{22} & u_{23} \\ 0 & 0 & u_{33} \end{pmatrix}$$

### Solving $Ax = b$

1. **Factor**: $A = LU$
2. **Forward substitution**: Solve $Ly = b$ for $y$
3. **Back substitution**: Solve $Ux = y$ for $x$

**Complexity**: $O(n^3)$ for factorization, $O(n^2)$ per solve.

### PLU Decomposition (with pivoting)

$$PA = LU$$

Where $P$ is a permutation matrix (row swaps for stability).

---

## Worked Example

> [!example] Problem
> Find the LU decomposition of:
> $$A = \begin{pmatrix} 2 & 1 & 1 \\ 4 & 3 & 3 \\ 8 & 7 & 9 \end{pmatrix}$$

**Solution:**

1. **Eliminate column 1:**
   - $R_2 \leftarrow R_2 - 2R_1$: multiplier $l_{21} = 2$
   - $R_3 \leftarrow R_3 - 4R_1$: multiplier $l_{31} = 4$

2. **Eliminate column 2:**
   - $R_3 \leftarrow R_3 - 3R_2$: multiplier $l_{32} = 3$

3. **Result:**
$$L = \begin{pmatrix} 1 & 0 & 0 \\ 2 & 1 & 0 \\ 4 & 3 & 1 \end{pmatrix}, \quad U = \begin{pmatrix} 2 & 1 & 1 \\ 0 & 1 & 1 \\ 0 & 0 & 2 \end{pmatrix}$$

4. **Verify**: $LU = A$ ✅

5. **Determinant**: $\det(A) = u_{11} \cdot u_{22} \cdot u_{33} = 2 \cdot 1 \cdot 2 = 4$

---

## Python Implementation

```python
import numpy as np
from scipy.linalg import lu, lu_factor, lu_solve

# Matrix
A = np.array([
    [2, 1, 1],
    [4, 3, 3],
    [8, 7, 9]
], dtype=float)

# Method 1: scipy.linalg.lu (returns P, L, U)
P, L, U = lu(A)
print("L:\n", L)
print("U:\n", U)
print("Verify P @ L @ U:\n", P @ L @ U)

# Method 2: lu_factor for solving systems
lu_piv = lu_factor(A)

# Solve Ax = b for multiple b vectors
b1 = np.array([1, 2, 3])
b2 = np.array([4, 5, 6])

x1 = lu_solve(lu_piv, b1)
x2 = lu_solve(lu_piv, b2)
print(f"x1: {x1}")
print(f"x2: {x2}")

# Determinant from U diagonal
det_A = np.prod(np.diag(U)) * np.linalg.det(P)
print(f"Determinant: {det_A}")

# Manual LU (no pivoting - educational only)
def lu_no_pivot(A):
    n = A.shape[0]
    L = np.eye(n)
    U = A.copy()
    
    for k in range(n-1):
        for i in range(k+1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]
    
    return L, U

L_manual, U_manual = lu_no_pivot(A)
print("Manual L:\n", L_manual)
```

---

## R Implementation

```r
library(Matrix)

A <- matrix(c(2, 4, 8, 1, 3, 7, 1, 3, 9), nrow = 3)

# LU decomposition
lu_result <- expand(lu(A))
L <- lu_result$L
U <- lu_result$U
P <- lu_result$P

print("L:"); print(L)
print("U:"); print(U)

# Solve system
b <- c(1, 2, 3)
x <- solve(A, b)
print(paste("Solution:", x))

# Determinant
det_A <- prod(diag(U)) * det(P)
print(paste("Determinant:", det_A))
```

---

## Comparison with Other Methods

| Method | Use Case | Complexity |
|--------|----------|------------|
| **LU** | General square systems | $O(n^3)$ |
| **Cholesky** | Symmetric positive definite | $O(n^3/3)$ |
| **QR** | Least squares, non-square | $O(2n^2m)$ |
| **SVD** | Rank-deficient, conditioning | $O(n^3)$ |

---

## ML Applications

| Application | Why LU Helps |
|-------------|--------------|
| **Linear Regression** | Solve normal equations $(X^TX)\beta = X^Ty$. |
| **Determinant Calculation** | For Gaussian likelihood normalization. |
| **Matrix Inversion** | When explicit inverse needed. |
| **Preconditioning** | Approximate factorization for iterative solvers. |

---

## Related Concepts

- [[stats/01_Foundations/Determinants & Matrix Inversion\|Determinants & Matrix Inversion]] - LU provides efficient determinant.
- [[stats/01_Foundations/Singular Value Decomposition (SVD)\|Singular Value Decomposition (SVD)]] - More general decomposition.
- [[stats/01_Foundations/Positive Definite Matrices\|Positive Definite Matrices]] - Use Cholesky instead for PD matrices.
- [[stats/01_Foundations/Matrix Multiplication\|Matrix Multiplication]] - LU is a factorization into simpler products.

---

## References

- **Book:** Trefethen, L. N., & Bau, D. (1997). *Numerical Linear Algebra*. SIAM.
- **Book:** Golub, G. H., & Van Loan, C. F. (2013). *Matrix Computations* (4th ed.). Johns Hopkins University Press.
