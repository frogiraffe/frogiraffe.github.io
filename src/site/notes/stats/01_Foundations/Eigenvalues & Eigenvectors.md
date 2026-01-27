---
{"dg-publish":true,"permalink":"/stats/01-foundations/eigenvalues-and-eigenvectors/","tags":["Math","Linear-Algebra","Dimensionality-Reduction"]}
---

## Definition

> [!abstract] Core Statement
> For a square matrix $A$, an **Eigenvector** ($v$) is a non-zero vector that, when multiplied by $A$, does not change direction, only magnitude. The **Eigenvalue** ($\lambda$) is the scalar factor by which it stretches or shrinks.
> 
> $$ Av = \lambda v $$

![Eigenvectors Transformation Illustration](https://upload.wikimedia.org/wikipedia/commons/3/3c/Eigenvectors-extended.gif)

---

## Purpose

1.  **Principal Component Analysis (PCA):** The eigenvectors of the covariance matrix are the "Principal Components" (directions of max variance). The eigenvalues represent the *amount* of variance explained.
2.  **Google PageRank:** The dominant eigenvector of the web graph matrix determines page importance.
3.  **Stability Analysis:** Determining if a system (physics, economics) will explode or settle down.

---

## Intuition

Imagine a transformation matrix $A$ acts like stretching a piece of fabric.
-   Most vectors (threads) get rotated and stretched.
-   **Eigenvectors** are the specific threads that **stay pointing in the same line**.
-   **Eigenvalues** tell you how much that specific thread was stretched (2x? 0.5x? -1x?).

---

## Worked Example: PCA Context

> [!example] Variance of Data
> You have a dataset with Covariance Matrix $\Sigma$:
> $$ \Sigma = \begin{bmatrix} 4 & 2 \\ 2 & 3 \end{bmatrix} $$
> 
> You calculate eigenvalues and eigenvectors.
> 
> **Result:**
> 1.  $\lambda_1 = 5.56$, $v_1 = [0.79, 0.61]$
> 2.  $\lambda_2 = 1.44$, $v_2 = [-0.61, 0.79]$
> 
> **Interpretation:**
> -   **Direction:** The data varies most along the direction $[0.79, 0.61]$. This is **PC1**.
> -   **Magnitude:** The variance along this axis is 5.56.
> -   **Proportion Explained:** $\frac{5.56}{5.56 + 1.44} = \frac{5.56}{7} \approx 79.4\%$.
> -   PC1 captures 79.4% of the information.

---

## Assumptions

- [ ] **Square Matrix:** Eigenvalues exist for $n \times n$ matrices.
- [ ] **Decomposition:** Not all matrices are diagonalizable (defective matrices), though symmetric matrices (like Covariance) always are.

---

## Properties

| Property | Description |
|----------|-------------|
| **Trace** | Sum of eigenvalues = Sum of diagonal elements (Trace of A). |
| **Determinant** | Product of eigenvalues = Determinant of A. |
| **Symmetric Matrix** | Eigenvalues are Real numbers; Eigenvectors are Orthogonal (perpendicular). |
| **Covariance Matrix** | Always Symmetric Positive Semi-Definite ($\lambda \ge 0$). |

---

## Python Implementation

```python
import numpy as np

# 2x2 Covariance Matrix
A = np.array([[4, 2], 
              [2, 3]])

# Calculate Eig
eigenvalues, eigenvectors = np.linalg.eig(A)

print("Eigenvalues:", eigenvalues)
print("Eigenvectors:\n", eigenvectors)

# Check Av = lambda v
v1 = eigenvectors[:, 0]
lam1 = eigenvalues[0]

print("Av:", A @ v1)
print("lambda*v:", lam1 * v1)
```

---

## R Implementation

```r
# Define Matrix
A <- matrix(c(4, 1, 2, 3), nrow=2, byrow=TRUE)

# Eigendecomposition
eigen_res <- eigen(A)

# Values and Vectors
vals <- eigen_res$values
vecs <- eigen_res$vectors

print(vals)
print(vecs)

# Verify A*v = lambda*v
v1 <- vecs[,1]
lambda1 <- vals[1]
print(A %*% v1)
print(lambda1 * v1)
```

---

## Related Concepts

- [[stats/04_Machine_Learning/Principal Component Analysis (PCA)\|Principal Component Analysis (PCA)]] - Main application in stats.
- [[stats/01_Foundations/Matrix Multiplication\|Matrix Multiplication]] - The operation $Av$.
- [[stats/01_Foundations/Covariance Matrix\|Covariance Matrix]] - The input matrix often used.
- [[stats/01_Foundations/Singular Value Decomposition (SVD)\|Singular Value Decomposition (SVD)]] - Generalization for non-square matrices (used in calculating PCA in practice).

---

## References

- **Book:** Strang, G. (2016). *Introduction to Linear Algebra* (5th ed.). Wellesley-Cambridge Press. [Book Website](https://math.mit.edu/~gs/linearalgebra/)
- **Book:** Lay, D. C. (2012). *Linear Algebra and Its Applications* (4th ed.). Pearson. [Pearson Link](https://www.pearson.com/en-us/subject-catalog/p/linear-algebra-and-its-applications/P200000006322/)
- **Book:** Axler, S. (2015). *Linear Algebra Done Right* (3rd ed.). Springer. [Springer Link](https://link.springer.com/book/10.1007/978-3-319-11080-6)
