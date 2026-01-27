---
{"dg-publish":true,"permalink":"/stats/01-foundations/matrix-multiplication/","tags":["Math","Linear-Algebra","Foundations"]}
---

## Definition

> [!abstract] Core Statement
> **Matrix Multiplication** is a binary operation that takes a pair of matrices and produces another matrix. If $A$ is an $m \times n$ matrix and $B$ is an $n \times p$ matrix, their product $C = AB$ is an $m \times p$ matrix where each element $c_{ij}$ is the **dot product** of the $i$-th row of $A$ and the $j$-th column of $B$.

---

## Purpose

1.  **Linear Transformations:** Rotating, scaling, or shearing space (Computer Graphics, Data Preprocessing).
2.  **System of Equations:** Representing $Ax = b$ efficiently.
3.  **Neural Networks:** The "Forward Pass" is essentially a series of matrix multiplications (Weights $\times$ Inputs).
4.  **Covariance Calculations:** $X^T X$ constructs correlations.

---

## The Rules

### 1. Dimension Compatibility
To multiply $A \times B$:
-   Columns of $A$ Must Equal Rows of $B$.
-   $(m \times \mathbf{n}) \cdot (\mathbf{n} \times p) \to (m \times p)$
-   If inner dimensions don't match, multiplication is **undefined**.

### 2. The Dot Product Operation
For element $c_{ij}$:
$$ c_{ij} = \text{Row}_i(A) \cdot \text{Col}_j(B) = \sum_{k=1}^{n} a_{ik} b_{kj} $$

---

## Worked Example: Shopping Bill

> [!example] Problem
> **Prices (Matrix P):** Apple=\$1, Banana=\$0.5.
> **Quantities (Matrix Q):**
> -   Person A: 2 Apples, 4 Bananas.
> -   Person B: 1 Apple, 0 Bananas.
> 
> **Calculate Total Cost for each.**

**Setup:**
-   $Q$ (Quantities) is $2 \times 2$:
    $$ \begin{bmatrix} 2 & 4 \\ 1 & 0 \end{bmatrix} $$
-   $P$ (Prices) is $2 \times 1$:
    $$ \begin{bmatrix} 1 \\ 0.5 \end{bmatrix} $$

**Calculation ($Q \times P$):**
-   **Person A:** $(2 \times 1) + (4 \times 0.5) = 2 + 2 = 4$.
-   **Person B:** $(1 \times 1) + (0 \times 0.5) = 1 + 0 = 1$.

**Result:**
$$ \begin{bmatrix} 4 \\ 1 \end{bmatrix} $$
Person A spends \$4, Person B spends \$1.

---

## Geometric Intuition

Matrix multiplication $AB$ can be seen as:
1.  **Transformation:** Columns of $A$ specify where the basis vectors ($\hat{i}, \hat{j}$) allow the input vectors ($B$) to land.
2.  **Composition:** Doing transformation $B$ followed by transformation $A$.

---

## Python Implementation

```python
import numpy as np

# Define A (2x3)
A = np.array([[1, 2, 3],
              [4, 5, 6]])

# Define B (3x2)
B = np.array([[7, 8],
              [9, 1],
              [2, 3]])

# Multiply
C = np.dot(A, B)
# OR ideal modern syntax:
C_modern = A @ B

print(f"Result (2x2):\n{C_modern}")
```

---

## R Implementation

```r
# Matrices in R
A <- matrix(c(1, 2, 3, 4), nrow=2, byrow=TRUE)
B <- matrix(c(5, 6, 7, 8), nrow=2, byrow=TRUE)

# Check dimensions
print(dim(A))

# Dot Product (Matrix Multiplication)
# Operator is %*%
C_dot <- A %*% B
print(C_dot)

# Element-wise Multiplication (Hadamard)
# Operator is *
C_elem <- A * B
print(C_elem)
```

---

## Limitations & Pitfalls

> [!warning] Pitfalls
> 1.  **Non-Commutative:** In general, $AB \neq BA$. Order matters! (Rotation then Shear $\neq$ Shear then Rotation).
> 2.  **Element-wise confusion:** Matrix multiplication is **NOT** just multiplying corresponding elements (that is the Hadamard Product).
> 3.  **Broadcasting:** In Python/NumPy, `A * B` often does element-wise multiplication. You usually want `A @ B` or `np.dot()`.

---

## Related Concepts

- [[stats/01_Foundations/Eigenvalues & Eigenvectors\|Eigenvalues & Eigenvectors]] - What vectors stay the same after multiplication?
- [[stats/04_Supervised_Learning/Principal Component Analysis (PCA)\|Principal Component Analysis (PCA)]] - Uses correlation matrix ($X^T X$).
- [[stats/03_Regression_Analysis/Multiple Linear Regression\|Multiple Linear Regression]] - Solution involves $(X^T X)^{-1} X^T y$.

---

## References

- **Book:** Strang, G. (2016). *Introduction to Linear Algebra* (5th ed.). Wellesley-Cambridge Press. [Book Site](https://math.mit.edu/~gs/linearalgebra/)
- **Book:** Lay, D. C., et al. (2015). *Linear Algebra and Its Applications* (5th ed.). Pearson. [Pearson Link](https://www.pearson.com/us/higher-education/program/Lay-Linear-Algebra-and-Its-Applications-5th-Edition/PGM315848.html)
- **Book:** Axler, S. (2015). *Linear Algebra Done Right* (3rd ed.). Springer. [Springer Link](https://link.springer.com/book/10.1007/978-3-319-11080-6)
