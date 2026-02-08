---
{"dg-publish":true,"permalink":"/30-knowledge/stats/01-foundations/linear-transformations/","tags":["probability","foundations"]}
---

## Definition

> [!abstract] Core Statement
> A **linear transformation** $T: \mathbb{R}^n \to \mathbb{R}^m$ is a function that preserves vector addition and scalar multiplication:
> $$T(\alpha x + \beta y) = \alpha T(x) + \beta T(y)$$

Every linear transformation can be represented as matrix multiplication: $T(x) = Ax$.

---

## Purpose

1.  Describe **geometric operations** (rotation, scaling, projection).
2.  Foundation for understanding **neural network layers**.
3.  Analyze **dimensionality changes** in feature transformations.
4.  Understand **eigendecomposition** and **SVD**.

---

## Key Concepts

### Properties

| Property | Description |
|----------|-------------|
| $T(0) = 0$ | Origin always maps to origin. |
| **Linearity** | $T(x+y) = T(x) + T(y)$, $T(cx) = cT(x)$ |
| **Composition** | $(T_2 \circ T_1)(x) = T_2(T_1(x)) = A_2 A_1 x$ |

### Kernel and Image

| Concept | Definition | Meaning |
|---------|------------|---------|
| **Kernel** (null space) | $\ker(T) = \{x : T(x) = 0\}$ | Vectors collapsed to zero. |
| **Image** (range) | $\text{Im}(T) = \{T(x) : x \in \mathbb{R}^n\}$ | All possible outputs. |
| **Rank** | $\dim(\text{Im}(T))$ | Dimensions preserved. |
| **Nullity** | $\dim(\ker(T))$ | Dimensions lost. |

### Rank-Nullity Theorem

$$\text{Rank}(T) + \text{Nullity}(T) = n$$

---

## Geometric Transformations (2D)

| Transformation | Matrix | Effect |
|----------------|--------|--------|
| **Scaling** | $\begin{pmatrix} s_x & 0 \\ 0 & s_y \end{pmatrix}$ | Stretch by $s_x$, $s_y$ |
| **Rotation** (θ) | $\begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$ | Rotate CCW |
| **Reflection** (x-axis) | $\begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$ | Flip over x-axis |
| **Shear** | $\begin{pmatrix} 1 & k \\ 0 & 1 \end{pmatrix}$ | Slant horizontally |
| **Projection** (x-axis) | $\begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}$ | Collapse to x-axis |

---

## Worked Example

> [!example] Problem
> Apply a 90° counterclockwise rotation to the point $(3, 1)$.

**Solution:**

1. **Rotation matrix** ($\theta = 90° = \frac{\pi}{2}$):
$$R = \begin{pmatrix} \cos 90° & -\sin 90° \\ \sin 90° & \cos 90° \end{pmatrix} = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}$$

2. **Apply transformation:**
$$R \begin{pmatrix} 3 \\ 1 \end{pmatrix} = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix} \begin{pmatrix} 3 \\ 1 \end{pmatrix} = \begin{pmatrix} -1 \\ 3 \end{pmatrix}$$

**Answer:** $(3, 1) \to (-1, 3)$

---

## Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt

# Define transformations
def rotation_matrix(theta_degrees):
    theta = np.radians(theta_degrees)
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

def scaling_matrix(sx, sy):
    return np.array([[sx, 0], [0, sy]])

def shear_matrix(kx, ky=0):
    return np.array([[1, kx], [ky, 1]])

# Apply transformation to points
def transform(T, points):
    """Apply transformation matrix T to array of points."""
    return (T @ points.T).T

# Example: transform a square
square = np.array([[0,0], [1,0], [1,1], [0,1], [0,0]])

# Compose transformations: rotate 45°, then scale
T = scaling_matrix(1.5, 0.5) @ rotation_matrix(45)
transformed = transform(T, square)

# Visualize
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.plot(square[:,0], square[:,1], 'b-o', label='Original')
plt.axis('equal'); plt.grid(); plt.title('Original')

plt.subplot(1, 2, 2)
plt.plot(transformed[:,0], transformed[:,1], 'r-o', label='Transformed')
plt.axis('equal'); plt.grid(); plt.title('Rotated + Scaled')
plt.tight_layout()
plt.show()

# Kernel and Image
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
rank = np.linalg.matrix_rank(A)
nullity = A.shape[1] - rank
print(f"Rank: {rank}, Nullity: {nullity}")  # Rank: 2, Nullity: 1
```

---

## R Implementation

```r
# Rotation matrix
rotation_matrix <- function(theta_deg) {
  theta <- theta_deg * pi / 180
  matrix(c(cos(theta), sin(theta), -sin(theta), cos(theta)), nrow = 2)
}

# Apply transformation
point <- c(3, 1)
R <- rotation_matrix(90)
transformed <- R %*% point
print(transformed)  # (-1, 3)

# Visualize
library(ggplot2)
square <- data.frame(x = c(0, 1, 1, 0, 0), y = c(0, 0, 1, 1, 0))
T <- rotation_matrix(45)
transformed_sq <- as.data.frame(t(T %*% t(as.matrix(square))))
names(transformed_sq) <- c("x", "y")

ggplot() +
  geom_polygon(data = square, aes(x, y), fill = "blue", alpha = 0.3) +
  geom_polygon(data = transformed_sq, aes(x, y), fill = "red", alpha = 0.3) +
  coord_fixed() + theme_minimal()
```

---

## ML Applications

| Application | Role of Linear Transformations |
|-------------|-------------------------------|
| **Neural Networks** | Each layer: $h = \sigma(Wx + b)$, where $Wx$ is linear. |
| **PCA** | Project to subspace via eigenvector matrix. |
| **Whitening** | Transform data to identity covariance. |
| **Data Augmentation** | Rotation, scaling for image augmentation. |
| **Embeddings** | Linear projection to lower dimensions. |

---

## Related Concepts

- [[30_Knowledge/Stats/01_Foundations/Eigenvalues & Eigenvectors\|Eigenvalues & Eigenvectors]] - Special vectors preserved by transformation.
- [[30_Knowledge/Stats/01_Foundations/Matrix Multiplication\|Matrix Multiplication]] - How transformations are applied.
- [[30_Knowledge/Stats/05_Unsupervised_Learning/PCA (Principal Component Analysis)\|PCA (Principal Component Analysis)]] - Projection to principal subspace.
- [[30_Knowledge/Stats/01_Foundations/Singular Value Decomposition (SVD)\|Singular Value Decomposition (SVD)]] - Decompose any transformation.

---

## When to Use

> [!success] Use Linear Transformations When...
> - Refer to standard documentation
> - Refer to standard documentation

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Assumptions are violated
> - Alternative methods are more appropriate

---

## References

- **Book:** Strang, G. (2016). *Introduction to Linear Algebra* (5th ed.). Chapters 6-8.
- **Book:** 3Blue1Brown. (2016). *Essence of Linear Algebra* [Video Series]. YouTube.
