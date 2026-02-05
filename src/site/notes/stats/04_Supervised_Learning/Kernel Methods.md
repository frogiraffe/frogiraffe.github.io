---
{"dg-publish":true,"permalink":"/stats/04-supervised-learning/kernel-methods/","tags":["probability","machine-learning","non-linear","svm"]}
---


## Definition

> [!abstract] Core Statement
> **Kernel Methods** use the ==kernel trick== to implicitly map data to high-dimensional feature spaces, enabling non-linear decision boundaries while computing only inner products.

---

> [!tip] Intuition (ELI5)
> Data that's not separable by a line in 2D might be separable by a plane if lifted to 3D. Kernels do this lifting mathematically without actually computing the high-dimensional coordinates.

---

## The Kernel Trick

Instead of computing $\phi(x)^T\phi(y)$ in high-dimensional space:
$$
K(x, y) = \phi(x)^T\phi(y)
$$

Compute directly using kernel function!

---

## Common Kernels

| Kernel | Formula | Use |
|--------|---------|-----|
| **Linear** | $K(x,y) = x^Ty$ | Linearly separable |
| **Polynomial** | $(x^Ty + c)^d$ | Low-degree non-linearity |
| **RBF/Gaussian** | $\exp(-\gamma\|x-y\|^2)$ | Most common, flexible |
| **Sigmoid** | $\tanh(\alpha x^Ty + c)$ | Neural network-like |

---

## Python Implementation

```python
from sklearn.svm import SVC
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

# Non-linear data
X, y = make_moons(n_samples=200, noise=0.1)

# ========== LINEAR KERNEL (FAILS) ==========
svm_linear = SVC(kernel='linear')
svm_linear.fit(X, y)
print(f"Linear accuracy: {svm_linear.score(X, y):.2%}")

# ========== RBF KERNEL (WORKS) ==========
svm_rbf = SVC(kernel='rbf', gamma='scale')
svm_rbf.fit(X, y)
print(f"RBF accuracy: {svm_rbf.score(X, y):.2%}")

# ========== CUSTOM KERNEL ==========
def my_kernel(X, Y):
    return X @ Y.T  # Linear kernel manually

svm_custom = SVC(kernel=my_kernel)
```

---

## Kernel Parameters

| Kernel | Parameter | Effect |
|--------|-----------|--------|
| **RBF** | γ (gamma) | Higher = more complex boundary |
| **Polynomial** | d (degree) | Higher = more flexible |
| **Polynomial** | c | Influence of higher-order terms |

---

## Kernel Matrix (Gram Matrix)

$$
K_{ij} = K(x_i, x_j)
$$

Must be **positive semi-definite** for valid kernel.

---

## Applications

| Algorithm | Kernel Version |
|-----------|----------------|
| SVM | Kernel SVM (most common) |
| PCA | Kernel PCA |
| Ridge Regression | Kernel Ridge |
| K-Means | Kernel K-Means |

---

## Common Pitfalls

> [!warning] Traps
>
> **1. Wrong γ for RBF**
> - Too high → overfitting, spiky boundaries
> - Too low → underfitting, ignores local structure
>
> **2. Scaling**
> - RBF is sensitive to feature scales
> - Always standardize first!

---

## Related Concepts

- [[Support Vector Machines\|Support Vector Machines]] — Primary application
- [[stats/04_Supervised_Learning/Gaussian Processes\|Gaussian Processes]] — Kernel-based probabilistic model
- [[stats/01_Foundations/Feature Scaling\|Feature Scaling]] — Essential preprocessing

---

## References

- **Book:** Schölkopf, B., & Smola, A. J. (2002). *Learning with Kernels*. MIT Press.
- **Paper:** Shawe-Taylor, J., & Cristianini, N. (2004). *Kernel Methods for Pattern Analysis*. Cambridge.
