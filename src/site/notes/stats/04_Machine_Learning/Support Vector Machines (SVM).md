---
{"dg-publish":true,"permalink":"/stats/04-machine-learning/support-vector-machines-svm/","tags":["Machine-Learning","Supervised-Learning","Classification","Regression"]}
---


# Support Vector Machines (SVM)

## Definition

> [!abstract] Core Statement
> **Support Vector Machines (SVM)** are supervised learning models used for classification and regression. The core idea is to find the **hyperplane** that best separates the two classes with the ==maximum margin== (widest gap).
> The **"Support Vectors"** are the data points closest to the hyperplane that define the margin.

---

## Purpose

1.  **High-Dimensional Classification:** Effective when number of dimensions > number of samples.
2.  **Complex Boundaries:** Can model non-linear boundaries using the **Kernel Trick**.
3.  **Robustness:** Because it relies only on the support vectors (the "hard" cases), it is resistant to outliers far from the boundary.

---

## When to Use

> [!success] Use SVM When...
> - **Small to Medium datasets** (it doesn't scale well to millions of rows).
> - **High dimensionality** (e.g., Text Classification, Gene Expression).
> - You need a **robust** boundary.

> [!failure] Limitations
> - **Slow Training:** $O(N^3)$ complexity makes it painfully slow for large $N$.
> - **Noise Sensitivity:** If classes overlap heavily, the margin concept starts to fail (requires soft margin).
> - **No Probability:** Outputs are distances, not probabilities (unlike Logistic Regression). Requires `probability=True` (Platt scaling) which is slow.

---

## Theoretical Background

### The Margin

-   **Hyperplane:** A line (in 2D) or flat surface (in 3D) separating classes.
-   **Margin:** The distance between the hyperplane and the nearest points (Support Vectors).
-   **Goal:** Maximize the Margin. (A wide road is safer than a narrow one).

### The Kernel Trick

What if data isn't linearly separable? (e.g., a red circle inside a blue ring).
-   **Idea:** Project data into a **Higher Dimension** where it *is* separable.
-   **Kernel:** A math function that computes dot products in this high dimension without actually transforming the data (Computational shortcut).
    -   *Linear Kernel:* Standard straight line.
    -   *RBF (Radial Basis Function):* Creates circular/blobby boundaries. Infinite dimensions.
    -   *Polynomial:* Curved boundaries.

---

## Worked Example: 1D Separation

> [!example] Problem
> **Data:** Red dots at $x = [-3, -2]$. Blue dots at $x = [2, 3]$.
> **Goal:** Separate them.
> 
> **Linear SVM:**
> -   Hyperplane (Point): $x = 0$.
> -   Margin: Distance from 0 to -2 is 2. Distance from 0 to 2 is 2. Margin = 4.
> 
> **Non-Linear Data:**
> -   Red: $[-3, 3]$. Blue: $[-1, 1]$. (Blue is sandwiched).
> -   **Problem:** No single point separates them.
> -   **Kernel Trick:** Map $x \to x^2$.
>     -   Red: $9, 9$. Blue: $1, 1$.
>     -   Now separable! Cut at $x^2 = 5$.

---

## Python Implementation

```python
from sklearn.svm import SVC
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt

# complex non-linear data
X, y = make_circles(noise=0.1, factor=0.5, random_state=42)

# 1. Linear Kernel (Fails on circles)
# clf = SVC(kernel='linear') 

# 2. RBF Kernel (Works!)
clf = SVC(kernel='rbf', C=1.0, gamma='scale')
clf.fit(X, y)

# Visualization code (pseudo)
# plot_decision_boundary(clf, X, y)
# Result: A circular boundary separating the inner class.
```

---

## Key Parameters

| Parameter | Meaning | Effect of Increasing |
|-----------|---------|---------------------|
| **C** (Regularization) | Penalty for misclassification. | **High C:** Strict (Hard margin, fit training well, risk overfitting). <br> **Low C:** Loose (Soft margin, wider road, better generalization). |
| **Gamma** (Kernel width) | How far the influence of a single point reaches. | **High Gamma:** Points are isolated islands (Overfitting). <br> **Low Gamma:** Broad, smooth blobs (Underfitting). |

---

## Related Concepts

- [[stats/03_Regression_Analysis/Logistic Regression\|Logistic Regression]] - Alternative linear classifier (Probs).
- [[Kernel Density Estimation\|Kernel Density Estimation]] - Related to RBF.
- [[stats/04_Machine_Learning/Overfitting & Underfitting\|Overfitting & Underfitting]] - Controlled by C and Gamma.
