---
{"dg-publish":true,"permalink":"/stats/04-machine-learning/support-vector-machines-svm/","tags":["Machine-Learning","Classification","Algorithms"]}
---


## Definition

> [!abstract] Overview
> **Support Vector Machines (SVM)** are supervised learning models used for binary classification. The goal is to find a hyperplane that best separates the two classes.

- **Hyperplane:** The decision boundary.
- **Margin:** The distance between the hyperplane and the nearest data point (Support Vector). SVM maximizes this margin.

---

## 1. Key Concepts

### Kernel Trick
Data that is not linearly separable in 2D might be separable in 3D. The **Kernel Trick** calculates relationships in higher-dimensional space without actually transforming the data (computationally cheap).
- **RBF (Radial Basis Function):** Most common (Infinite dimensions).
- **Polynomial:** $K(x, y) = (x^T y + c)^d$.

### C (Regularization)
- **High C:** Lower bias, high variance. Punishes misclassification errors heavily (Risk of overfitting).
- **Low C:** High bias, low variance. Allows some misclassification for a wider margin (Smoother boundary).

---

## 2. Python Implementation

```python
from sklearn.svm import SVC
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

# Non-linear data
X, y = make_moons(n_samples=100, noise=0.1)

# Fit SVM with RBF Kernel
model = SVC(kernel='rbf', C=1.0, gamma='scale')
model.fit(X, y)

print(f"Accuracy: {model.score(X, y):.2f}")
```

---

## Related Concepts

- [[stats/03_Regression_Analysis/Logistic Regression\|Logistic Regression]] (Linear Alternative)
- [[stats/04_Machine_Learning/Overfitting\|Overfitting]] (Controlled by C)
- [[stats/01_Foundations/Feature Scaling\|Feature Scaling]] (Critical for SVM because of distance calculation)
