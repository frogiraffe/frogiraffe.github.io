---
{"dg-publish":true,"permalink":"/stats/04-supervised-learning/support-vector-machines/","tags":["Machine-Learning","Classification","Kernel-Methods"]}
---


## Definition

> [!abstract] Core Statement
> **Support Vector Machines** find the ==hyperplane that maximizes the margin== between classes. Points on the margin boundary are called "support vectors."

---

> [!tip] Intuition (ELI5)
> Draw a line (or surface) between two groups of points. SVM finds the line that has the most "breathing room" on both sides.

---

## Key Concepts

| Concept | Description |
|---------|-------------|
| **Margin** | Distance from hyperplane to nearest points |
| **Support Vectors** | Points exactly on the margin |
| **C parameter** | Trade-off between margin and misclassification |
| **Kernel** | For non-linear boundaries |

---

## Python Implementation

```python
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ========== PREPROCESSING (ESSENTIAL!) ==========
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ========== LINEAR SVM ==========
svm_linear = SVC(kernel='linear', C=1.0)
svm_linear.fit(X_train_scaled, y_train)

# ========== RBF KERNEL SVM ==========
svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_rbf.fit(X_train_scaled, y_train)

print(f"Accuracy: {svm_rbf.score(X_test_scaled, y_test):.2%}")
print(f"Support Vectors: {len(svm_rbf.support_vectors_)}")
```

---

## R Implementation

```r
library(e1071)

svm_model <- svm(y ~ ., data = train_data, kernel = "radial", cost = 1)
predictions <- predict(svm_model, test_data)
```

---

## Hyperparameters

| Parameter | Effect |
|-----------|--------|
| **C** (low) | Large margin, more misclassifications |
| **C** (high) | Small margin, fewer misclassifications |
| **γ (gamma)** | High = complex boundary, overfit risk |

---

## Kernel Comparison

| Kernel | Formula | Use |
|--------|---------|-----|
| Linear | $x^Ty$ | Linearly separable |
| RBF | $\exp(-\gamma\|x-y\|^2)$ | Most common |
| Polynomial | $(x^Ty + c)^d$ | Known polynomial relationship |

---

## Common Pitfalls

> [!warning] Traps
>
> **1. Not Scaling Data**
> - SVM is distance-based → features must be scaled!
>
> **2. Large Datasets**
> - Training is $O(n^3)$ → slow for >10k samples
> - Use LinearSVC or SGDClassifier instead

---

## Related Concepts

- [[stats/04_Supervised_Learning/Kernel Methods\|Kernel Methods]] — Theory behind kernels
- [[stats/01_Foundations/Feature Scaling\|Feature Scaling]] — Essential preprocessing
- [[stats/03_Regression_Analysis/Logistic Regression\|Logistic Regression]] — Simpler alternative

---

## References

- **Book:** Hastie, T., et al. (2009). *The Elements of Statistical Learning*. Chapter 12.
- **Paper:** Cortes, C., & Vapnik, V. (1995). Support-vector networks. *Machine Learning*, 20(3), 273-297.
