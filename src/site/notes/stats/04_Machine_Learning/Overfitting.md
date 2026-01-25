---
{"dg-publish":true,"permalink":"/stats/04-machine-learning/overfitting/","tags":["Machine-Learning","Theory","Bias-Variance"]}
---


## Definition

> [!abstract] Overview
> **Overfitting** occurs when a model learns the "noise" in the training data rather than the underlying "signal." The model performs exceptionally well on training data (memorization) but fails to generalize to new, unseen data.

- **High Variance:** Sensitive to small fluctuations in the training set.
- **Low Bias:** Captures the training data complexity perfectly (too perfectly).

**Analogy:** A student who memorizes the textbook answers verbatim but fails the exam because the questions are slightly rephrased.

---

## 1. Symptoms

| Metric | Underfitting (High Bias) | Overfitting (High Variance) | Good Fit |
|--------|--------------------------|-----------------------------|----------|
| **Training Error** | High | Low | Low |
| **Testing Error** | High | High | Low |

---

## 2. Bias-Variance Tradeoff

As model complexity increases (e.g., higher degree polynomial, deeper decision tree):
- **Bias decreases** (Better fit to training data).
- **Variance increases** (Worse generalization).

The goal is the **Sweet Spot** where Total Error (Bias + Variance + Noise) is minimized.

---

## 3. Prevention Techniques

1.  **More Data:** Helps the model distinguish signal from noise.
2.  **Regularization:** Penalizing complexity.
    - [[stats/03_Regression_Analysis/Ridge Regression\|Ridge Regression]] (L2)
    - [[stats/03_Regression_Analysis/Lasso Regression\|Lasso Regression]] (L1)
3.  **Cross-Validation:** Using [[stats/04_Machine_Learning/Cross-Validation\|Cross-Validation]] (k-fold) to tune hyperparameters.
4.  **Pruning:** For Decision Trees (limiting depth).
5.  **Dropout:** For Neural Networks (randomly ignoring neurons).

---

## 4. Python Example (Polynomial Regression)

```python
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

# High Degree = Overfitting
model = Pipeline([
    ("poly", PolynomialFeatures(degree=15)), # Too complex!
    ("linear", LinearRegression())
])

# Scoring (Negative MSE)
scores = cross_val_score(model, X, y, cv=5, scoring="neg_mean_squared_error")
print(f"Mean Error: {-scores.mean():.2f} (High Error likely indicates overfitting)")
```

---

## Related Concepts

- [[Underfitting\|Underfitting]] - The opposite problem (High Bias).
- [[stats/03_Regression_Analysis/Regularization\|Regularization]] - The solution.
- [[stats/01_Foundations/Feature Selection\|Feature Selection]] - Removing irrelevant features reduces noise.
