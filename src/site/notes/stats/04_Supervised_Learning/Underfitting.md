---
{"dg-publish":true,"permalink":"/stats/04-supervised-learning/underfitting/","tags":["Machine-Learning","Model-Evaluation","Bias-Variance"]}
---


## Definition

> [!abstract] Core Statement
> **Underfitting** occurs when a model is ==too simple to capture the underlying patterns== in the data. It results in high error on both training and test sets.

---

> [!tip] Intuition (ELI5): The Lazy Student
> An underfitted model is like a student who memorizes one simple rule: "always pick C." They perform poorly on everything — practice tests AND real tests — because they haven't actually learned anything.

---

## Signs of Underfitting

| Indicator | What You See |
|-----------|--------------|
| **Training Error** | High |
| **Validation Error** | High (similar to training) |
| **Learning Curve** | Both curves plateau at high error |
| **Bias-Variance** | High bias, low variance |

---

## Causes

1. **Model too simple** — Linear model for non-linear data
2. **Insufficient features** — Missing important predictors
3. **Too much regularization** — L1/L2 penalty too strong
4. **Not enough training** — Too few iterations/epochs
5. **Wrong algorithm** — Linear regression for image data

---

## Solutions

| Solution | Implementation |
|----------|----------------|
| **Increase complexity** | More layers, more trees, polynomial features |
| **Add features** | [[stats/04_Supervised_Learning/Feature Engineering\|Feature Engineering]] |
| **Reduce regularization** | Lower L1/L2 penalty, lower dropout |
| **Train longer** | More epochs, more iterations |
| **Use better algorithm** | Gradient boosting instead of linear |

---

## Python Detection

```python
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# ========== GENERATE NON-LINEAR DATA ==========
np.random.seed(42)
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, 100)

# ========== UNDERFITTING: LINEAR MODEL ==========
linear = LinearRegression()
linear.fit(X, y)
y_pred_linear = linear.predict(X)

train_mse_linear = mean_squared_error(y, y_pred_linear)
print(f"Linear Model MSE: {train_mse_linear:.4f}")  # High = underfit

# ========== BETTER FIT: POLYNOMIAL ==========
poly = PolynomialFeatures(degree=5)
X_poly = poly.fit_transform(X)
poly_model = LinearRegression()
poly_model.fit(X_poly, y)
y_pred_poly = poly_model.predict(X_poly)

train_mse_poly = mean_squared_error(y, y_pred_poly)
print(f"Polynomial Model MSE: {train_mse_poly:.4f}")  # Much lower

# ========== VISUALIZATION ==========
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.scatter(X, y, alpha=0.5, label='Data')
plt.plot(X, y_pred_linear, 'r-', linewidth=2, label='Linear (Underfit)')
plt.title(f'Underfitting (MSE: {train_mse_linear:.3f})')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(X, y, alpha=0.5, label='Data')
plt.plot(X, y_pred_poly, 'g-', linewidth=2, label='Polynomial')
plt.title(f'Good Fit (MSE: {train_mse_poly:.3f})')
plt.legend()

plt.tight_layout()
plt.show()
```

---

## Underfitting vs Overfitting

| Aspect | Underfitting | Overfitting |
|--------|--------------|-------------|
| **Training Error** | High | Low |
| **Test Error** | High | High |
| **Gap** | Small | Large |
| **Bias** | High | Low |
| **Variance** | Low | High |
| **Fix** | More complexity | More regularization |

---

## Related Concepts

- [[stats/04_Supervised_Learning/Overfitting\|Overfitting]] — Opposite problem
- [[stats/04_Supervised_Learning/Learning Curves\|Learning Curves]] — Diagnosis tool
- [[stats/04_Supervised_Learning/Bias-Variance Tradeoff\|Bias-Variance Tradeoff]] — Theoretical framework
- [[stats/03_Regression_Analysis/Regularization\|Regularization]] — Can cause underfitting if too strong

---

## References

- **Book:** Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Chapter 7.
