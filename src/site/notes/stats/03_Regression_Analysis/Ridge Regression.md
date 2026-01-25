---
{"dg-publish":true,"permalink":"/stats/03-regression-analysis/ridge-regression/","tags":["Regression","Regularization"]}
---


## Definition

> [!abstract] Overview
> **Ridge Regression** (L2 Regularization) modifies the Least Squares objective function by adding a penalty term proportional to the square of the magnitude of coefficients.

$$ \min \sum (y - \hat{y})^2 + \lambda \sum \beta^2 $$

- **Goal:** Shrink coefficients to prevent overfitting.
- **Multicollinearity:** Ridge handles highly correlated features well by shrinking them together.

---

## 1. When to Use?

- Many features ($p > n$).
- High Multicollinearity.
- Overfitting (High Variance).

---

## 2. Python Implementation

```python
from sklearn.linear_model import Ridge
from sklearn.datasets import make_regression

X, y = make_regression(n_features=10, noise=0.1)

# alpha is lambda (penalty strength)
model = Ridge(alpha=1.0)
model.fit(X, y)

print(f"Coefficients: {model.coef_}")
```

---

## Related Concepts

- [[stats/03_Regression_Analysis/Lasso Regression\|Lasso Regression]] (L1)
- [[stats/01_Foundations/Elastic Net\|Elastic Net]] (L1 + L2)
- [[Bias-Variance Tradeoff\|Bias-Variance Tradeoff]]