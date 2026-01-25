---
{"dg-publish":true,"permalink":"/stats/03-regression-analysis/lasso-regression/","tags":["Regression","Regularization","Feature-Selection"]}
---


## Definition

> [!abstract] Overview
> **Lasso Regression** (L1 Regularization) adds a penalty term proportional to the **absolute value** of the coefficients.

$$ \min \sum (y - \hat{y})^2 + \lambda \sum |\beta| $$

**Key Feature:** Lasso can shrink coefficients completely to **ZERO**. Thus, it performs **Feature Selection**.

---

## 1. Ridge vs Lasso

| Feature | Ridge (L2) | Lasso (L1) |
|---------|------------|------------|
| **Penalty** | Square ($\beta^2$) | Absolute ($|\beta|$) |
| **Result** | Small coeffs, none zero. | Many zero coeffs (Sparse). |
| **Use Case** | Multicollinearity. | Feature Selection. |

---

## 2. Python Implementation

```python
from sklearn.linear_model import Lasso

# High alpha = stronger penalty = more features dropped
model = Lasso(alpha=0.1)
model.fit(X, y)

print(f"Zero Coefficients: {sum(model.coef_ == 0)}")
```

---

## Related Concepts

- [[stats/03_Regression_Analysis/Ridge Regression\|Ridge Regression]]
- [[stats/01_Foundations/Feature Selection\|Feature Selection]]
