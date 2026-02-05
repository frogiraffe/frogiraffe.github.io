---
{"dg-publish":true,"permalink":"/stats/03-regression-analysis/polynomial-regression/","tags":["probability","regression","non-linear"]}
---


## Definition

> [!abstract] Core Statement
> **Polynomial Regression** extends linear regression by adding ==powers of the independent variable== as predictors, allowing the model to fit curved relationships.

$$
y = \beta_0 + \beta_1 x + \beta_2 x^2 + \ldots + \beta_d x^d + \epsilon
$$

---

> [!tip] Key Insight
> It's still a **linear** model (linear in parameters), just non-linear in features.

---

## Python Implementation

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import numpy as np

# ========== CREATE POLYNOMIAL FEATURES ==========
poly = PolynomialFeatures(degree=3, include_bias=False)
X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)

# ========== PIPELINE ==========
pipe = Pipeline([
    ('poly', PolynomialFeatures(degree=3)),
    ('linear', LinearRegression())
])
pipe.fit(X, y)
y_pred = pipe.predict(X)
```

---

## Choosing Degree

| Degree | Risk |
|--------|------|
| Too low | Underfitting |
| Too high | Overfitting |

Use **cross-validation** to choose optimal degree.

---

## Common Pitfalls

> [!warning] Overfitting
> High-degree polynomials oscillate wildly outside training data. Use [[stats/03_Regression_Analysis/Regularization\|Regularization]] or [[GAMs\|GAMs]] instead.

---

## Related Concepts

- [[stats/03_Regression_Analysis/Simple Linear Regression\|Simple Linear Regression]] — Degree 1
- [[stats/03_Regression_Analysis/Generalized Additive Models\|Generalized Additive Models]] — Flexible alternative
- [[stats/03_Regression_Analysis/Regularization\|Regularization]] — Prevents overfitting

---

## References

- **Book:** James, G., et al. (2013). *An Introduction to Statistical Learning*. Chapter 7.
