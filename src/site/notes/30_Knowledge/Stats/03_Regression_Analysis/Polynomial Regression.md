---
{"dg-publish":true,"permalink":"/30-knowledge/stats/03-regression-analysis/polynomial-regression/","tags":["regression","modeling"]}
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
> High-degree polynomials oscillate wildly outside training data. Use [[30_Knowledge/Stats/03_Regression_Analysis/Regularization\|Regularization]] or GAMs instead.

---

## Related Concepts

- [[30_Knowledge/Stats/03_Regression_Analysis/Simple Linear Regression\|Simple Linear Regression]] — Degree 1
- [[30_Knowledge/Stats/03_Regression_Analysis/Generalized Additive Models\|Generalized Additive Models]] — Flexible alternative
- [[30_Knowledge/Stats/03_Regression_Analysis/Regularization\|Regularization]] — Prevents overfitting

---

## When to Use

> [!success] Use Polynomial Regression When...
> - Refer to standard documentation
> - Refer to standard documentation

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Relationship is highly non-linear
> - Severe multicollinearity exists

---

## R Implementation

```r
# Polynomial Regression in R
set.seed(42)

# Sample data
df <- data.frame(
  x1 = rnorm(100),
  x2 = rnorm(100)
)
df$y <- 2 + 3*df$x1 + 1.5*df$x2 + rnorm(100)

# Fit model
model <- lm(y ~ x1 + x2, data = df)
summary(model)
```

---

## References

- **Book:** James, G., et al. (2013). *An Introduction to Statistical Learning*. Chapter 7.
