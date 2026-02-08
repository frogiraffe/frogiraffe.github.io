---
{"dg-publish":true,"permalink":"/30-knowledge/stats/03-regression-analysis/local-linear-regression/","tags":["regression","modeling"]}
---


## Definition

> [!abstract] Overview
> **Local Linear Regression (LLR)** is a non-parametric method that fits a linear regression model within a local window around each point of interest. Instead of assuming a global linear relationship, it allows the relationship to vary locally, effectively smoothing the data.
>
> It solves the following optimization problem for each target point $x_0$:
> $$ \min_{\alpha, \beta} \sum_{i=1}^{n} w_i(x_0) (y_i - \alpha - \beta(x_i - x_0))^2 $$
> where $w_i(x_0)$ is a **kernel weight** that gives more importance to points close to $x_0$.

---

## Key Components

1.  **Kernel Function ($K$):** Determines the shape of the weights (e.g., Gaussian, Epanechnikov).
2.  **Bandwidth ($h$):** Controls the size of the local window.
    *   **Small $h$:** High variance, low bias (overfitting, wiggly curve).
    *   **Large $h$:** Low variance, high bias (underfitting, overly smooth).

---

## Python Implementation

```python
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(42)
x = np.linspace(0, 10, 100)
y = np.sin(x) + np.random.normal(0, 0.2, 100)

# Local Linear Regression using Lowess (Locally Weighted Scatterplot Smoothing)
# frac: The fraction of the data used when estimating each y-value (bandwidth proxy)
lowess = sm.nonparametric.lowess
z = lowess(y, x, frac=0.2)

# Plot
plt.scatter(x, y, label='Data', alpha=0.5)
plt.plot(z[:, 0], z[:, 1], color='red', label='Local Linear Regression (Lowess)')
plt.legend()
plt.title("Local Linear Regression with Lowess")
plt.show()
```

---

## Related Concepts

- [[30_Knowledge/Stats/03_Regression_Analysis/Simple Linear Regression\|Simple Linear Regression]] - The global counterpart.
- [[30_Knowledge/Stats/01_Foundations/Kernel Density Estimation\|Kernel Density Estimation]] - Uses kernels for density instead of regression.
- [[30_Knowledge/Stats/08_Time_Series_Analysis/Smoothing\|Smoothing]] - The general category LLR belongs to.

## When to Use

> [!success] Use Local Linear Regression When...
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
# Local Linear Regression in R
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

1. See related concepts for further reading
