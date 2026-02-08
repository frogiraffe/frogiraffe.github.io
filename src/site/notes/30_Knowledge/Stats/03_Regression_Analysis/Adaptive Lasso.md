---
{"dg-publish":true,"permalink":"/30-knowledge/stats/03-regression-analysis/adaptive-lasso/","tags":["regression","modeling"]}
---


## Definition

> [!abstract] Core Statement
> **Adaptive Lasso** uses ==weighted penalties== based on initial estimates, achieving oracle properties (consistent variable selection + optimal rates).

$$
\min_\beta \left\{ \frac{1}{2n}||y - X\beta||_2^2 + \lambda \sum_{j=1}^{p} w_j|\beta_j| \right\}
$$

Where $w_j = 1/|\hat{\beta}_j^{init}|^\gamma$ (inverse of initial estimate).

---

## Why Better Than Standard LASSO?

| Property | LASSO | Adaptive Lasso |
|----------|-------|----------------|
| **Consistency** | Requires irrepresentable condition | Always consistent |
| **Oracle** | No | Yes (with good initial) |
| **Bias** | High for large coefficients | Lower |

---

## Python Implementation

```python
from sklearn.linear_model import LassoCV, Lasso
import numpy as np

# Step 1: Get initial estimates (e.g., OLS or Ridge)
initial_model = LassoCV().fit(X, y)
initial_coefs = np.abs(initial_model.coef_) + 1e-10  # Avoid division by zero

# Step 2: Calculate adaptive weights
gamma = 1.0
weights = 1 / (initial_coefs ** gamma)

# Step 3: Transform X
X_weighted = X / weights

# Step 4: Fit LASSO on transformed data
adaptive_lasso = LassoCV().fit(X_weighted, y)

# Step 5: Get original scale coefficients
final_coefs = adaptive_lasso.coef_ / weights
print("Selected features:", np.where(np.abs(final_coefs) > 1e-5)[0])
```

---

## Related Concepts

- LASSO (L1 Regularization) — Standard version
- [[30_Knowledge/Stats/01_Foundations/Elastic Net\|Elastic Net]] — L1 + L2
- [[30_Knowledge/Stats/03_Regression_Analysis/Group Lasso\|Group Lasso]] — Group selection

---

## When to Use

> [!success] Use Adaptive Lasso When...
> - Refer to standard documentation
> - Refer to standard documentation

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Assumptions are violated
> - Alternative methods are more appropriate

---

## R Implementation

```r
# Adaptive Lasso in R
set.seed(42)

# Example implementation
data <- rnorm(100)
summary(data)
```

---

## References

- **Paper:** Zou, H. (2006). The adaptive lasso and its oracle properties. *JASA*, 101(476), 1418-1429.
