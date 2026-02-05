---
{"dg-publish":true,"permalink":"/stats/03-regression-analysis/adaptive-lasso/","tags":["probability","regularization","machine-learning","feature-selection"]}
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

- [[LASSO (L1 Regularization)\|LASSO (L1 Regularization)]] — Standard version
- [[stats/01_Foundations/Elastic Net\|Elastic Net]] — L1 + L2
- [[stats/03_Regression_Analysis/Group Lasso\|Group Lasso]] — Group selection

---

## References

- **Paper:** Zou, H. (2006). The adaptive lasso and its oracle properties. *JASA*, 101(476), 1418-1429.
