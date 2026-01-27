---
{"dg-publish":true,"permalink":"/stats/03-regression-analysis/group-lasso/","tags":["Regularization","Machine-Learning","Feature-Selection"]}
---


## Definition

> [!abstract] Core Statement
> **Group Lasso** extends LASSO to select ==groups of features== together (all-in or all-out), useful when features have natural groupings.

$$
\min_\beta \left\{ \frac{1}{2n}||y - X\beta||_2^2 + \lambda \sum_{g=1}^{G} \sqrt{p_g}||\beta_g||_2 \right\}
$$

---

## When to Use

| Scenario | Example |
|----------|---------|
| **Categorical one-hot** | All dummies for a variable |
| **Gene groups** | Pathway-level selection |
| **Polynomial features** | All terms for one variable |

---

## Python Implementation

```python
from group_lasso import GroupLasso

# Define groups (e.g., features 0-2 = group 0, features 3-5 = group 1)
groups = np.array([0, 0, 0, 1, 1, 1, 2, 2])

model = GroupLasso(
    groups=groups,
    group_reg=0.05,
    l1_reg=0.0
)
model.fit(X, y)

# Entire groups are zeroed out together
print("Coefficients:", model.coef_)
```

---

## Related Concepts

- [[LASSO (L1 Regularization)\|LASSO (L1 Regularization)]] — Individual feature selection
- [[stats/03_Regression_Analysis/Ridge Regression\|Ridge Regression]] — L2 regularization
- [[stats/01_Foundations/Elastic Net\|Elastic Net]] — L1 + L2

---

## References

- **Paper:** Yuan, M., & Lin, Y. (2006). Model selection and estimation in regression with grouped variables. *JRSS-B*, 68(1), 49-67.
