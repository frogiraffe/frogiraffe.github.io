---
{"dg-publish":true,"permalink":"/stats/01-foundations/optimization/","tags":["Machine-Learning","Mathematics","Gradient-Descent"]}
---


## Definition

> [!abstract] Core Statement
> **Optimization** is the process of ==finding parameter values that minimize (or maximize) an objective function==, typically a loss function in machine learning.
> $$\theta^* = \arg\min_\theta L(\theta)$$

---

## Common Methods

| Method | Type | Use Case |
|--------|------|----------|
| **Gradient Descent** | First-order | Deep learning |
| **Newton's Method** | Second-order | Convex, small n |
| **Adam** | Adaptive | Default for DL |
| **L-BFGS** | Quasi-Newton | Classical ML |
| **Grid Search** | Brute force | Hyperparameters |

---

## Gradient Descent Update

$$\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$$

Where η = learning rate.

---

## Python Implementation

```python
from scipy.optimize import minimize

def loss(theta):
    return (theta[0] - 2)**2 + (theta[1] - 3)**2

result = minimize(loss, x0=[0, 0], method='BFGS')
print(f"Optimal: {result.x}")  # [2, 3]
```

---

## R Implementation

```r
# Convex Function to Minimize: f(x) = (x-3)^2
f <- function(x) (x - 3)^2

# 1. 1D Optimization
res <- optimize(f, interval = c(0, 10))
print(paste("Min at:", res$minimum))

# 2. General Optimization (Nelder-Mead, BFGS)
f2 <- function(x) (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2  # Rosenbrock
res_bfgs <- optim(par = c(0,0), fn = f2, method = "BFGS")
print(res_bfgs$par)
```

---

## Related Concepts

- [[stats/01_Foundations/Loss Function\|Loss Function]] - What we optimize
- [[stats/01_Foundations/Convergence\|Convergence]] - When to stop
- [[stats/03_Regression_Analysis/Regularization\|Regularization]] - Constrained optimization

---

## References

- **Book:** Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*. Cambridge. [Free Online Version](https://web.stanford.edu/~boyd/cvxbook/)
