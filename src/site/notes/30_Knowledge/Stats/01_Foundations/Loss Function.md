---
{"dg-publish":true,"permalink":"/30-knowledge/stats/01-foundations/loss-function/","tags":["probability","foundations"]}
---


## Definition

> [!abstract] Core Statement
> A **Loss Function** quantifies the ==discrepancy between predictions and actual values==, guiding model training by defining what "good" means.

---

## Common Loss Functions

| Name | Formula | Use Case |
|------|---------|----------|
| **MSE** | $\frac{1}{n}\sum(y - \hat{y})^2$ | Regression |
| **MAE** | $\frac{1}{n}\sum\|y - \hat{y}\|$ | Robust regression |
| **Cross-Entropy** | $-\sum y \log \hat{y}$ | Classification |
| **Hinge** | $\max(0, 1 - y \cdot \hat{y})$ | SVM |
| **Huber** | Quadratic (small errors) + Linear (large) | Robust |

---

## Properties to Consider

| Property | MSE | MAE | Cross-Entropy |
|----------|-----|-----|---------------|
| Differentiable | ✓ | ✗ at 0 | ✓ |
| Outlier sensitive | High | Low | Medium |
| Probabilistic | ✗ | ✗ | ✓ |

---

## Python Implementation

```python
import numpy as np

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def cross_entropy(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred + 1e-10))
```

---

## R Implementation

```r
# True values and predictions
y_true <- c(3.0, -0.5, 2.0, 7.0)
y_pred <- c(2.5, 0.0, 2.1, 7.8)

# MSE
mse <- mean((y_true - y_pred)^2)
print(paste("MSE:", mse))

# MAE
mae <- mean(abs(y_true - y_pred))
print(paste("MAE:", mae))

# Log Loss (Binary Classification)
# y_true must be 0 or 1, y_pred is probability
log_loss <- function(y_true, y_pred) {
  eps <- 1e-15
  y_pred <- pmax(pmin(y_pred, 1 - eps), eps)
  -mean(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))
}
```

---

## Related Concepts

- [[30_Knowledge/Stats/01_Foundations/Optimization\|Optimization]] - Minimizing the loss
- [[30_Knowledge/Stats/04_Supervised_Learning/Gradient Descent\|Gradient Descent]] - Common optimization method
- [[30_Knowledge/Stats/03_Regression_Analysis/Regularization\|Regularization]] - Adding penalty to loss

---

## When to Use

> [!success] Use Loss Function When...
> - Refer to standard documentation
> - Refer to standard documentation

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Assumptions are violated
> - Alternative methods are more appropriate

---

## References

- **Book:** Goodfellow, I., et al. (2016). *Deep Learning*. MIT Press. [Book Website](http://www.deeplearningbook.org/)
