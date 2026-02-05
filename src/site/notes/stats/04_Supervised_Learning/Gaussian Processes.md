---
{"dg-publish":true,"permalink":"/stats/04-supervised-learning/gaussian-processes/","tags":["probability","machine-learning","probabilistic","bayesian"]}
---


## Definition

> [!abstract] Core Statement
> **Gaussian Processes** are non-parametric Bayesian models that define a ==distribution over functions==. They provide predictions with uncertainty estimates.

---

> [!tip] Intuition (ELI5)
> Instead of fitting one curve, GPs consider ALL possible curves and weight them by how well they fit the data. The result is a "fuzzy" prediction with confidence bounds.

---

## Key Concepts

| Concept | Description |
|---------|-------------|
| **Kernel** | Defines similarity between points |
| **Mean Function** | Prior expectation (often 0) |
| **Posterior** | Updated belief after seeing data |
| **Predictive Variance** | Uncertainty estimate |

---

## Python Implementation

```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
import numpy as np
import matplotlib.pyplot as plt

# Define kernel
kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)

# Fit GP
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
gp.fit(X_train, y_train)

# Predict with uncertainty
y_pred, y_std = gp.predict(X_test, return_std=True)

# Plot with confidence interval
plt.plot(X_test, y_pred)
plt.fill_between(X_test.ravel(), 
                 y_pred - 1.96*y_std, 
                 y_pred + 1.96*y_std, 
                 alpha=0.3)
plt.show()
```

---

## Common Kernels

| Kernel | Formula | Use |
|--------|---------|-----|
| **RBF/Squared Exp** | $\exp(-\frac{d^2}{2l^2})$ | Smooth functions |
| **Matérn** | Various | Controllable smoothness |
| **Periodic** | Sine-based | Seasonal patterns |

---

## Strengths & Weaknesses

| ✓ Strengths | ✗ Weaknesses |
|-------------|--------------|
| Uncertainty estimates | Scales as O(n³) |
| Flexible (kernel choice) | Large datasets slow |
| Works with small data | Hyperparameter sensitive |

---

## Related Concepts

- [[stats/04_Supervised_Learning/Kernel Methods\|Kernel Methods]] — Kernel functions
- [[stats/01_Foundations/Bayesian Statistics\|Bayesian Statistics]] — Probabilistic framework

---

## References

- **Book:** Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press. [Free Online](http://www.gaussianprocess.org/gpml/)
