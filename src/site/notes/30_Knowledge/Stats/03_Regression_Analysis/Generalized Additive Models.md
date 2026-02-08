---
{"dg-publish":true,"permalink":"/30-knowledge/stats/03-regression-analysis/generalized-additive-models/","tags":["regression","modeling"]}
---


## Definition

> [!abstract] Core Statement
> **Generalized Additive Models** extend linear models by allowing ==non-linear relationships== through smooth functions of each predictor, while maintaining interpretability through additive structure.

$$
g(E[Y]) = \beta_0 + f_1(x_1) + f_2(x_2) + \ldots + f_p(x_p)
$$

Where $f_j$ are smooth functions (splines) learned from data.

---

> [!tip] Intuition (ELI5): The Flexible Rulers
> Linear regression uses only straight rulers. GAM uses flexible rulers (splines) that can bend to fit the data, while still keeping each variable's effect separate and interpretable.

---

## When to Use GAMs

> [!success] Use GAMs When...
> - Relationships are **non-linear** but you want **interpretability**
> - You need to **visualize** each predictor's effect
> - You prefer **additive structure** over interactions

> [!failure] Consider Alternatives When...
> - You need complex **interactions** → Gradient Boosting
> - **High-dimensional** data (many features) → Regularization
> - Need **maximum predictive power** → Neural networks

---

## Python Implementation

```python
from pygam import LinearGAM, LogisticGAM, s, f, l
import numpy as np
import matplotlib.pyplot as plt

# ========== LINEAR GAM ==========
# s() = spline, f() = factor, l() = linear
gam = LinearGAM(s(0) + s(1) + f(2))  # Two splines + one factor
gam.fit(X, y)

print(gam.summary())

# ========== VISUALIZE PARTIAL EFFECTS ==========
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for i, ax in enumerate(axes):
    XX = gam.generate_X_grid(term=i)
    pdep, confi = gam.partial_dependence(term=i, width=0.95)
    
    ax.plot(XX[:, i], pdep)
    ax.fill_between(XX[:, i], confi[:, 0], confi[:, 1], alpha=0.3)
    ax.set_xlabel(f'Feature {i}')
    ax.set_ylabel('Partial Effect')
    ax.set_title(f'Effect of Feature {i}')

plt.tight_layout()
plt.show()

# ========== LOGISTIC GAM ==========
gam_logistic = LogisticGAM(s(0) + s(1) + s(2))
gam_logistic.fit(X_train, y_train)

# ========== HYPERPARAMETER TUNING ==========
from pygam import LinearGAM
import numpy as np

# Grid search for smoothing parameter
lams = np.logspace(-3, 3, 11)
gam = LinearGAM(s(0) + s(1))
gam.gridsearch(X, y, lam=lams)
```

---

## R Implementation

```r
library(mgcv)

# ========== FIT GAM ==========
gam_model <- gam(y ~ s(x1) + s(x2) + factor(x3), 
                 data = data, 
                 family = gaussian)

summary(gam_model)

# ========== VISUALIZE EFFECTS ==========
plot(gam_model, pages = 1, residuals = TRUE)

# ========== AIC COMPARISON ==========
# Compare with linear model
lm_model <- lm(y ~ x1 + x2 + factor(x3), data = data)
AIC(lm_model, gam_model)

# ========== INTERACTION (TENSOR PRODUCT) ==========
gam_interaction <- gam(y ~ te(x1, x2) + s(x3), data = data)

# ========== VARYING SMOOTHNESS ==========
gam_k <- gam(y ~ s(x1, k = 5) + s(x2, k = 20), data = data)
```

---

## Spline Basis Functions

| Type | Description | Use |
|------|-------------|-----|
| **Thin plate (tp)** | Default, good all-rounder | General purpose |
| **Cubic regression (cr)** | Cubic splines | Speed, simpler |
| **Cyclic (cc)** | Endpoints meet | Time-of-day, seasonal |
| **Tensor product (te)** | For interactions | 2D+ smooth surfaces |

---

## Model Selection

**EDF (Effective Degrees of Freedom):**
- EDF ≈ 1: Nearly linear
- EDF = k-1: Maximum wiggliness
- GAM automatically selects via GCV/REML

**Checking fit:**
```r
gam.check(gam_model)  # Residual plots + basis dimension check
```

---

## GAM vs Other Models

| Model | Interpretability | Non-linearity | Interactions |
|-------|-----------------|---------------|--------------|
| **Linear Regression** | ⭐⭐⭐ | ✗ | Manual |
| **GAM** | ⭐⭐⭐ | ✓ | Manual |
| **Random Forest** | ⭐ | ✓ | ✓ |
| **XGBoost** | ⭐ | ✓ | ✓ |

---

## Common Pitfalls

> [!warning] Real-World Traps
>
> **1. Overfitting with Too Many Knots**
> - *Problem:* Wiggly curves that don't generalize
> - *Solution:* Use `k` parameter, let GCV choose smoothness
>
> **2. Ignoring Concurvity**
> - *Problem:* Collinearity between smooth terms
> - *Solution:* Check with `concurvity(gam_model)`
>
> **3. Missing Interactions**
> - *Problem:* GAM assumes additive effects
> - *Solution:* Use `te()` for important interactions

---

## Related Concepts

- [[30_Knowledge/Stats/03_Regression_Analysis/Polynomial Regression\|Polynomial Regression]] — Fixed-degree non-linearity
- [[30_Knowledge/Stats/04_Supervised_Learning/Gradient Boosting\|Gradient Boosting]] — Black-box alternative
- [[30_Knowledge/Stats/04_Supervised_Learning/Feature Engineering\|Feature Engineering]] — Manual non-linearity
- [[30_Knowledge/Stats/03_Regression_Analysis/Regularization\|Regularization]] — Controls smoothness

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Assumptions are violated
> - Alternative methods are more appropriate

---

## References

- **Book:** Wood, S. N. (2017). *Generalized Additive Models: An Introduction with R* (2nd ed.). CRC Press.
- **Paper:** Hastie, T., & Tibshirani, R. (1986). Generalized Additive Models. *Statistical Science*, 1(3), 297-318.
- **Package:** [pyGAM](https://pygam.readthedocs.io/), [mgcv](https://cran.r-project.org/package=mgcv)
