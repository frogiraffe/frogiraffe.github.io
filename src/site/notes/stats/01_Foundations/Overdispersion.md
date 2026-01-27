---
{"dg-publish":true,"permalink":"/stats/01-foundations/overdispersion/","tags":["Count-Data","GLM","Model-Diagnostics"]}
---


## Definition

> [!abstract] Core Statement
> **Overdispersion** occurs when the ==observed variance exceeds the theoretical variance== implied by a model (e.g., Poisson assumes Var = Mean).

$$\text{Dispersion} = \frac{\text{Observed Variance}}{\text{Model Variance}} > 1$$

---

## Detection

1. **Poisson:** If Var(Y) > E(Y) → overdispersed
2. **Residual deviance:** If deviance >> df → overdispersion
3. **Dispersion test:** Formal statistical test

---

## Causes

- Unmeasured heterogeneity
- Clustering/correlation
- Zero inflation
- Model misspecification

---

## Solutions

| Method | Approach |
|--------|----------|
| **Quasi-Poisson** | Estimate dispersion parameter |
| **Negative Binomial** | Model-based solution |
| **Robust SE** | Correct standard errors only |
| **Mixed Models** | Account for clustering |

---

## Python Implementation

```python
import statsmodels.api as sm

# Fit Poisson
poisson = sm.GLM(y, X, family=sm.families.Poisson()).fit()

# Check dispersion (should be ~1)
dispersion = poisson.deviance / poisson.df_resid
print(f"Dispersion: {dispersion:.2f}")

# Negative Binomial if overdispersed
if dispersion > 1.5:
    nb = sm.GLM(y, X, family=sm.families.NegativeBinomial()).fit()
```

---

## R Implementation

```r
model <- glm(y ~ x, family = poisson, data = df)
dispersion <- deviance(model) / df.residual(model)

# Quasi-Poisson
quasi <- glm(y ~ x, family = quasipoisson, data = df)

# Negative Binomial
library(MASS)
nb <- glm.nb(y ~ x, data = df)
```

---

## Related Concepts

- [[stats/03_Regression_Analysis/Poisson Regression\|Poisson Regression]] - Assumes no overdispersion
- [[stats/03_Regression_Analysis/Negative Binomial Regression\|Negative Binomial Regression]] - Handles overdispersion
- [[stats/01_Foundations/Hurdle Models\|Hurdle Models]] - For zero inflation

---

## References

- **Book:** Cameron, A. C., & Trivedi, P. K. (2013). *Regression Analysis of Count Data*. Cambridge. [DOI Link](https://doi.org/10.1017/CBO9781139013567)
