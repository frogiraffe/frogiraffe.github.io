---
{"dg-publish":true,"permalink":"/stats/03-regression-analysis/generalized-least-squares/","tags":["probability","regression","econometrics"]}
---


## Definition

> [!abstract] Core Statement
> **Generalized Least Squares** extends OLS to handle ==non-spherical errors== (heteroscedasticity and/or autocorrelation) by transforming the model.

$$
\hat{\beta}_{GLS} = (X'\Omega^{-1}X)^{-1}X'\Omega^{-1}y
$$

Where $\Omega = \text{Var}(\epsilon)$ is the error covariance matrix.

---

## When to Use

| Violation | OLS Problem | GLS Solution |
|-----------|-------------|--------------|
| **Heteroscedasticity** | Inefficient, wrong SE | Weight by inverse variance |
| **Autocorrelation** | Inefficient, wrong SE | Model error structure |

---

## Python Implementation

```python
import statsmodels.api as sm

# If you know the error structure (e.g., AR(1))
# Use GLSAR for autocorrelated errors
model = sm.GLSAR(y, X, rho=1)
result = model.iterative_fit(maxiter=10)
print(result.summary())

# ========== FEASIBLE GLS ==========
# Step 1: Estimate with OLS
ols = sm.OLS(y, X).fit()

# Step 2: Estimate variance function from residuals
resid_sq = ols.resid ** 2
var_model = sm.OLS(resid_sq, X).fit()
weights = 1 / var_model.fittedvalues

# Step 3: WLS (a form of GLS)
wls = sm.WLS(y, X, weights=weights).fit()
print(wls.summary())
```

---

## R Implementation

```r
library(nlme)

# GLS with AR(1) errors
model <- gls(y ~ x, data = df, correlation = corAR1())
summary(model)

# GLS with heteroscedasticity
model <- gls(y ~ x, data = df, weights = varPower())
summary(model)
```

---

## Related Concepts

- [[stats/03_Regression_Analysis/Weighted Least Squares (WLS)\|Weighted Least Squares (WLS)]] — Special case of GLS
- [[stats/01_Foundations/Robust Standard Errors\|Robust Standard Errors]] — Alternative approach
- [[stats/03_Regression_Analysis/Heteroscedasticity\|Heteroscedasticity]] — Common motivation

---

## References

- **Book:** Greene, W. H. (2018). *Econometric Analysis* (8th ed.). Pearson.
