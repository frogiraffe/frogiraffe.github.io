---
{"dg-publish":true,"permalink":"/stats/01-foundations/robust-standard-errors/","tags":["Inference","Standard-Error","Regression"]}
---


## Definition

> [!abstract] Core Statement
> **Robust Standard Errors** are standard error estimates that remain ==valid under heteroscedasticity or clustering==, unlike classical OLS standard errors.

---

## Types

| Type | Handles |
|------|---------|
| **HC0-HC3** | Heteroscedasticity |
| **Clustered SE** | Within-group correlation |
| **HAC (Newey-West)** | Heteroscedasticity + autocorrelation |

---

## Python Implementation

```python
import statsmodels.api as sm

model = sm.OLS(y, X).fit(cov_type='HC3')  # Robust SE
print(model.summary())

# Clustered
# model.fit(cov_type='cluster', cov_kwds={'groups': cluster_id})
```

---

## R Implementation

```r
library(sandwich)
library(lmtest)

model <- lm(y ~ x, data = df)
coeftest(model, vcov = vcovHC(model, type = "HC3"))

# Clustered
# vcovCL(model, cluster = df$cluster_id)
```

---

## When to Use

- [[stats/03_Regression_Analysis/Heteroscedasticity\|Heteroscedasticity]] detected in residuals
- Panel/clustered data (repeated measures from same units)
- Time series with autocorrelation

---

## Related Concepts

- [[stats/03_Regression_Analysis/Heteroscedasticity\|Heteroscedasticity]] - Motivation for robust SE
- [[stats/03_Regression_Analysis/Weighted Least Squares (WLS)\|Weighted Least Squares (WLS)]] - Alternative fix

---

## References

- **Article:** White, H. (1980). A heteroskedasticity-consistent covariance matrix estimator. *Econometrica*, 48(4), 817-838. [JSTOR](https://www.jstor.org/stable/1912934)
