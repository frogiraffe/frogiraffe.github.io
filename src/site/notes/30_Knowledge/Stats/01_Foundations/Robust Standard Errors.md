---
{"dg-publish":true,"permalink":"/30-knowledge/stats/01-foundations/robust-standard-errors/","tags":["probability","foundations"]}
---


## Definition

> [!abstract] Core Statement
> **Robust Standard Errors** are standard error estimates that remain ==valid under heteroscedasticity or clustering==, unlike classical OLS standard errors.

---

> [!tip] Intuition (ELI5)
> Classical SEs assume all residuals have the same spread. If rich people's incomes vary wildly while poor people's are tight, classical SEs are wrong. Robust SEs adapt to this unequal spread.

---

## Why Use Them?

```
Classical SE assumes:      Robust SE handles:
   ε                          ε
   │                          │
   ●  ●  ●  ●  ●              ●  ●●●●●●●●●●
───┼─────────────→ X      ───┼─────────────→ X
   │                          │
Same spread everywhere      Spread increases with X
```

---

## Types

| Type | Handles | When to Use |
|------|---------|-------------|
| **HC0** | Heteroscedasticity | Large n |
| **HC1** | HC0 with df correction | Default in many packages |
| **HC2** | Better for small n | Leverage issues |
| **HC3** | Best for small n | Recommended default |
| **Clustered** | Within-group correlation | Panel data, experiments |
| **HAC (Newey-West)** | Heteroscedasticity + autocorrelation | Time series |

---

## Python Implementation

```python
import statsmodels.api as sm
import numpy as np
import pandas as pd

# ========== SIMULATE HETEROSCEDASTIC DATA ==========
np.random.seed(42)
n = 100
X = np.random.uniform(1, 10, n)
epsilon = np.random.normal(0, X)  # Variance increases with X!
y = 2 + 3*X + epsilon

X_const = sm.add_constant(X)

# ========== COMPARE SE TYPES ==========
# Classical (WRONG for heteroscedastic data)
model_classic = sm.OLS(y, X_const).fit()
print("Classical SE:", model_classic.bse[1])

# Robust HC3 (CORRECT)
model_robust = sm.OLS(y, X_const).fit(cov_type='HC3')
print("Robust HC3 SE:", model_robust.bse[1])

# ========== CLUSTERED SE ==========
# For panel data with group structure
# model.fit(cov_type='cluster', cov_kwds={'groups': df['group_id']})

# ========== HAC (NEWEY-WEST) ==========
# For time series with autocorrelation
# model.fit(cov_type='HAC', cov_kwds={'maxlags': 5})
```

---

## Comparison: Classical vs Robust

| Aspect | Classical SE | Robust SE |
|--------|--------------|-----------|
| **Assumes** | Homoscedasticity | Nothing about variance |
| **Coefficients** | Same | Same (unchanged) |
| **Standard Errors** | Biased if heteroscedastic | Consistent |
| **P-values** | Wrong | Correct |
| **Confidence Intervals** | Wrong width | Correct width |

---

## R Implementation

```r
library(sandwich)
library(lmtest)

model <- lm(y ~ x, data = df)

# ========== COMPARE ==========
# Classical
summary(model)

# Robust HC3
coeftest(model, vcov = vcovHC(model, type = "HC3"))

# ========== CLUSTERED ==========
library(clubSandwich)
coef_test(model, vcov = vcovCR(model, cluster = df$group_id, type = "CR2"))

# ========== HAC (NEWEY-WEST) ==========
coeftest(model, vcov = NeweyWest(model, lag = 5))
```

---

## Worked Example

> [!example] Detecting the Problem
> 
> **Step 1:** Run OLS and plot residuals vs fitted values
> ```python
> import matplotlib.pyplot as plt
> plt.scatter(model.fittedvalues, model.resid)
> plt.xlabel('Fitted'); plt.ylabel('Residuals')
> plt.title('Residual Plot')
> ```
> 
> **Step 2:** If fan-shaped pattern → heteroscedasticity present
> 
> **Step 3:** Use robust SEs
> ```python
> model_robust = sm.OLS(y, X).fit(cov_type='HC3')
> ```

---

## Common Pitfalls

> [!warning] Traps
>
> **1. Robust SEs don't "fix" OLS**
> - Coefficients are still consistent
> - Only SEs are corrected
>
> **2. Overcorrection with small n**
> - HC0/HC1 can be wrong for n < 50
> - Use HC3 for safety
>
> **3. Ignoring clustering**
> - Experiments with multiple measures per person
> - Schools with students

---

## When to Use

- ✅ [[30_Knowledge/Stats/03_Regression_Analysis/Heteroscedasticity\|Heteroscedasticity]] detected in residuals
- ✅ Panel/clustered data (students within schools)
- ✅ Time series with autocorrelation
- ✅ **Default:** Always use HC3 when in doubt

---

## Related Concepts

- [[30_Knowledge/Stats/03_Regression_Analysis/Heteroscedasticity\|Heteroscedasticity]] — Motivation for robust SE
- [[30_Knowledge/Stats/03_Regression_Analysis/Weighted Least Squares (WLS)\|Weighted Least Squares (WLS)]] — Alternative fix
- [[30_Knowledge/Stats/03_Regression_Analysis/Generalized Least Squares\|Generalized Least Squares]] — Full variance modeling

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Assumptions are violated
> - Alternative methods are more appropriate

---

## References

- **Paper:** White, H. (1980). A heteroskedasticity-consistent covariance matrix estimator. *Econometrica*, 48(4), 817-838.
- **Paper:** MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-consistent covariance matrix estimators with improved finite sample properties.
- **Book:** Angrist, J. D., & Pischke, J. S. (2008). *Mostly Harmless Econometrics*. Princeton.

