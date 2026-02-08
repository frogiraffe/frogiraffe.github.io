---
{"dg-publish":true,"permalink":"/30-knowledge/stats/08-time-series-analysis/breusch-godfrey-test/","tags":["time-series"]}
---

## Overview

> [!abstract] Definition
> The **Breusch-Godfrey Test** is a Lagrange Multiplier test for autocorrelation in the errors in a regression model. Unlike the [[30_Knowledge/Stats/08_Time_Series_Analysis/Durbin-Watson Test\|Durbin-Watson Test]], it is valid in the presence of lagged dependent variables and can test for higher-order serial correlation (up to lag $p$).

---

## 1. Procedure

1. Estimate the OLS regression: $Y_t = X_t\beta + u_t$
2. Obtain residuals: $\hat{u}_t$
3. Run auxiliary regression: Regress $\hat{u}_t$ on original $X_t$ and lagged residuals $\hat{u}_{t-1}, \dots, \hat{u}_{t-p}$.
4. Calculate $LM = (N-p)R^2$ from the auxiliary regression.

---

## 2. Comparison: BG vs. DW

| Feature | Durbin-Watson | Breusch-Godfrey |
|---------|---------------|-----------------|
| **Order defined** | First-order (AR(1)) only | Any order $p$ |
| **Lagged Dependent Variables** | Invalid (Biased) | Valid |
| **Distribution** | Exact Bounds tables | Asymptotic Chi-Square |

**Recommendation:** Breusch-Godfrey is generally preferred for its flexibility.

---

## 3. Python Implementation Example

```python
import statsmodels.stats.diagnostic as dg

# Perform Test (up to lag 2)
# Results: [LM Stat, LM p-value, F-Stat, F p-value]
bg_test = dg.acorr_breusch_godfrey(model, nlags=2)

print(f"LM Statistic: {bg_test[0]:.3f}")
print(f"LM p-value: {bg_test[1]:.4f}")
```

---

## 4. Related Concepts

- [[30_Knowledge/Stats/08_Time_Series_Analysis/Durbin-Watson Test\|Durbin-Watson Test]] - Simpler alternative.
- [[30_Knowledge/Stats/03_Regression_Analysis/Multiple Linear Regression\|Multiple Linear Regression]] - Framework.

---

## Definition

> [!abstract] Core Statement
> **Breusch-Godfrey Test** ... Refer to standard documentation

---

> [!tip] Intuition (ELI5)
> Refer to standard documentation

---

## When to Use

> [!success] Use Breusch-Godfrey Test When...
> - Refer to standard documentation
> - Refer to standard documentation

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Assumptions of the test are violated
> - Sample size doesn't meet minimum requirements

---

## Python Implementation

```python
from scipy import stats
import numpy as np

# Sample data
group1 = np.random.normal(10, 2, 30)
group2 = np.random.normal(12, 2, 30)

# Perform test
statistic, pvalue = stats.ttest_ind(group1, group2)

print(f"Test Statistic: {statistic:.4f}")
print(f"P-value: {pvalue:.4f}")
print(f"Significant at α=0.05: {pvalue < 0.05}")
```

---

## R Implementation

```r
# Breusch-Godfrey Test in R
set.seed(42)

# Sample data
group1 <- rnorm(30, mean = 10, sd = 2)
group2 <- rnorm(30, mean = 12, sd = 2)

# Perform test
result <- t.test(group1, group2)
print(result)
```

---

## Related Concepts

- [[30_Knowledge/Stats/08_Time_Series_Analysis/ARIMA Models\|ARIMA Models]]
- [[30_Knowledge/Stats/08_Time_Series_Analysis/Stationarity (ADF & KPSS)\|Stationarity (ADF & KPSS)]]
- [[30_Knowledge/Stats/08_Time_Series_Analysis/Auto-Correlation (ACF & PACF)\|Auto-Correlation (ACF & PACF)]]

---

## References

- **Article:** Breusch, T. S. (1978). Testing for autocorrelation in dynamic linear models. *Australian Economic Papers*, 17(31), 334-355. [Wiley Link](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1467-8454.1978.tb00635.x)
- **Article:** Godfrey, L. G. (1978). Testing for higher order serial correlation in regression equations when the regressors include lagged dependent variables. *Econometrica*, 46(6), 1303-1310. [JSTOR Link](http://www.jstor.org/stable/1913831)
- **Book:** Greene, W. H. (2018). *Econometric Analysis* (8th ed.). Pearson. [Pearson Link](https://www.pearson.com/us/higher-education/program/Greene-Econometric-Analysis-8th-Edition/PGM334862.html)
