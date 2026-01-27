---
{"dg-publish":true,"permalink":"/stats/05-time-series/breusch-godfrey-test/","tags":["Diagnostics","Time-Series","Regression"]}
---

## Overview

> [!abstract] Definition
> The **Breusch-Godfrey Test** is a Lagrange Multiplier test for autocorrelation in the errors in a regression model. Unlike the [[stats/05_Time_Series/Durbin-Watson Test\|Durbin-Watson Test]], it is valid in the presence of lagged dependent variables and can test for higher-order serial correlation (up to lag $p$).

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

- [[stats/05_Time_Series/Durbin-Watson Test\|Durbin-Watson Test]] - Simpler alternative.
- [[stats/03_Regression_Analysis/Multiple Linear Regression\|Multiple Linear Regression]] - Framework.

---

## References

- **Article:** Breusch, T. S. (1978). Testing for autocorrelation in dynamic linear models. *Australian Economic Papers*, 17(31), 334-355. [Wiley Link](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1467-8454.1978.tb00635.x)
- **Article:** Godfrey, L. G. (1978). Testing for higher order serial correlation in regression equations when the regressors include lagged dependent variables. *Econometrica*, 46(6), 1303-1310. [JSTOR Link](http://www.jstor.org/stable/1913831)
- **Book:** Greene, W. H. (2018). *Econometric Analysis* (8th ed.). Pearson. [Pearson Link](https://www.pearson.com/us/higher-education/program/Greene-Econometric-Analysis-8th-Edition/PGM334862.html)
