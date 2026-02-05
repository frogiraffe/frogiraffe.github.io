---
{"dg-publish":true,"permalink":"/stats/08-time-series-analysis/durbin-watson-test/","tags":["diagnostics","time-series","regression","autocorrelation"]}
---

## Definition

> [!abstract] Core Statement
> The **Durbin-Watson (DW) Test** detects **autocorrelation** (or serial correlation) in the residuals of a regression analysis. It tests whether adjacent residuals are correlated, which violates the assumption of independence in [[stats/03_Regression_Analysis/Simple Linear Regression\|Simple Linear Regression]].

---

## Purpose

1.  **Diagnose Assumption Violation:** Ensure regression error terms are independent.
2.  **Flag Time Series Issues:** Detect if a model fails to capture time-dependent patterns.

---

## When to Use

> [!success] Use Durbin-Watson When...
> - Data was collected over **time** (Time Series).
> - You fit a Linear Regression model.
> - You suspect the error at time $t$ depends on error at time $t-1$.

> [!failure] Do NOT Use When...
> - Data is cross-sectional (order doesn't matter).
> - Use [[stats/08_Time_Series_Analysis/Breusch-Godfrey Test\|Breusch-Godfrey Test]] for higher-order autocorrelation (lag > 1).

---

## Theoretical Background

### The Statistic ($d$)

$$
d = \frac{\sum_{t=2}^T (e_t - e_{t-1})^2}{\sum_{t=1}^T e_t^2}
$$
where $e_t$ is the residual at time $t$.

### Interpretation Range

The $d$ statistic ranges from **0 to 4**:
- **$d \approx 2$:** **No Autocorrelation** (Ideal).
- **$0 \le d < 1.5$:** **Positive Autocorrelation** (Common in time series).
- **$2.5 < d \le 4$:** **Negative Autocorrelation** (Rapid oscillation).

---

## Worked Numerical Example

> [!example] Stock Returns vs Market Returns
> **Regression:** $Return_{Asset} = \beta_0 + \beta_1 Return_{Market}$
> **Residuals:** [0.5, 0.4, 0.6, -0.2, -0.3, -0.5]
> 
> **Observation:** Positive residuals cluster together; Negative residuals cluster together.
> **Calculation:**
> - DW Statistic $d = 0.8$.
> 
> **Conclusion:** $d \ll 2$. Strong **Positive Autocorrelation**.
> **Implication:** Standard errors are underestimated. Calculated t-stats are inflated. The relationship appears more significant than it really is.

---

## Assumptions

- [ ] **First-Order Autocorrelation:** Tests only for correlation between $e_t$ and $e_{t-1}$.
- [ ] **Interceptor:** Regression must include an intercept.
- [ ] **No Lagged DV:** Independent variables cannot include lagged response variables ($Y_{t-1}$). Use [[stats/08_Time_Series_Analysis/Breusch-Godfrey Test\|Breusch-Godfrey Test]] instead.

---

## Limitations

> [!warning] Pitfalls
> 1.  **Order Matters:** Calculating DW on unsorted cross-sectional data is meaningless.
> 2.  **Inconclusive Region:** The test has "bounds" ($d_L, d_U$). If $d$ falls between them, the test is inconclusive.
> 3.  **Only Lag 1:** Does not detect seasonal patterns (e.g., Lag 4 or Lag 12 autocorrelation).

---

## Python Implementation

```python
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson

# Feature: Residuals from a fitted model
residuals = model.resid

# Calculate DW
dw_stat = durbin_watson(residuals)
print(f"Durbin-Watson: {dw_stat:.2f}")

if dw_stat < 1.5:
    print("Warning: Positive Autocorrelation")
elif dw_stat > 2.5:
    print("Warning: Negative Autocorrelation")
else:
    print("Assumption Met: Residuals are independent")
```

---

## R Implementation

```r
library(lmtest)

# Fit Model
model <- lm(Sales ~ Time, data = df)

# Durbin-Watson Test
dwtest(model)
# Output includes DW statistic and p-value
```

---

## Interpretation Guide

| DW Value | Interpretation | Edge Case Notes |
|----------|----------------|-----------------|
| 2.0 | No Autocorrelation. | Perfect independence. |
| 1.8 - 2.2 | Acceptable range. | Usually considered "close enough" to 2. |
| 0.5 | Strong Positive Autocorrelation. | Standard errors will be biased downwards. Risk of spurious regression. |
| 1.3 | Inconclusive Zone? | Check critical value tables ($d_L, d_U$ based on n and k). |

---

## Related Concepts

- [[stats/08_Time_Series_Analysis/Breusch-Godfrey Test\|Breusch-Godfrey Test]] - More general test for AR(p) errors.
- [[stats/02_Statistical_Inference/Ljung-Box Test\|Ljung-Box Test]] - Tests white noise in ARIMA residuals.
- [[stats/08_Time_Series_Analysis/Time Series Analysis\|Time Series Analysis]]
- [[stats/08_Time_Series_Analysis/Stationarity (ADF & KPSS)\|Stationarity (ADF & KPSS)]]

---

## References

- **Historical:** Durbin, J., & Watson, G. S. (1950). Testing for serial correlation in least squares regression I. *Biometrika*. [JSTOR](https://www.jstor.org/stable/2332391)
- **Historical:** Durbin, J., & Watson, G. S. (1951). Testing for serial correlation in least squares regression II. *Biometrika*. [JSTOR](https://www.jstor.org/stable/2332325)
- **Book:** Greene, W. H. (2018). *Econometric Analysis* (8th ed.). Pearson. [Pearson](https://www.pearson.com/us/higher-education/program/Greene-Econometric-Analysis-8th-Edition/PGM334704.html)