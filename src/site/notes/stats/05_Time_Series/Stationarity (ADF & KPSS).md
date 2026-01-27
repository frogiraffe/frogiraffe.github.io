---
{"dg-publish":true,"permalink":"/stats/05-time-series/stationarity-adf-and-kpss/","tags":["Time-Series","Diagnostics","Stationarity"]}
---

## Definition

> [!abstract] Core Statement
> A time series is **Stationary** if its statistical properties (mean, variance, autocorrelation) ==do not change over time==. Stationarity is a **prerequisite** for many time series models like [[stats/05_Time_Series/ARIMA Models\|ARIMA Models]].

---

> [!tip] Intuition (ELI5): The Treadmill
> A stationary time series is like a person walking on a treadmill. They are moving, but their average position and how much they "bounce" (variance) stay the same. Non-stationarity is like a person walking through a city—where they are now depends on where they were a block ago, and their average position and path are constantly changing.

![Stationary vs Non-Stationary Time Series](https://upload.wikimedia.org/wikipedia/commons/e/e1/Stationarycomparison.png)

---

## Purpose

1.  Determine if a time series can be modeled with standard techniques.
2.  Decide if **differencing** or other transformations are needed.

---

## When to Use

> [!success] Test for Stationarity When...
> - Preparing data for ARIMA, VAR, or Granger Causality.
> - Visual inspection of the series shows a trend or changing variance.

---

## Theoretical Background

### Types of Non-Stationarity

| Type | Description | Solution |
|------|-------------|----------|
| **Trend** | Mean changes over time. | Differencing ($\Delta Y_t = Y_t - Y_{t-1}$). |
| **Changing Variance** | Volatility changes over time. | Log transform; GARCH models. |
| **Seasonality** | Periodic patterns. | Seasonal differencing. |

### The Tests

| Test | Null Hypothesis ($H_0$) | Interpretation |
|------|------------------------|----------------|
| **ADF (Augmented Dickey-Fuller)** | Series **has a unit root** (Non-Stationary). | Reject if p < 0.05: Series is stationary. |
| **KPSS** | Series **is stationary**. | Reject if p < 0.05: Series is non-stationary. |

> [!important] Use Both Tests Together
> | ADF Result | KPSS Result | Conclusion |
> |------------|-------------|------------|
> | Reject (Stationary) | Fail to Reject (Stationary) | **Stationary.** |
> | Fail to Reject | Reject (Non-Stationary) | **Non-Stationary.** Difference needed. |
> | Conflict | Conflict | Trend-stationary or need more investigation. |

---

## Worked Example: Is the Trend Real?

> [!example] Problem
> You have a stock price series $P_t$ that looks like it's going up.
> - **ADF p-value:** 0.65
> - **KPSS p-value:** 0.01
> 
> **Question:** Is the series stationary? Should you difference it?

**Analysis:**

1.  **ADF Test ($H_0$: Non-Stationary):**
    -   $p=0.65 > 0.05$. Fail to reject $H_0$.
    -   Evidence suggests **Non-Stationary**.

2.  **KPSS Test ($H_0$: Stationary):**
    -   $p=0.01 < 0.05$. Reject $H_0$.
    -   Evidence suggests **Non-Stationary**.

3.  **Conclusion:**
    -   Both tests agree. The series has a **Unit Root**.
    -   **Action:** Apply First Difference ($\Delta P_t = P_t - P_{t-1}$).

4.  **Re-Test (After Differencing):**
    -   New ADF $p < 0.01$. New KPSS $p > 0.10$.
    -   Now it is stationary.

---

## Assumptions

- [ ] **Time-Ordered Data:** Observations must be sequential in time.
- [ ] **Sufficient Length:** Short series have low power.
- [ ] **Constant Parameters:** The underlying process doesn't change regimes mid-stream.

---

## Limitations

> [!warning] Pitfalls
> 1.  **Trend Stationarity vs Difference Stationarity:** ADF assumes a *stochastic* trend suitable for differencing. If the trend is *deterministic* (e.g., a straight line $y=mx+b$), differencing introduces overdifferencing artifacts.
> 2.  **Structural Breaks:** A sudden crash (like 2008 financial crisis) is often interpreted by ADF as non-stationarity. Using standard ADF here is wrong; use **Zivot-Andrews Test** instead.
> 3.  **Seasonality:** A strong seasonal cycle can look like non-stationarity. Remove seasonality *before* testing.

---

## Python Implementation

```python
from statsmodels.tsa.stattools import adfuller, kpss

# ADF Test
adf_result = adfuller(series, autolag='AIC')
print(f"ADF Statistic: {adf_result[0]:.4f}")
print(f"ADF p-value: {adf_result[1]:.4f}")

# KPSS Test
kpss_result = kpss(series, regression='c')
print(f"KPSS Statistic: {kpss_result[0]:.4f}")
print(f"KPSS p-value: {kpss_result[1]:.4f}")

# Interpretation
if adf_result[1] < 0.05 and kpss_result[1] > 0.05:
    print("Series is STATIONARY.")
elif adf_result[1] > 0.05 and kpss_result[1] < 0.05:
    print("Series is NON-STATIONARY. Apply differencing.")
```

---

## R Implementation

```r
library(tseries)

# ADF Test
adf.test(series)

# KPSS Test
kpss.test(series, null = "Level")

# Interpretation:
# ADF p < 0.05 AND KPSS p > 0.05 -> Stationary.
```

---

## Interpretation Guide

| Test | p-value | Meaning |
|------|---------|---------|
| Test | p-value | Meaning |
|------|---------|---------|
| **ADF** | 0.02 | Reject $H_0$: **Stationary**. |
| **ADF** | 0.35 | Fail to reject: **Unit Root (Non-Stationary)**. |
| **KPSS** | 0.10 | Fail to reject: **Stationary**. |
| **KPSS** | 0.01 | Reject: **Non-Stationary**. |
| **Conflicting** | ADF > 0.05 & KPSS < 0.05 | **Non-Stationary**. Difference it. |
| **Conflicting** | ADF < 0.05 & KPSS > 0.05 | **Stationary**. Use as is. |

---

## Related Concepts

- [[stats/05_Time_Series/ARIMA Models\|ARIMA Models]] - Requires stationarity.
- [[stats/01_Foundations/Differencing\|Differencing]] - Transformation to achieve stationarity.
- [[stats/05_Time_Series/GARCH Models\|GARCH Models]] - For non-constant variance.

---

## References

- **Historical:** Dickey, D. A., & Fuller, W. A. (1979). Estimators for autoregressive time series with a unit root. *JASA*. [JSTOR](https://www.jstor.org/stable/2286331)
- **Historical:** Kwiatkowski, D., et al. (1992). Testing the null hypothesis of stationarity against the alternative of a unit root. *Journal of Econometrics*. [Elsevier](https://doi.org/10.1016/0304-4076(92)90104-Y)
- **Book:** Hamilton, J. D. (1994). *Time Series Analysis*. Princeton University Press. [Book Info](https://press.princeton.edu/books/hardcover/9780691042893/time-series-analysis)
