---
{"dg-publish":true,"permalink":"/stats/08-time-series-analysis/unit-root-tests/","tags":["Time-Series","Statistics","Stationarity"]}
---


## Definition

> [!abstract] Core Statement
> **Unit Root Tests** determine whether a time series is ==stationary or non-stationary==. A unit root indicates the series has a stochastic trend and requires differencing to achieve stationarity.

---

> [!tip] Intuition (ELI5): The Drunk Walk
> Imagine a person taking random steps. If they have a "unit root," each step's effect is permanent — they wander without returning. Stationary series are like a person on a leash — they may wander but always return to a central point.

---

## Why It Matters

| Non-Stationary Series | Problems |
|-----------------------|----------|
| Trending mean | Regression is spurious |
| Time-varying variance | Standard errors wrong |
| Non-constant autocorrelation | ARMA doesn't work |

**Most time series methods require stationarity!**

---

## Common Tests

| Test | Null Hypothesis | Alternative | Use |
|------|-----------------|-------------|-----|
| **ADF** (Augmented Dickey-Fuller) | Unit root (non-stationary) | Stationary | Most common |
| **KPSS** | Stationary | Unit root | Confirmation |
| **Phillips-Perron** | Unit root | Stationary | Robust to heteroskedasticity |

> [!important] Use Both ADF and KPSS
> - ADF rejects H₀ → stationary
> - KPSS fails to reject H₀ → stationary
> - If they disagree → investigate further

---

## Python Implementation

```python
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
import matplotlib.pyplot as plt

# ========== GENERATE DATA ==========
np.random.seed(42)
n = 200

# Stationary series
stationary = np.random.normal(0, 1, n)

# Non-stationary (random walk)
non_stationary = np.cumsum(np.random.normal(0, 1, n))

# Trend-stationary
trend_stationary = 0.1 * np.arange(n) + np.random.normal(0, 1, n)

# ========== ADF TEST ==========
def adf_test(series, name):
    result = adfuller(series, autolag='AIC')
    print(f"\n{name}")
    print(f"ADF Statistic: {result[0]:.4f}")
    print(f"p-value: {result[1]:.4f}")
    print(f"Critical Values: {result[4]}")
    if result[1] < 0.05:
        print("→ STATIONARY (reject H0)")
    else:
        print("→ NON-STATIONARY (fail to reject H0)")

adf_test(stationary, "Stationary Series")
adf_test(non_stationary, "Random Walk (Non-Stationary)")
adf_test(trend_stationary, "Trend-Stationary")

# ========== KPSS TEST ==========
def kpss_test(series, name):
    result = kpss(series, regression='c')  # 'ct' for trend
    print(f"\n{name}")
    print(f"KPSS Statistic: {result[0]:.4f}")
    print(f"p-value: {result[1]:.4f}")
    print(f"Critical Values: {result[3]}")
    if result[1] > 0.05:
        print("→ STATIONARY (fail to reject H0)")
    else:
        print("→ NON-STATIONARY (reject H0)")

kpss_test(stationary, "Stationary Series")
kpss_test(non_stationary, "Random Walk")

# ========== DIFFERENCING ==========
diff_series = np.diff(non_stationary)
adf_test(diff_series, "Differenced Random Walk")

# ========== PLOT ==========
fig, axes = plt.subplots(3, 1, figsize=(12, 8))
axes[0].plot(stationary)
axes[0].set_title('Stationary')
axes[1].plot(non_stationary)
axes[1].set_title('Non-Stationary (Random Walk)')
axes[2].plot(trend_stationary)
axes[2].set_title('Trend-Stationary')
plt.tight_layout()
plt.show()
```

---

## R Implementation

```r
library(tseries)
library(urca)

# ========== ADF TEST ==========
adf.test(series)

# ========== KPSS TEST ==========
kpss.test(series)

# ========== MORE DETAILED (urca) ==========
adf_result <- ur.df(series, type = "drift", lags = 5)
summary(adf_result)

# ========== DIFFERENCING ==========
diff_series <- diff(series)
adf.test(diff_series)
```

---

## Interpretation Matrix

| ADF Result | KPSS Result | Conclusion |
|------------|-------------|------------|
| Reject H₀ | Fail to reject | **Stationary** |
| Fail to reject | Reject H₀ | **Non-stationary** |
| Both reject | — | Conflicting: likely trend-stationary |
| Neither rejects | — | Conflicting: more investigation needed |

---

## Making Series Stationary

| Transformation | Fixes |
|----------------|-------|
| **Differencing** | Trend and unit root |
| **Log transformation** | Exponential growth, stabilizes variance |
| **Seasonal differencing** | Seasonal unit root |
| **Detrending** | Deterministic trend only |

```python
# First difference
y_diff = np.diff(y)

# Seasonal difference (monthly data, lag=12)
y_seasonal_diff = y[12:] - y[:-12]

# Log + difference (common for financial data)
y_log_diff = np.diff(np.log(y))
```

---

## Common Pitfalls

> [!warning] Real-World Traps
>
> **1. Ignoring Deterministic Trend**
> - *Problem:* ADF doesn't distinguish trend-stationary from unit root
> - *Solution:* Use `regression='ct'` in KPSS for trend test
>
> **2. Wrong Number of Lags in ADF**
> - *Problem:* Too few → biased, too many → low power
> - *Solution:* Use `autolag='AIC'`
>
> **3. Over-differencing**
> - *Problem:* Differencing stationary series makes it worse
> - *Solution:* Test before differencing

---

## Related Concepts

- [[stats/08_Time_Series_Analysis/Stationarity (ADF & KPSS)\|Stationarity (ADF & KPSS)]] — Detailed note
- [[stats/08_Time_Series_Analysis/ARIMA Models\|ARIMA Models]] — Requires stationarity
- [[stats/08_Time_Series_Analysis/SARIMA\|SARIMA]] — Handles seasonal unit roots
- [[stats/08_Time_Series_Analysis/Granger Causality\|Granger Causality]] — Requires stationarity

---

## References

- **Paper:** Dickey, D. A., & Fuller, W. A. (1979). Distribution of the estimators for autoregressive time series with a unit root. *JASA*, 74(366), 427-431.
- **Paper:** Kwiatkowski, D., et al. (1992). Testing the null hypothesis of stationarity against the alternative of a unit root. *JoE*, 54(1-3), 159-178.
- **Book:** Hamilton, J. D. (1994). *Time Series Analysis*. Princeton University Press.
