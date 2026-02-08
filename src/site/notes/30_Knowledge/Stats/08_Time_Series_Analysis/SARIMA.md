---
{"dg-publish":true,"permalink":"/30-knowledge/stats/08-time-series-analysis/sarima/","tags":["time-series"]}
---


## Definition

> [!abstract] Core Statement
> **SARIMA** (Seasonal ARIMA) extends [[30_Knowledge/Stats/08_Time_Series_Analysis/ARIMA Models\|ARIMA Models]] to handle ==seasonal patterns== in time series data. It adds seasonal AR, I, and MA terms that operate at the seasonal lag.

$$
\text{SARIMA}(p, d, q)(P, D, Q)_m
$$

Where:
- $(p, d, q)$ = Non-seasonal ARIMA parameters
- $(P, D, Q)_m$ = Seasonal parameters at lag $m$
- $m$ = Seasonal period (12 for monthly, 4 for quarterly, 7 for daily weekly)

---

> [!tip] Intuition (ELI5): The Double Pattern
> Ice cream sales have two patterns: a daily trend (it's been getting hotter) and a yearly cycle (summer = high). SARIMA captures both — the regular day-to-day changes AND the "same month last year" pattern.

---

## When to Use

> [!success] Use SARIMA When...
> - Data shows **repeating seasonal patterns**
> - You can identify the **seasonal period** clearly
> - Data needs both **regular and seasonal differencing** for stationarity

> [!failure] Consider Alternatives When...
> - Multiple seasonality (both weekly and yearly) → Use Prophet or TBATS
> - Very long seasonal periods (365 for daily) → Use Fourier terms

---

## Model Components

| Component | Symbol | Example (Monthly) |
|-----------|--------|-------------------|
| Seasonal AR | $P$ | Correlation with 12, 24 months ago |
| Seasonal Differencing | $D$ | $y_t - y_{t-12}$ |
| Seasonal MA | $Q$ | Shocks from 12, 24 months ago |
| Seasonal Period | $m$ | 12 |

### Full Model

$$
\Phi_P(B^m) \phi_p(B) (1-B)^d (1-B^m)^D y_t = \Theta_Q(B^m) \theta_q(B) \epsilon_t
$$

---

## Python Implementation

```python
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

# ========== SIMULATE SEASONAL DATA ==========
np.random.seed(42)
n = 120  # 10 years monthly data
t = np.arange(n)
seasonal = 10 * np.sin(2 * np.pi * t / 12)  # Yearly cycle
trend = 0.05 * t
noise = np.random.normal(0, 1, n)
y = 50 + trend + seasonal + noise

dates = pd.date_range(start='2015-01-01', periods=n, freq='M')
ts = pd.Series(y, index=dates)

# ========== FIT SARIMA ==========
model = SARIMAX(ts, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
results = model.fit(disp=False)
print(results.summary())

# ========== FORECAST ==========
forecast = results.get_forecast(steps=24)
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()

plt.figure(figsize=(12, 4))
plt.plot(ts, label='Observed')
plt.plot(forecast_mean, label='Forecast', color='red')
plt.fill_between(forecast_ci.index, 
                 forecast_ci.iloc[:, 0], 
                 forecast_ci.iloc[:, 1], 
                 alpha=0.2, color='red')
plt.legend()
plt.title('SARIMA(1,1,1)(1,1,1)₁₂ Forecast')
plt.show()

# ========== AUTO SARIMA (pmdarima) ==========
from pmdarima import auto_arima

auto_model = auto_arima(ts, seasonal=True, m=12, 
                        suppress_warnings=True, stepwise=True)
print(auto_model.summary())
```

---

## R Implementation

```r
library(forecast)

# ========== LOAD DATA ==========
data <- AirPassengers  # Classic seasonal dataset

# ========== DECOMPOSE ==========
decomp <- stl(data, s.window = "periodic")
plot(decomp)

# ========== FIT SARIMA ==========
model <- Arima(data, order = c(1, 1, 1), seasonal = c(1, 1, 1))
summary(model)

# ========== AUTO SARIMA ==========
auto_model <- auto.arima(data, seasonal = TRUE)
summary(auto_model)

# ========== FORECAST ==========
fc <- forecast(auto_model, h = 24)
plot(fc, main = "SARIMA Forecast with 95% CI")

# ========== DIAGNOSTICS ==========
checkresiduals(auto_model)
```

---

## Model Selection

1. **Seasonal Differencing:** Does `y[t] - y[t-m]` look stationary?
2. **ACF/PACF at seasonal lags:** Look at lags $m$, $2m$, $3m$...
3. **Use auto.arima/pmdarima:** Let the algorithm search

| ACF at lag m | PACF at lag m | Suggests |
|--------------|---------------|----------|
| Cuts off after lag m | Dies down | MA(1) seasonal: $Q=1$ |
| Dies down | Cuts off after lag m | AR(1) seasonal: $P=1$ |
| Dies down | Dies down | Mixed: $P=1, Q=1$ |

---

## Common Pitfalls

> [!warning] Real-World Traps
>
> **1. Over-differencing**
> - *Problem:* $D=1$ and $d=1$ when only one is needed
> - *Symptom:* ACF shows strong negative at lag 1
> - *Solution:* Test stationarity before differencing
>
> **2. High Seasonal Period**
> - *Problem:* $m=365$ for daily data → computationally expensive
> - *Solution:* Use Fourier terms with ARIMA errors instead
>
> **3. Ignoring Diagnostics**
> - *Solution:* Always check residuals for remaining autocorrelation

---

## Related Concepts

- [[30_Knowledge/Stats/08_Time_Series_Analysis/ARIMA Models\|ARIMA Models]] — Non-seasonal version
- [[30_Knowledge/Stats/08_Time_Series_Analysis/Stationarity (ADF & KPSS)\|Stationarity (ADF & KPSS)]] — Required assumption
- [[30_Knowledge/Stats/08_Time_Series_Analysis/Auto-Correlation (ACF & PACF)\|Auto-Correlation (ACF & PACF)]] — Model identification
- [[30_Knowledge/Stats/08_Time_Series_Analysis/Forecasting\|Forecasting]] — Application context

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Assumptions are violated
> - Alternative methods are more appropriate

---

## References

- **Book:** Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). Chapter 9. [Online](https://otexts.com/fpp3/arima.html)
- **Book:** Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). *Time Series Analysis: Forecasting and Control* (5th ed.). Wiley.
