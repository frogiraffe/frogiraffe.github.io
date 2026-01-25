---
{"dg-publish":true,"permalink":"/stats/05-time-series/arima-models/","tags":["Time-Series","Forecasting","Econometrics"]}
---


# ARIMA Models

## Definition

> [!abstract] Core Statement
> **ARIMA (AutoRegressive Integrated Moving Average)** is a class of models for forecasting ==stationary time series==. It combines three components:
> - **AR(p):** Autoregressive (past values).
> - **I(d):** Integrated (differencing for stationarity).
> - **MA(q):** Moving Average (past errors).

---

## Purpose

1.  Forecast future values of a time series.
2.  Understand the temporal structure of data.
3.  Benchmark model for univariate forecasting.

---

## When to Use

> [!success] Use ARIMA When...
> - Data is a **univariate time series**.
> - Series is **stationary** (or can be made so via differencing).
> - Goal is **short-term forecasting**.

> [!failure] Alternatives
> - **Multiple predictors:** Use [[stats/05_Time_Series/Vector Autoregression (VAR)\|Vector Autoregression (VAR)]] or regression.
> - **Volatility clustering:** Use [[stats/05_Time_Series/GARCH Models\|GARCH Models]].
> - **Seasonality:** Use SARIMA (Seasonal ARIMA).

---

## Theoretical Background

### The Components

| Component | Notation | Meaning |
|-----------|----------|---------|
| **AR(p)** | $\phi_1 Y_{t-1} + \dots + \phi_p Y_{t-p}$ | Regress on past values. |
| **I(d)** | $\nabla^d Y_t$ | Difference $d$ times to achieve stationarity. |
| **MA(q)** | $\theta_1 \varepsilon_{t-1} + \dots + \theta_q \varepsilon_{t-q}$ | Regress on past errors. |

### Model Equation (ARIMA(p,d,q))

$$
\nabla^d Y_t = c + \phi_1 Y_{t-1} + \dots + \phi_p Y_{t-p} + \varepsilon_t + \theta_1 \varepsilon_{t-1} + \dots + \theta_q \varepsilon_{t-q}
$$

### Identification (Box-Jenkins Method)

1.  **Plot ACF/PACF:** Use patterns to guess $p$ and $q$.
2.  **Test Stationarity:** [[stats/05_Time_Series/Stationarity (ADF & KPSS)\|Stationarity (ADF & KPSS)]]. Apply differencing if needed.
3.  **Fit Model:** Estimate parameters.
4.  **Diagnose Residuals:** Should be white noise (no autocorrelation).
5.  **Forecast.**

---

## Assumptions

- [ ] **Stationarity:** After differencing, the series has constant mean and variance.
- [ ] **No Structural Breaks.**
- [ ] **Residuals are White Noise:** Check with Ljung-Box test.

---

## Limitations

> [!warning] Pitfalls
> 1.  **Requires Stationarity:** Non-stationary series must be differenced.
> 2.  **Univariate:** Does not incorporate external predictors. Use ARIMAX or VAR.
> 3.  **Short-Term Forecasts Only:** Long-term forecasts revert to the mean.
> 4.  **Manual Order Selection:** Box-Jenkins requires expertise. Use `auto.arima()` in R.

---

## Python Implementation

```python
import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Auto-ARIMA (Automatic Order Selection)
auto_model = pm.auto_arima(series, seasonal=False, stepwise=True, trace=True)
print(auto_model.summary())

# Manual ARIMA
model = ARIMA(series, order=(2, 1, 2)).fit()
print(model.summary())

# Forecast
forecast = model.get_forecast(steps=10)
print(forecast.predicted_mean)
forecast.plot_predict()
plt.show()
```

---

## R Implementation

```r
library(forecast)

# Auto-ARIMA (Best practice)
fit <- auto.arima(ts_data)
summary(fit)

# Forecast
fc <- forecast(fit, h = 12)
plot(fc)

# Check Residuals (Should show no autocorrelation)
checkresiduals(fit)
```

---

## Worked Numerical Example

> [!example] Forecasting Weekly Sales
> **Data:** 100 weeks of sales data.
> **Process:**
> 1. **Visual Check:** Trend is upward (Non-stationary).
> 2. **Differencing (d=1):** Values become stationary ($\Delta Y_t = Y_t - Y_{t-1}$).
> 3. **ACF Plot:** Sharp cutoff after Lag 1. Suggests MA(1).
> 4. **PACF Plot:** Exponential decay. Suggests MA(1).
> 
> **Model:** ARIMA(0, 1, 1).
> - Equation: $Y_t - Y_{t-1} = \theta_1 \epsilon_{t-1} + \epsilon_t$.
> - This is "Simple Exponential Smoothing".
> 
> **Forecast:** Next week's sales are a weighted average of recent sales, with more weight on the most recent week.

---

## Interpretation Guide

| Output | Interpretation | Edge Case Notes |
|--------|----------------|-----------------|
| ARIMA(1,1,1) | 1 AR term, 1 diff, 1 MA term. | Standard robust baseline. |
| ARIMA(0,1,0) | "Random Walk". Forecast = Last Value. | Common for stock prices. Hard to beat. |
| AIC = 300 vs 350 | AIC 300 is superior. | Improvement > 2 is significant. |
| p-value > 0.05 (Ljung-Box) | Residuals are white noise (Good). | Model has captured all signal. |
| Coef close to 1 | Unit root issue? | Series might need more differencing. |

---

## Common Pitfall Example

> [!warning] The Stock Price Fallacy
> **Scenario:** Trying to forecast Google Stock Price ($P_t$) using ARIMA.
> 
> **Mistake:** Fit ARIMA(2,1,2). Get great "in-sample" fit ($R^2=0.99$).
> 
> **Reality Check:** 
> - Plotting the forecast shows it just lags the real price by 1 day.
> - The best predictor of tomorrow's price is often today's price (Random Walk).
> - ARIMA cannot predict "shocks" or news.
> 
> **Lesson:** ARIMA works best for **inertial** systems (sales, temperature, inventory), not efficient markets (stocks).

---

## Related Concepts

- [[stats/05_Time_Series/Stationarity (ADF & KPSS)\|Stationarity (ADF & KPSS)]] - Prerequisite.
- [[stats/05_Time_Series/Vector Autoregression (VAR)\|Vector Autoregression (VAR)]] - Multivariate extension.
- [[stats/05_Time_Series/GARCH Models\|GARCH Models]] - For volatility.
- [[stats/05_Time_Series/Granger Causality\|Granger Causality]] - Testing predictive relationships.
