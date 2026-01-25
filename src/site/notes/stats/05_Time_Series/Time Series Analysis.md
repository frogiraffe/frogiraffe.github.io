---
{"dg-publish":true,"permalink":"/stats/05-time-series/time-series-analysis/","tags":["Time-Series","Forecasting","Analysis"]}
---


## Definition

> [!abstract] Overview
> **Time Series Analysis** involves analyzing data points collected or sequenced over time. The goal is to extract meaningful statistics (Analysis) or predict future values (Forecasting).

Key difference from standard regression: **Observations are NOT independent.** Today's stock price depends on yesterday's.

---

## 1. Components of Time Series

A time series $Y_t$ is often decomposed into:

1.  **Trend ($T_t$):** Long-term movement (e.g., Global warming temps rising).
2.  **Seasonality ($S_t$):** Repeating patterns at fixed intervals (e.g., Ice cream sales in Summer).
3.  **Cyclical ($C_t$):** Fluctuations not of fixed frequency (e.g., Economic recessions).
4.  **Residual / Noise ($\epsilon_t$):** Random properties.

**Additive Model:** $Y_t = T_t + S_t + \epsilon_t$
**Multiplicative Model:** $Y_t = T_t \times S_t \times \epsilon_t$

---

## 2. Stationarity

A time series is **Stationary** if its statistical properties (Mean, Variance) do not change over time.

- Most models (ARIMA) **require** stationarity.
- If data is not stationary, we apply **Differencing** ($Y_t - Y_{t-1}$) or Log Transforms.
- Test: [[stats/05_Time_Series/Stationarity (ADF & KPSS)\|Stationarity (ADF & KPSS)]].

---

## 3. Python Decomposition

```python
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

# Load Data (Index must be Datetime)
df = pd.read_csv('sales.csv', parse_dates=['Date'], index_col='Date')

# Decompose
result = seasonal_decompose(df['Sales'], model='additive', period=12)

# Plot
result.plot()
plt.show()

# Access components
trend = result.trend
seasonal = result.seasonal
resid = result.resid
```

---

## 4. Common Models

- [[stats/05_Time_Series/Auto-Correlation (ACF & PACF)\|Auto-Correlation (ACF & PACF)]] - Measuring relationship with past values.
- [[stats/05_Time_Series/ARIMA Models\|ARIMA Models]] - The standard forecasting tool.
- [[stats/05_Time_Series/GARCH Models\|GARCH Models]] - Forecasting Volatility (Variance).
- [[stats/05_Time_Series/Vector Autoregression (VAR)\|Vector Autoregression (VAR)]] - Multivariate time series.

---

## Related Concepts

- [[stats/04_Machine_Learning/Cross-Validation\|Cross-Validation]] (Time Series Split - Don't shuffle!)
- [[stats/05_Time_Series/Smoothing\|Smoothing]] (Moving Averages)
