---
{"dg-publish":true,"permalink":"/30-knowledge/stats/08-time-series-analysis/smoothing/","tags":["time-series"]}
---


## Definition

> [!abstract] Core Statement
> **Smoothing** refers to time series techniques that reduce noise and extract underlying patterns (trend, seasonality) by computing ==weighted averages of observations==. Unlike ARIMA's Box-Jenkins approach, smoothing methods are intuitive, computationally simple, and often competitive for short-term forecasting.

**Intuition (ELI5):** Imagine a noisy heartbeat monitor. Smoothing is like tracing a clean line through the jagged peaks—removing the "jitter" to see the true pattern. More recent beats matter more than old ones.

---

## Purpose

1.  **Noise Reduction:** Remove random fluctuations to see underlying signal.
2.  **Forecasting:** Generate short-term predictions with minimal model complexity.
3.  **Trend Extraction:** Identify upward/downward movements.
4.  **Decomposition:** Separate series into trend, seasonal, and residual components.

---

## When to Use

> [!success] Use Smoothing When...
> - You need **quick, interpretable forecasts**.
> - Series has **clear trend and/or seasonality**.
> - **Short-term forecasts** (1-12 periods ahead).
> - Computational resources are limited.
> - You want a **benchmark model** before complex methods.

> [!failure] Consider Alternatives When...
> - Series has **complex autocorrelation** structure (use [[30_Knowledge/Stats/08_Time_Series_Analysis/ARIMA Models\|ARIMA Models]]).
> - You need **confidence intervals** with theoretical backing.
> - **Long-term forecasts** are needed (smoothing reverts to flat).
> - Multiple predictors are available (use regression).

---

## Theoretical Background

### Types of Smoothing Methods

| Method | Components | Use Case |
|--------|------------|----------|
| **Simple Moving Average (SMA)** | None | Stationary series |
| **Simple Exponential Smoothing (SES)** | Level only | No trend, no seasonality |
| **Holt's Linear** | Level + Trend | Trend, no seasonality |
| **Holt-Winters** | Level + Trend + Season | Trend + Seasonality |
| **ETS Framework** | Error, Trend, Season | Automatic model selection |

---

### 1. Simple Moving Average (SMA)

Average of the last $k$ observations:

$$
\hat{Y}_t = \frac{1}{k} \sum_{i=0}^{k-1} Y_{t-i}
$$

- **Pros:** Simple, removes high-frequency noise.
- **Cons:** Lags behind trend, equal weights (old = recent).

---

### 2. Simple Exponential Smoothing (SES)

Weighted average where recent observations get more weight:

$$
\hat{Y}_{t+1} = \alpha Y_t + (1 - \alpha) \hat{Y}_t
$$

Or equivalently:
$$
\hat{Y}_{t+1} = \hat{Y}_t + \alpha (Y_t - \hat{Y}_t)
$$

Where $\alpha$ (0 < α < 1) is the **smoothing parameter**:
- **α close to 1:** More weight on recent data (reactive).
- **α close to 0:** More weight on historical average (stable).

**Forecast:** All future forecasts are flat at the last smoothed value.

---

### 3. Holt's Linear Method (Double Exponential Smoothing)

Adds a **trend component**:

$$
\begin{aligned}
L_t &= \alpha Y_t + (1-\alpha)(L_{t-1} + T_{t-1}) \quad \text{(Level)} \\
T_t &= \beta (L_t - L_{t-1}) + (1-\beta) T_{t-1} \quad \text{(Trend)} \\
\hat{Y}_{t+h} &= L_t + h \cdot T_t
\end{aligned}
$$

Where:
- $\alpha$ = level smoothing parameter
- $\beta$ = trend smoothing parameter
- $h$ = forecast horizon

---

### 4. Holt-Winters Method (Triple Exponential Smoothing)

Adds **seasonality** with period $m$:

**Additive Version** (constant seasonal amplitude):
$$
\begin{aligned}
L_t &= \alpha (Y_t - S_{t-m}) + (1-\alpha)(L_{t-1} + T_{t-1}) \\
T_t &= \beta (L_t - L_{t-1}) + (1-\beta) T_{t-1} \\
S_t &= \gamma (Y_t - L_t) + (1-\gamma) S_{t-m} \\
\hat{Y}_{t+h} &= L_t + h \cdot T_t + S_{t+h-m}
\end{aligned}
$$

**Multiplicative Version** (seasonal amplitude proportional to level):
$$
\begin{aligned}
L_t &= \alpha \frac{Y_t}{S_{t-m}} + (1-\alpha)(L_{t-1} + T_{t-1}) \\
T_t &= \beta (L_t - L_{t-1}) + (1-\beta) T_{t-1} \\
S_t &= \gamma \frac{Y_t}{L_t} + (1-\gamma) S_{t-m} \\
\hat{Y}_{t+h} &= (L_t + h \cdot T_t) \times S_{t+h-m}
\end{aligned}
$$

### 5. ETS Framework

**E**rror, **T**rend, **S**easonality taxonomy:

| Component | Options |
|-----------|---------|
| Error (E) | Additive (A), Multiplicative (M) |
| Trend (T) | None (N), Additive (A), Additive Damped (Ad), Multiplicative (M) |
| Season (S) | None (N), Additive (A), Multiplicative (M) |

**Example:** ETS(A,Ad,M) = Additive errors, Damped additive trend, Multiplicative seasonality.

---

## Assumptions

- [ ] **Stationarity (SMA, SES):** For SES, series should have no trend or seasonality.
- [ ] **Constant Variance (Additive):** Seasonal fluctuations don't scale with level.
- [ ] **Multiplicative:** Seasonal amplitude scales with level.
- [ ] **No Structural Breaks:** Patterns are stable over time.

---

## Limitations

> [!warning] Pitfalls
> 1. **Flat Forecasts (SES):** All future predictions are the same value.
> 2. **Initial Values:** Results depend on how L₀, T₀, S₀ are set.
> 3. **Over-smoothing:** Low α can miss important changes.
> 4. **Under-smoothing:** High α is overly reactive to noise.
> 5. **Trend Explosion:** Holt's linear can project unrealistic long-term trends.

---

## Python Implementation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose

# ========== GENERATE SAMPLE DATA ==========
np.random.seed(42)
n = 120  # 10 years of monthly data

# Create trend + seasonality + noise
time = np.arange(n)
trend = 100 + 0.5 * time
seasonality = 10 * np.sin(2 * np.pi * time / 12)
noise = np.random.normal(0, 3, n)
y = trend + seasonality + noise

# Create time series
dates = pd.date_range(start='2015-01-01', periods=n, freq='MS')
ts = pd.Series(y, index=dates)

# Split train/test
train = ts[:-12]
test = ts[-12:]

print("Data shape:", ts.shape)
print(ts.head())

# ========== 1. SIMPLE MOVING AVERAGE ==========
window = 12
sma = ts.rolling(window=window).mean()

plt.figure(figsize=(12, 4))
plt.plot(ts, label='Original', alpha=0.7)
plt.plot(sma, label=f'{window}-Month SMA', color='red')
plt.title('Simple Moving Average')
plt.legend()
plt.show()

# ========== 2. SIMPLE EXPONENTIAL SMOOTHING ==========
ses_model = SimpleExpSmoothing(train, initialization_method='heuristic')
ses_fit = ses_model.fit(smoothing_level=0.3, optimized=False)
ses_forecast = ses_fit.forecast(12)

print("\n=== Simple Exponential Smoothing ===")
print(f"Alpha: {ses_fit.params['smoothing_level']:.3f}")
print(f"Forecast (flat): {ses_forecast.iloc[0]:.2f}")

# ========== 3. HOLT'S LINEAR METHOD ==========
holt_model = ExponentialSmoothing(train, trend='add', seasonal=None,
                                   initialization_method='estimated')
holt_fit = holt_model.fit()
holt_forecast = holt_fit.forecast(12)

print("\n=== Holt's Linear Method ===")
print(f"Alpha: {holt_fit.params['smoothing_level']:.3f}")
print(f"Beta: {holt_fit.params['smoothing_trend']:.3f}")

# ========== 4. HOLT-WINTERS ==========
# Additive seasonality
hw_add = ExponentialSmoothing(train, trend='add', seasonal='add', 
                               seasonal_periods=12,
                               initialization_method='estimated')
hw_add_fit = hw_add.fit()
hw_add_forecast = hw_add_fit.forecast(12)

# Multiplicative seasonality
hw_mul = ExponentialSmoothing(train, trend='add', seasonal='mul',
                               seasonal_periods=12,
                               initialization_method='estimated')
hw_mul_fit = hw_mul.fit()
hw_mul_forecast = hw_mul_fit.forecast(12)

print("\n=== Holt-Winters (Additive) ===")
print(f"Alpha: {hw_add_fit.params['smoothing_level']:.3f}")
print(f"Beta: {hw_add_fit.params['smoothing_trend']:.3f}")
print(f"Gamma: {hw_add_fit.params['smoothing_seasonal']:.3f}")
print(f"AIC: {hw_add_fit.aic:.2f}")

# ========== 5. AUTOMATIC ETS ==========
from statsmodels.tsa.api import ETSModel

ets_model = ETSModel(train, error='add', trend='add', seasonal='add',
                      seasonal_periods=12, damped_trend=True)
ets_fit = ets_model.fit()
ets_forecast = ets_fit.forecast(12)

print("\n=== ETS(A,Ad,A) - Damped Trend ===")
print(ets_fit.summary().tables[1])

# ========== VISUALIZATION ==========
fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(train, label='Train', color='blue')
ax.plot(test, label='Test', color='green')
ax.plot(ses_forecast, label='SES', linestyle='--', color='orange')
ax.plot(holt_forecast, label='Holt Linear', linestyle='--', color='red')
ax.plot(hw_add_forecast, label='Holt-Winters', linestyle='--', color='purple')
ax.plot(ets_forecast, label='ETS (Damped)', linestyle='--', color='brown')

ax.axvline(train.index[-1], color='gray', linestyle=':')
ax.set_title('Smoothing Methods Comparison')
ax.legend()
ax.grid(alpha=0.3)
plt.show()

# ========== FORECAST ACCURACY ==========
from sklearn.metrics import mean_absolute_error, mean_squared_error

results = pd.DataFrame({
    'Method': ['SES', 'Holt', 'HW-Add', 'ETS'],
    'MAE': [
        mean_absolute_error(test, ses_forecast),
        mean_absolute_error(test, holt_forecast),
        mean_absolute_error(test, hw_add_forecast),
        mean_absolute_error(test, ets_forecast)
    ],
    'RMSE': [
        np.sqrt(mean_squared_error(test, ses_forecast)),
        np.sqrt(mean_squared_error(test, holt_forecast)),
        np.sqrt(mean_squared_error(test, hw_add_forecast)),
        np.sqrt(mean_squared_error(test, ets_forecast))
    ]
})
print("\n=== Forecast Accuracy ===")
print(results.round(2))
```

---

## R Implementation

```r
library(forecast)
library(ggplot2)

# ========== GENERATE SAMPLE DATA ==========
set.seed(42)
n <- 120

time <- 1:n
trend <- 100 + 0.5 * time
seasonality <- 10 * sin(2 * pi * time / 12)
noise <- rnorm(n, 0, 3)
y <- trend + seasonality + noise

ts_data <- ts(y, start = c(2015, 1), frequency = 12)
train <- window(ts_data, end = c(2024, 12))
test <- window(ts_data, start = c(2025, 1))

# ========== 1. SIMPLE MOVING AVERAGE ==========
library(zoo)
sma <- rollmean(ts_data, k = 12, align = "right", fill = NA)

autoplot(ts_data) +
  autolayer(sma, series = "12-Month SMA", color = "red") +
  labs(title = "Simple Moving Average")

# ========== 2. SIMPLE EXPONENTIAL SMOOTHING ==========
ses_fit <- ses(train, h = 12)
summary(ses_fit)

autoplot(ses_fit) +
  autolayer(test, series = "Actual") +
  labs(title = "Simple Exponential Smoothing")

# ========== 3. HOLT'S LINEAR METHOD ==========
holt_fit <- holt(train, h = 12)
summary(holt_fit)

autoplot(holt_fit) +
  autolayer(test, series = "Actual") +
  labs(title = "Holt's Linear Method")

# Damped trend version
holt_damped <- holt(train, h = 12, damped = TRUE)

# ========== 4. HOLT-WINTERS ==========
# Additive
hw_add <- hw(train, h = 12, seasonal = "additive")
summary(hw_add)

# Multiplicative
hw_mul <- hw(train, h = 12, seasonal = "multiplicative")

autoplot(hw_add) +
  autolayer(test, series = "Actual") +
  labs(title = "Holt-Winters (Additive)")

# ========== 5. AUTOMATIC ETS ==========
ets_fit <- ets(train)
summary(ets_fit)

ets_forecast <- forecast(ets_fit, h = 12)

autoplot(ets_forecast) +
  autolayer(test, series = "Actual") +
  labs(title = paste("Automatic ETS:", ets_fit$method))

# ========== DECOMPOSITION ==========
decomp <- stl(ts_data, s.window = "periodic")
autoplot(decomp) + labs(title = "STL Decomposition")

# ========== FORECAST ACCURACY ==========
accuracy_table <- rbind(
  SES = accuracy(ses_fit, test)[2, ],
  Holt = accuracy(holt_fit, test)[2, ],
  HW_Add = accuracy(hw_add, test)[2, ],
  ETS = accuracy(ets_forecast, test)[2, ]
)

print(round(accuracy_table[, c("MAE", "RMSE", "MAPE")], 2))

# ========== COMPARISON PLOT ==========
autoplot(train) +
  autolayer(test, series = "Actual", PI = FALSE) +
  autolayer(ses_fit, series = "SES", PI = FALSE) +
  autolayer(holt_fit, series = "Holt", PI = FALSE) +
  autolayer(hw_add, series = "HW", PI = FALSE) +
  autolayer(ets_forecast, series = "ETS", PI = FALSE) +
  labs(title = "Smoothing Methods Comparison") +
  guides(color = guide_legend(title = "Method"))
```

---

## Worked Numerical Example

> [!example] Monthly Sales Forecasting
> **Data:** Last 6 months of sales: 100, 110, 108, 115, 120, 125.
> **Task:** Forecast month 7 using SES with α = 0.3.
> 
> **Step 1: Initialize**
> $\hat{Y}_1 = Y_1 = 100$ (First observation as starting point)
> 
> **Step 2: Apply SES formula**
> 
> | Month | $Y_t$ | $\hat{Y}_t$ | Calculation |
> |-------|-------|-------------|-------------|
> | 1 | 100 | 100.0 | Initial |
> | 2 | 110 | 100.0 | $0.3(100) + 0.7(100)$ |
> | 3 | 108 | 103.0 | $0.3(110) + 0.7(100)$ |
> | 4 | 115 | 104.5 | $0.3(108) + 0.7(103)$ |
> | 5 | 120 | 107.7 | $0.3(115) + 0.7(104.5)$ |
> | 6 | 125 | 111.4 | $0.3(120) + 0.7(107.7)$ |
> | **7** | ? | **115.5** | $0.3(125) + 0.7(111.4)$ |
> 
> **Forecast for Month 7:** 115.5
> 
> **Insight:** The forecast (115.5) lags behind the actual trend. With α = 0.3, we're being conservative. Higher α would react faster but be noisier.

---

## Interpretation Guide

| Parameter | Value | Interpretation | Edge Case |
|-----------|-------|----------------|-----------|
| α (level) | 0.9 | Very reactive; follows data closely | Prone to noise |
| α (level) | 0.1 | Very stable; slow to adapt | May miss trends |
| β (trend) | 0.8 | Trend updates quickly | Unstable projections |
| γ (season) | 0.1 | Seasonal pattern stable over time | |
| AIC | Lower better | Model complexity trade-off | Compare same family |
| Damping φ | 0.98 | Trend flattens slowly | φ < 0.8 = rapid damping |

---

## Common Pitfall Example

> [!warning] Choosing Wrong Seasonal Type
> **Scenario:** Forecasting hotel bookings with summer peaks.
> 
> **Data Pattern:**
> - 2015: Peak = 1000 (Baseline = 500) → Swing = 500
> - 2020: Peak = 2000 (Baseline = 1000) → Swing = 1000
> 
> **Mistake:** Using additive Holt-Winters (constant swing of ~750).
> 
> **Problem:** Model predicts 2025 peak = Baseline + 750, but actual swing scales with level.
> 
> **Correct Approach:** Multiplicative (swing = 100% of baseline).
> 
> **Rule of Thumb:**
> - Constant seasonal amplitude → Additive
> - Seasonal amplitude grows with level → Multiplicative

---

## Related Concepts

**Time Series Methods:**
- [[30_Knowledge/Stats/08_Time_Series_Analysis/ARIMA Models\|ARIMA Models]] - Box-Jenkins approach
- [[30_Knowledge/Stats/08_Time_Series_Analysis/Stationarity (ADF & KPSS)\|Stationarity (ADF & KPSS)]] - Prerequisite for some methods
- [[30_Knowledge/Stats/08_Time_Series_Analysis/Auto-Correlation (ACF & PACF)\|Auto-Correlation (ACF & PACF)]] - Pattern identification

**Components:**
- [[30_Knowledge/Stats/01_Foundations/Differencing\|Differencing]] - Trend removal
- [[30_Knowledge/Stats/08_Time_Series_Analysis/Time Series Analysis\|Time Series Analysis]] - Overview

**Applications:**
- [[30_Knowledge/Stats/08_Time_Series_Analysis/Forecasting\|Forecasting]] - General methodology
- Seasonal Decomposition - STL, X-13

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Assumptions are violated
> - Alternative methods are more appropriate

---

## References

- **Book:** Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts. (Chapters 8-9) https://otexts.com/fpp3/
- **Book:** Chatfield, C. (2003). *The Analysis of Time Series: An Introduction* (6th ed.). Chapman and Hall/CRC. [Routledge Link](https://www.routledge.com/The-Analysis-of-Time-Series-An-Introduction-with-R/Chatfield-Haustien/p/book/9781138066229)
- **Article:** Holt, C. C. (1957/2004). Forecasting seasonals and trends by exponentially weighted moving averages. *International Journal of Forecasting*. [Elsevier](https://doi.org/10.1016/j.ijforecast.2003.09.015)
- **Article:** Winters, P. R. (1960). Forecasting sales by exponentially weighted moving averages. *Management Science*. [JSTOR](https://www.jstor.org/stable/2627490)
- **Article:** Hyndman, R. J., Koehler, A. B., Snyder, R. D., & Grose, S. (2002). A state space framework for automatic forecasting using exponential smoothing methods. *International Journal of Forecasting*. [Elsevier](https://doi.org/10.1016/S0169-2070(01)00129-8)
