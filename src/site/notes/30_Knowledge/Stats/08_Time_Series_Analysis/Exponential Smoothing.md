---
{"dg-publish":true,"permalink":"/30-knowledge/stats/08-time-series-analysis/exponential-smoothing/","tags":["time-series"]}
---


## Definition

> [!abstract] Core Statement
> **Exponential Smoothing** is a family of forecasting methods that use ==weighted averages of past observations==, with weights decaying exponentially over time. More recent observations get higher weights.

---

> [!tip] Intuition (ELI5): The Fading Memory
> Imagine remembering recent events vividly but older events fuzzily. Exponential smoothing does the same with data — yesterday matters most, last month matters less, last year is a faint whisper.

---

## Types of Exponential Smoothing

| Method | Trend | Seasonality | Notation |
|--------|-------|-------------|----------|
| **Simple (SES)** | ✗ | ✗ | (N,N) |
| **Holt's Linear** | ✓ | ✗ | (A,N) or (Ad,N) |
| **Holt-Winters Additive** | ✓ | ✓ Additive | (A,A) |
| **Holt-Winters Multiplicative** | ✓ | ✓ Multiplicative | (A,M) |

**ETS Notation:** (Error, Trend, Seasonal) with:
- N = None, A = Additive, M = Multiplicative, Ad = Additive damped

---

## Simple Exponential Smoothing (SES)

For data without trend or seasonality:

$$
\hat{y}_{t+1} = \alpha y_t + (1-\alpha) \hat{y}_t
$$

Where $\alpha \in [0,1]$ is the smoothing parameter:
- $\alpha$ close to 1 → more weight on recent observations
- $\alpha$ close to 0 → smoother, more historical influence

---

## Holt-Winters (Triple Exponential Smoothing)

For data with trend AND seasonality:

**Level:** $\ell_t = \alpha(y_t - s_{t-m}) + (1-\alpha)(\ell_{t-1} + b_{t-1})$

**Trend:** $b_t = \beta(\ell_t - \ell_{t-1}) + (1-\beta)b_{t-1}$

**Seasonal:** $s_t = \gamma(y_t - \ell_t) + (1-\gamma)s_{t-m}$

**Forecast:** $\hat{y}_{t+h} = \ell_t + hb_t + s_{t+h-m}$

---

## Python Implementation

```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ========== GENERATE DATA ==========
np.random.seed(42)
t = np.arange(120)
trend = 0.1 * t
seasonal = 10 * np.sin(2 * np.pi * t / 12)
noise = np.random.normal(0, 2, 120)
y = 50 + trend + seasonal + noise

dates = pd.date_range(start='2015-01-01', periods=120, freq='M')
ts = pd.Series(y, index=dates)

# ========== SIMPLE EXPONENTIAL SMOOTHING ==========
ses_model = SimpleExpSmoothing(ts).fit(smoothing_level=0.2)
ses_forecast = ses_model.forecast(12)

# ========== HOLT'S LINEAR (TREND) ==========
holt_model = ExponentialSmoothing(ts, trend='add').fit()
holt_forecast = holt_model.forecast(12)

# ========== HOLT-WINTERS (TREND + SEASONALITY) ==========
hw_add = ExponentialSmoothing(
    ts, 
    trend='add', 
    seasonal='add', 
    seasonal_periods=12
).fit()
hw_add_forecast = hw_add.forecast(12)

hw_mul = ExponentialSmoothing(
    ts, 
    trend='add', 
    seasonal='mul', 
    seasonal_periods=12
).fit()
hw_mul_forecast = hw_mul.forecast(12)

# ========== PLOT ==========
plt.figure(figsize=(14, 6))
plt.plot(ts, label='Observed', color='black')
plt.plot(ses_forecast, label='SES', linestyle='--')
plt.plot(holt_forecast, label='Holt Linear', linestyle='--')
plt.plot(hw_add_forecast, label='Holt-Winters Add', linestyle='--')
plt.plot(hw_mul_forecast, label='Holt-Winters Mul', linestyle='--')
plt.legend()
plt.title('Exponential Smoothing Forecasts')
plt.show()

# ========== AUTOMATIC ETS ==========
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

ets = ETSModel(ts, error='add', trend='add', seasonal='add', 
               seasonal_periods=12).fit()
print(ets.summary())
```

---

## R Implementation

```r
library(forecast)

# ========== SIMPLE EXPONENTIAL SMOOTHING ==========
ses_model <- ses(ts, h = 12)
plot(ses_model)

# ========== HOLT'S LINEAR ==========
holt_model <- holt(ts, h = 12)
plot(holt_model)

# ========== HOLT-WINTERS ==========
hw_add <- hw(ts, seasonal = "additive", h = 12)
hw_mul <- hw(ts, seasonal = "multiplicative", h = 12)

# ========== AUTOMATIC ETS ==========
ets_model <- ets(ts)
summary(ets_model)

forecast_ets <- forecast(ets_model, h = 12)
plot(forecast_ets)
```

---

## Choosing Parameters

| Parameter | Symbol | Typical Range | Effect |
|-----------|--------|---------------|--------|
| **Level smoothing** | α | 0.1-0.3 | Higher = more responsive |
| **Trend smoothing** | β | 0.01-0.2 | Higher = faster trend adaptation |
| **Seasonal smoothing** | γ | 0.1-0.5 | Higher = faster seasonal changes |
| **Damping** | φ | 0.8-0.98 | Dampens trend toward flat |

---

## Additive vs Multiplicative Seasonality

| Type | When to Use | Example |
|------|-------------|---------|
| **Additive** | Seasonal variation is constant | +1000 sales in December regardless of base |
| **Multiplicative** | Seasonal variation proportional | +20% sales in December |

**Rule of thumb:** If seasonal swings grow with the level, use multiplicative.

---

## Common Pitfalls

> [!warning] Real-World Traps
>
> **1. Wrong Seasonality Type**
> - *Problem:* Using additive when multiplicative is needed
> - *Solution:* Plot data, check if amplitude changes with level
>
> **2. Missing Trend When Present**
> - *Problem:* SES for trending data → biased forecasts
> - *Solution:* Use Holt's method or let ETS auto-select
>
> **3. Over-smoothing**
> - *Problem:* Very low α → forecasts lag reality
> - *Solution:* Use cross-validation to tune parameters

---

## Related Concepts

- [[30_Knowledge/Stats/08_Time_Series_Analysis/ARIMA Models\|ARIMA Models]] — Alternative approach
- [[30_Knowledge/Stats/08_Time_Series_Analysis/SARIMA\|SARIMA]] — ARIMA with seasonality
- [[30_Knowledge/Stats/08_Time_Series_Analysis/Smoothing\|Smoothing]] — General smoothing techniques
- [[30_Knowledge/Stats/08_Time_Series_Analysis/Forecasting\|Forecasting]] — Broader context

---

## When to Use

> [!success] Use Exponential Smoothing When...
> - Refer to standard documentation
> - Refer to standard documentation

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Assumptions are violated
> - Alternative methods are more appropriate

---

## References

- **Book:** Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). Chapter 8. [Online](https://otexts.com/fpp3/expsmooth.html)
- **Historical:** Holt, C. C. (1957). Forecasting trends and seasonals by exponentially weighted moving averages. *ONR Research Memorandum*.
