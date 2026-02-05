---
{"dg-publish":true,"permalink":"/stats/08-time-series-analysis/prophet/","tags":["time-series","forecasting","machine-learning"]}
---


## Definition

> [!abstract] Core Statement
> **Prophet** is an open-source forecasting library by Meta (Facebook) designed for ==business time series with strong seasonality==. It's robust to missing data, trend changes, and outliers.

---

> [!tip] Intuition (ELI5): The Business Calendar
> Prophet understands that Fridays behave differently than Mondays, December differently than July, and holidays differently than normal days. It builds all these patterns into one model automatically.

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Multiple Seasonality** | Daily, weekly, yearly patterns |
| **Holiday Effects** | Built-in holiday handling |
| **Trend Changepoints** | Automatic detection of trend shifts |
| **Robustness** | Handles missing data, outliers |
| **Uncertainty Intervals** | Built-in confidence intervals |

---

## The Model

Prophet decomposes time series as:

$$
y(t) = g(t) + s(t) + h(t) + \epsilon_t
$$

| Component | Description |
|-----------|-------------|
| $g(t)$ | Trend (linear or logistic growth) |
| $s(t)$ | Seasonality (Fourier series) |
| $h(t)$ | Holiday effects |
| $\epsilon_t$ | Error term |

---

## Python Implementation

```python
from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt

# ========== PREPARE DATA ==========
# Prophet requires columns: 'ds' (date) and 'y' (value)
df = pd.DataFrame({
    'ds': pd.date_range(start='2020-01-01', periods=365*3, freq='D'),
    'y': data_values
})

# ========== BASIC MODEL ==========
model = Prophet()
model.fit(df)

# ========== FORECAST ==========
future = model.make_future_dataframe(periods=90)  # 90 days ahead
forecast = model.predict(future)

# ========== PLOT ==========
fig = model.plot(forecast)
plt.title('Prophet Forecast')
plt.show()

# Components plot
fig2 = model.plot_components(forecast)
plt.show()

# ========== WITH SEASONALITY & HOLIDAYS ==========
from prophet.make_holidays import make_holidays_df

model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    seasonality_mode='multiplicative',  # or 'additive'
    changepoint_prior_scale=0.05,       # Trend flexibility
    holidays=holidays_df
)

# Add custom seasonality
model.add_seasonality(name='monthly', period=30.5, fourier_order=5)

# Add custom regressor
df['is_promotion'] = df['ds'].isin(promo_dates).astype(int)
model.add_regressor('is_promotion')

model.fit(df)

# ========== CROSS-VALIDATION ==========
from prophet.diagnostics import cross_validation, performance_metrics

cv_results = cross_validation(
    model, 
    initial='365 days',    # Training period
    period='90 days',      # Spacing between cutoffs
    horizon='30 days'      # Forecast horizon
)

metrics = performance_metrics(cv_results)
print(metrics[['horizon', 'mape', 'rmse', 'coverage']].tail())

# Plot CV results
from prophet.plot import plot_cross_validation_metric
fig = plot_cross_validation_metric(cv_results, metric='mape')
```

---

## R Implementation

```r
library(prophet)

# ========== PREPARE DATA ==========
df <- data.frame(
  ds = seq(as.Date("2020-01-01"), by = "day", length.out = 365*3),
  y = data_values
)

# ========== FIT MODEL ==========
model <- prophet(df)

# ========== FORECAST ==========
future <- make_future_dataframe(model, periods = 90)
forecast <- predict(model, future)

# ========== PLOT ==========
plot(model, forecast)
prophet_plot_components(model, forecast)

# ========== WITH HOLIDAYS ==========
holidays <- data.frame(
  holiday = 'black_friday',
  ds = as.Date(c('2021-11-26', '2022-11-25', '2023-11-24')),
  lower_window = -1,
  upper_window = 1
)

model <- prophet(holidays = holidays)
model <- add_seasonality(model, name = 'monthly', period = 30.5, fourier.order = 5)
model <- fit.prophet(model, df)
```

---

## Key Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `changepoint_prior_scale` | 0.05 | Trend flexibility (higher = more flexible) |
| `seasonality_prior_scale` | 10 | Seasonality strength |
| `holidays_prior_scale` | 10 | Holiday effect strength |
| `seasonality_mode` | additive | 'additive' or 'multiplicative' |
| `growth` | 'linear' | 'linear' or 'logistic' (for saturation) |

---

## When to Use Prophet

> [!success] Prophet Works Well For...
> - **Business metrics** with strong seasonal patterns
> - **Daily data** with weekly and yearly cycles
> - Data with **holidays and special events**
> - When you need **fast, automatic forecasting**

> [!failure] Consider Alternatives When...
> - Very short time series (< 2 years)
> - High-frequency data (sub-hourly)
> - Need probabilistic forecasts → Use Bayesian methods
> - Complex multivariate relationships → [[stats/08_Time_Series_Analysis/ARIMA Models\|ARIMA Models]] + exogenous

---

## Related Concepts

- [[stats/08_Time_Series_Analysis/Forecasting\|Forecasting]] — General forecasting overview
- [[stats/08_Time_Series_Analysis/SARIMA\|SARIMA]] — Traditional alternative
- [[stats/08_Time_Series_Analysis/Time Series Analysis\|Time Series Analysis]] — Broader context
- [[stats/08_Time_Series_Analysis/Stationarity (ADF & KPSS)\|Stationarity (ADF & KPSS)]] — Not required for Prophet

---

## References

- **Paper:** Taylor, S. J., & Letham, B. (2018). Forecasting at scale. *The American Statistician*, 72(1), 37-45. [PDF](https://peerj.com/preprints/3190/)
- **Documentation:** [Prophet Docs](https://facebook.github.io/prophet/)
