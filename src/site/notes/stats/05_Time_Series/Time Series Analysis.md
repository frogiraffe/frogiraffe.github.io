---
{"dg-publish":true,"permalink":"/stats/05-time-series/time-series-analysis/","tags":["Time-Series","Forecasting","Analysis","Temporal-Data"]}
---


## Definition

> [!abstract] Core Statement
> **Time Series Analysis** involves analyzing data points collected **sequentially over time** to extract meaningful patterns, identify structure, and forecast future values. Unlike cross-sectional data, time series observations are **not independent** — today's value depends on yesterday's.

**Intuition (ELI5):** Regular statistics treats each observation like a marble in a bag — independent, interchangeable. Time series treats data like a movie — sequence matters, each frame connects to the next, and you can predict what happens next based on what happened before.

**Key Applications:**
- Stock price forecasting
- Demand planning & inventory
- Weather prediction
- Economic indicators
- Anomaly detection in sensors

---

## When to Use

> [!success] Use Time Series Methods When...
> - Data is collected at **regular intervals** over time.
> - **Temporal dependencies** exist (today's value relates to past values).
> - You want to **forecast** future values.
> - You want to **decompose** data into trend, seasonality, and residuals.
> - You need to detect **anomalies** or **change points**.

> [!failure] Do NOT Use Time Series Methods When...
> - Observations are **independent** (no temporal relationship) — use standard regression.
> - Time intervals are **irregular** — may need interpolation first.
> - You need **causal inference** — time series shows correlation, not causation.
> - Data is **too short** (<50 points) for reliable pattern detection.

---

## Theoretical Background

### Components of Time Series

Every time series $Y_t$ can be decomposed into:

| Component | Symbol | Description | Example |
|-----------|--------|-------------|---------|
| **Trend** | $T_t$ | Long-term direction | Global temperatures rising |
| **Seasonality** | $S_t$ | Fixed, repeating patterns | Ice cream sales peak every summer |
| **Cyclical** | $C_t$ | Variable-length fluctuations | Economic boom-bust cycles |
| **Residual** | $\epsilon_t$ | Random noise | Unexplained variation |

**Decomposition Models:**

$$
\text{Additive:} \quad Y_t = T_t + S_t + \epsilon_t
$$

$$
\text{Multiplicative:} \quad Y_t = T_t \times S_t \times \epsilon_t
$$

Use **Additive** when seasonal variation is constant. Use **Multiplicative** when seasonal variation grows with trend.

### Stationarity

> [!important] Most Time Series Models Require Stationarity
> A **stationary** series has constant mean, variance, and autocorrelation over time.

**Tests for Stationarity:**
- **Augmented Dickey-Fuller (ADF):** $H_0$: Non-stationary
- **KPSS:** $H_0$: Stationary

**Making Data Stationary:**
1. **Differencing:** $Y'_t = Y_t - Y_{t-1}$
2. **Seasonal Differencing:** $Y'_t = Y_t - Y_{t-m}$
3. **Log Transform:** Stabilizes variance
4. **Detrending:** Subtract fitted trend

### Autocorrelation

**Autocorrelation Function (ACF):** Correlation between $Y_t$ and $Y_{t-k}$

$$
\rho_k = \frac{\text{Cov}(Y_t, Y_{t-k})}{\text{Var}(Y_t)}
$$

**Partial Autocorrelation Function (PACF):** Correlation between $Y_t$ and $Y_{t-k}$, controlling for intermediate lags.

| ACF Pattern | PACF Pattern | Suggested Model |
|-------------|--------------|-----------------|
| Exponential decay | Sharp cutoff at lag $p$ | AR($p$) |
| Sharp cutoff at lag $q$ | Exponential decay | MA($q$) |
| Exponential decay | Exponential decay | ARMA($p$,$q$) |

---

## Assumptions & Diagnostics

- [ ] **Regular Intervals:** Data must be equally spaced in time.
- [ ] **Stationarity:** For ARIMA models, test with ADF/KPSS.
- [ ] **No Missing Values:** Impute or interpolate gaps.
- [ ] **Sufficient Length:** Generally need 50+ observations per seasonal period.

### Visual Diagnostics

| Plot | Purpose | What to Look For |
|------|---------|------------------|
| **Time Plot** | Overview of data | Trend, seasonality, outliers |
| **ACF/PACF** | Identify model order | Sharp cutoffs, decay patterns |
| **Seasonal Plot** | Compare seasons | Consistent patterns across years |
| **Residual Plot** | Check model fit | Random scatter (no patterns) |

---

## Implementation

### Python

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

# ========== LOAD DATA ==========
# Example: Monthly airline passengers
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
df = pd.read_csv(url, parse_dates=['Month'], index_col='Month')
df.columns = ['Passengers']

# ========== STEP 1: VISUALIZE ==========
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Time plot
axes[0, 0].plot(df)
axes[0, 0].set_title('Time Series Plot')

# Decomposition
decomposition = seasonal_decompose(df, model='multiplicative', period=12)
decomposition.plot()
plt.tight_layout()
plt.show()

# ========== STEP 2: TEST STATIONARITY ==========
# ADF Test (H0: Non-stationary)
adf_result = adfuller(df['Passengers'])
print(f"ADF Statistic: {adf_result[0]:.4f}")
print(f"p-value: {adf_result[1]:.4f}")
print("Stationary" if adf_result[1] < 0.05 else "Non-stationary")

# KPSS Test (H0: Stationary)
kpss_result = kpss(df['Passengers'], regression='c', nlags='auto')
print(f"\nKPSS Statistic: {kpss_result[0]:.4f}")
print(f"p-value: {kpss_result[1]:.4f}")

# ========== STEP 3: MAKE STATIONARY (if needed) ==========
# Differencing
df['Diff'] = df['Passengers'].diff()
df['LogDiff'] = np.log(df['Passengers']).diff()

# Check ACF/PACF of differenced series
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(df['LogDiff'].dropna(), ax=axes[0], lags=24)
plot_pacf(df['LogDiff'].dropna(), ax=axes[1], lags=24)
plt.tight_layout()
plt.show()

# ========== STEP 4: FIT ARIMA MODEL ==========
# Log-transform to stabilize variance
df['LogPassengers'] = np.log(df['Passengers'])

# ARIMA(1,1,1) with seasonal(1,1,1,12)
from statsmodels.tsa.statespace.sarimax import SARIMAX
model = SARIMAX(df['LogPassengers'], 
                order=(1, 1, 1), 
                seasonal_order=(1, 1, 1, 12))
fitted = model.fit(disp=False)
print(fitted.summary())

# ========== STEP 5: FORECAST ==========
forecast = fitted.get_forecast(steps=24)
forecast_mean = np.exp(forecast.predicted_mean)  # Back-transform
forecast_ci = np.exp(forecast.conf_int())

plt.figure(figsize=(12, 6))
plt.plot(df['Passengers'], label='Observed')
plt.plot(forecast_mean.index, forecast_mean, label='Forecast', color='red')
plt.fill_between(forecast_ci.index, 
                 forecast_ci.iloc[:, 0], 
                 forecast_ci.iloc[:, 1], 
                 color='pink', alpha=0.3)
plt.legend()
plt.title('SARIMA Forecast')
plt.show()
```

### R

```r
library(forecast)
library(tseries)
library(ggplot2)

# ========== LOAD DATA ==========
data("AirPassengers")
ts_data <- AirPassengers

# ========== STEP 1: VISUALIZE ==========
autoplot(ts_data) + 
  ggtitle("Monthly Airline Passengers") +
  ylab("Passengers (thousands)")

# Decomposition
decomposed <- decompose(ts_data, type = "multiplicative")
plot(decomposed)

# ========== STEP 2: TEST STATIONARITY ==========
# ADF Test
adf.test(ts_data)

# KPSS Test
kpss.test(ts_data)

# ========== STEP 3: ACF/PACF ==========
par(mfrow = c(1, 2))
acf(ts_data, main = "ACF")
pacf(ts_data, main = "PACF")

# ACF/PACF of differenced series
diff_data <- diff(log(ts_data))
par(mfrow = c(1, 2))
acf(diff_data, main = "ACF (Differenced)")
pacf(diff_data, main = "PACF (Differenced)")

# ========== STEP 4: FIT ARIMA MODEL ==========
# Auto ARIMA finds best model automatically
fit <- auto.arima(ts_data, seasonal = TRUE, 
                  stepwise = FALSE, approximation = FALSE)
summary(fit)

# Check residuals (should be white noise)
checkresiduals(fit)

# ========== STEP 5: FORECAST ==========
fc <- forecast(fit, h = 24)
autoplot(fc) + 
  ggtitle("SARIMA Forecast: 2-Year Ahead") +
  ylab("Passengers")

# Prediction intervals
print(fc)
```

---

## Interpretation Guide

| Output | Example Value | Interpretation | Edge Case/Warning |
|--------|---------------|----------------|-------------------|
| **ADF p-value** | 0.02 | p < 0.05 → Reject $H_0$ → Series is stationary. | If borderline, difference once and retest. |
| **ADF p-value** | 0.35 | p > 0.05 → Fail to reject → Series is non-stationary. | Apply differencing: $Y'_t = Y_t - Y_{t-1}$. |
| **ACF lags 1-12 significant** | | Strong autocorrelation. Seasonality at lag 12. | If lag 12 is highest, use seasonal model. |
| **ARIMA(1,1,1)** | | 1 AR term, 1 difference, 1 MA term. | More differences (d=2) if still non-stationary after d=1. |
| **AIC = 450** | | Model fit metric. Lower = better (for same data). | Compare across candidate models only. |
| **Ljung-Box p > 0.05** | | Residuals are white noise (good fit). | If p < 0.05, model misses structure. Add AR/MA terms. |
| **Forecast CI widening** | | Uncertainty grows with horizon. | Long-range forecasts are increasingly uncertain. |

---

## Common Pitfalls

> [!warning] Real-World Traps
>
> **1. Modeling Non-Stationary Data Directly**
> - *Problem:* Fitting AR model to trending data without differencing.
> - *Result:* Model captures trend, but forecasts diverge wildly.
> - *Solution:* Always test stationarity. Difference if needed.
>
> **2. Confusing Seasonality Period**
> - *Problem:* Monthly data with annual seasonality → period = 12 (not 4).
> - *Result:* Using wrong period produces nonsensical decomposition.
> - *Solution:* Match period to data frequency (monthly=12, quarterly=4, daily=7 or 365).
>
> **3. Using Future Data to Predict Past (Data Leakage)**
> - *Problem:* Training on 2010-2020, but feature includes "average of 2015-2025."
> - *Result:* Artificially good performance, failure in production.
> - *Solution:* All features must be computable at prediction time.
>
> **4. Ignoring Structural Breaks**
> - *Problem:* COVID-19 changed all patterns, but model trained on pre-COVID data.
> - *Result:* Model predicts "business as usual" during disruption.
> - *Solution:* Detect change points. Retrain on post-break data or use regime models.

---

## Worked Numerical Example

> [!example] Seasonal Decomposition of Monthly Sales
> **Scenario:** Monthly widget sales data (2 years, 24 observations).
>
> **Raw Data:**
> ```
> Month:  J   F   M   A   M   J   J   A   S   O   N   D  (Year 1)
> Sales: 100 110 130 140 160 180 190 185 150 130 115 120
>        105 115 135 145 165 185 195 190 155 135 120 125 (Year 2)
> ```
>
> **Step 1: Identify Pattern**
> - Clear upward trend in summer months
> - Peak in July-August, trough in January-February
> - Pattern repeats → **Additive seasonality**
>
> **Step 2: Calculate Trend (12-Month Moving Average)**
> ```
> MA(July Y1) = (100+110+...+120+105+...+185)/12 = 145.8
> MA(Aug Y1)  = (110+...+105+115+135)/12 = 147.5
> ...
> ```
>
> **Step 3: Detrend (Sales - Trend)**
> ```
> July Y1: 190 - 145.8 = +44.2
> Aug Y1:  185 - 147.5 = +37.5
> Jan Y2:  105 - 151.7 = -46.7
> ...
> ```
>
> **Step 4: Estimate Seasonal Factors**
> ```
> Average deviation for each month:
> July: (44.2 + 46.0)/2 = +45.1
> August: (37.5 + 38.2)/2 = +37.9
> January: (-45.0 + -46.7)/2 = -45.9
> ...
> ```
>
> **Step 5: Final Decomposition**
> ```
> July Y2 Observed: 195
>   = Trend (155.0)
>   + Seasonal (+45.1)
>   + Residual (-5.1)
> ```
>
> **Interpretation:**
> - Trend shows ~5 units/year growth
> - Summer seasonal boost: +45 units
> - Small residual: Model fits well

---

## Common Models

| Model | Use Case | Key Feature |
|-------|----------|-------------|
| [[stats/05_Time_Series/ARIMA Models\|ARIMA Models]] | General forecasting | Handles trend + autocorrelation |
| [[SARIMA\|SARIMA]] | Seasonal data | ARIMA + seasonal components |
| [[Exponential Smoothing\|Exponential Smoothing]] | Simple forecasting | Weighted average of past |
| [[Prophet\|Prophet]] | Business forecasting | Handles holidays, changepoints |
| [[stats/05_Time_Series/GARCH Models\|GARCH Models]] | Volatility forecasting | Time-varying variance |
| [[stats/05_Time_Series/Vector Autoregression (VAR)\|Vector Autoregression (VAR)]] | Multivariate series | Multiple interrelated series |

---

## Related Concepts

**Prerequisites:**
- [[stats/05_Time_Series/Auto-Correlation (ACF & PACF)\|Auto-Correlation (ACF & PACF)]] — Fundamental diagnostic
- [[stats/05_Time_Series/Stationarity (ADF & KPSS)\|Stationarity (ADF & KPSS)]] — Required assumption

**Core Models:**
- [[stats/05_Time_Series/ARIMA Models\|ARIMA Models]] — Standard forecasting
- [[Exponential Smoothing\|Exponential Smoothing]] — Simpler alternative
- [[stats/05_Time_Series/GARCH Models\|GARCH Models]] — Volatility modeling

**Advanced:**
- [[stats/04_Machine_Learning/Cross-Validation\|Cross-Validation]] — Time series split (no shuffling!)
- [[Anomaly Detection\|Anomaly Detection]] — Finding unusual points
