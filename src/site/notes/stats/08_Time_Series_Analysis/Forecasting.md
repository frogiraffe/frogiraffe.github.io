---
{"dg-publish":true,"permalink":"/stats/08-time-series-analysis/forecasting/","tags":["time-series","forecasting","machine-learning"]}
---


## Definition

> [!abstract] Core Statement
> **Forecasting** is the process of making predictions about future values based on historical data. Modern approaches combine classical statistical methods ([[stats/08_Time_Series_Analysis/ARIMA Models\|ARIMA]], [[stats/08_Time_Series_Analysis/Smoothing\|Exponential Smoothing]]) with ==machine learning== (XGBoost, Neural Networks, Prophet).

---

> [!tip] Intuition (ELI5): The Crystal Ball
> Forecasting is like having a partially cloudy crystal ball. You can see patterns from the past (seasonality, trends), but the future is never certain. The goal is to make the best guess and know how uncertain that guess is.

---

## Purpose

1. **Business Planning:** Demand forecasting, inventory management
2. **Finance:** Stock prices, economic indicators
3. **Operations:** Resource allocation, staffing
4. **Science:** Climate prediction, epidemic modeling

---

## Forecasting Methods Overview

| Category | Methods | Best For |
|----------|---------|----------|
| **Classical Statistical** | ARIMA, ETS, SARIMA | Univariate, interpretable |
| **Decomposition** | STL, X-13 | Trend/seasonal extraction |
| **Machine Learning** | XGBoost, Random Forest | Complex patterns, tabular |
| **Deep Learning** | LSTM, Transformer | Long sequences, multivariate |
| **Hybrid** | Prophet, NeuralProphet | Business time series |

---

## Classical vs ML Approaches

| Aspect | Classical (ARIMA) | Machine Learning |
|--------|-------------------|------------------|
| **Interpretability** | High | Low to Medium |
| **Feature Engineering** | Automatic (lags) | Manual often needed |
| **Seasonality** | Built-in (SARIMA) | Needs encoding |
| **Uncertainty** | Native confidence intervals | Requires extra steps |
| **External Variables** | ARIMAX | Natural |
| **Computation** | Fast | Can be slow |

---

## Prophet (Facebook/Meta)

### Overview
Prophet is designed for **business time series** with:
- Strong seasonal effects (daily, weekly, yearly)
- Missing data and outliers
- Holiday effects
- Trend changes (changepoints)

### Model
$$y(t) = g(t) + s(t) + h(t) + \epsilon_t$$

Where:
- $g(t)$ = Trend (linear or logistic growth)
- $s(t)$ = Seasonality (Fourier series)
- $h(t)$ = Holiday effects
- $\epsilon_t$ = Error

### Python Implementation

```python
from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt

# ========== DATA PREPARATION ==========
# Prophet requires columns: 'ds' (date) and 'y' (value)
df = pd.DataFrame({
    'ds': pd.date_range(start='2020-01-01', periods=365*3, freq='D'),
    'y': np.cumsum(np.random.randn(365*3)) + 100 + 10*np.sin(2*np.pi*np.arange(365*3)/365)
})

# ========== FIT MODEL ==========
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    changepoint_prior_scale=0.05  # Flexibility of trend
)
model.fit(df)

# ========== FORECAST ==========
future = model.make_future_dataframe(periods=90)  # 90 days ahead
forecast = model.predict(future)

# ========== VISUALIZATION ==========
fig1 = model.plot(forecast)
plt.title('Prophet Forecast')
plt.show()

fig2 = model.plot_components(forecast)
plt.show()

# ========== CROSS-VALIDATION ==========
from prophet.diagnostics import cross_validation, performance_metrics

cv_results = cross_validation(model, initial='730 days', period='30 days', horizon='30 days')
metrics = performance_metrics(cv_results)
print(metrics[['horizon', 'mape', 'rmse']].head())
```

### R Implementation

```r
library(prophet)

# ========== DATA PREPARATION ==========
df <- data.frame(
  ds = seq(as.Date('2020-01-01'), by = 'day', length.out = 365*3),
  y = cumsum(rnorm(365*3)) + 100 + 10*sin(2*pi*seq_len(365*3)/365)
)

# ========== FIT MODEL ==========
model <- prophet(df, yearly.seasonality = TRUE, weekly.seasonality = TRUE)

# ========== FORECAST ==========
future <- make_future_dataframe(model, periods = 90)
forecast <- predict(model, future)

# ========== VISUALIZATION ==========
plot(model, forecast)
prophet_plot_components(model, forecast)
```

---

## XGBoost for Time Series

### Feature Engineering Required

```python
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit

def create_features(df, target_col, lags=[1, 7, 14, 30]):
    """Create lag features and date features"""
    df = df.copy()
    
    # Lag features
    for lag in lags:
        df[f'lag_{lag}'] = df[target_col].shift(lag)
    
    # Rolling statistics
    df['rolling_mean_7'] = df[target_col].shift(1).rolling(7).mean()
    df['rolling_std_7'] = df[target_col].shift(1).rolling(7).std()
    
    # Date features
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['dayofyear'] = df.index.dayofyear
    
    return df.dropna()

# ========== PREPARE DATA ==========
df = create_features(df.set_index('ds'), 'y')
feature_cols = [c for c in df.columns if c != 'y']

X = df[feature_cols]
y = df['y']

# ========== TIME SERIES SPLIT ==========
tscv = TimeSeriesSplit(n_splits=5)

for train_idx, val_idx in tscv.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

# ========== TRAIN MODEL ==========
model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    objective='reg:squarederror'
)
model.fit(X_train, y_train)

# ========== EVALUATE ==========
from sklearn.metrics import mean_absolute_error, mean_squared_error

predictions = model.predict(X_val)
print(f"MAE: {mean_absolute_error(y_val, predictions):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_val, predictions)):.2f}")

# ========== FEATURE IMPORTANCE ==========
import matplotlib.pyplot as plt
xgb.plot_importance(model, max_num_features=10)
plt.title('Feature Importance')
plt.show()
```

---

## LSTM for Forecasting

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# ========== PREPARE SEQUENCES ==========
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# Scale data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['y']])

seq_length = 30
X, y = create_sequences(scaled_data, seq_length)

# Split
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# ========== BUILD MODEL ==========
model = Sequential([
    LSTM(50, activation='relu', input_shape=(seq_length, 1), return_sequences=True),
    LSTM(50, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=0)

# ========== PREDICT ==========
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
```

---

## Evaluation Metrics

| Metric | Formula | Notes |
|--------|---------|-------|
| **MAE** | $\frac{1}{n}\sum\|y_i - \hat{y}_i\|$ | Scale-dependent |
| **RMSE** | $\sqrt{\frac{1}{n}\sum(y_i - \hat{y}_i)^2}$ | Penalizes big errors |
| **MAPE** | $\frac{100}{n}\sum\|\frac{y_i - \hat{y}_i}{y_i}\|$ | Scale-free, fails if $y_i = 0$ |
| **sMAPE** | $\frac{200}{n}\sum\frac{\|y_i - \hat{y}_i\|}{\|y_i\| + \|\hat{y}_i\|}$ | Symmetric MAPE |

---

## Common Pitfalls

> [!warning] Real-World Traps
>
> **1. Data Leakage**
> - *Problem:* Using future data to predict the past (train/test split wrong)
> - *Solution:* Always use time-aware splits (TimeSeriesSplit)
>
> **2. Ignoring Seasonality**
> - *Problem:* Model misses weekly/yearly patterns
> - *Solution:* Add seasonal features or use SARIMA/Prophet
>
> **3. Over-Differencing**
> - *Problem:* Differencing already stationary data → loses information
> - *Solution:* Test stationarity before differencing
>
> **4. Point Forecasts Only**
> - *Problem:* No uncertainty estimates
> - *Solution:* Use prediction intervals, ensemble methods

---

## Related Concepts

**Prerequisites:**
- [[stats/08_Time_Series_Analysis/ARIMA Models\|ARIMA Models]] — Classical statistical approach
- [[stats/08_Time_Series_Analysis/Stationarity (ADF & KPSS)\|Stationarity (ADF & KPSS)]] — Key assumption
- [[stats/08_Time_Series_Analysis/Auto-Correlation (ACF & PACF)\|Auto-Correlation (ACF & PACF)]] — Pattern identification

**Extensions:**
- [[stats/08_Time_Series_Analysis/GARCH Models\|GARCH Models]] — Volatility forecasting
- [[stats/08_Time_Series_Analysis/Vector Autoregression (VAR)\|Vector Autoregression (VAR)]] — Multivariate

---

## References

- **Prophet:** Taylor, S. J., & Letham, B. (2018). Forecasting at scale. *The American Statistician*, 72(1), 37-45. [DOI](https://doi.org/10.1080/00031305.2017.1380080)
- **Book:** Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). [Online Book](https://otexts.com/fpp3/)
- **Documentation:** [Facebook Prophet](https://facebook.github.io/prophet/)
