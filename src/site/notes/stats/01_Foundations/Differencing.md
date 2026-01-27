---
{"dg-publish":true,"permalink":"/stats/01-foundations/differencing/","tags":["Time-Series","Stationarity","ARIMA"]}
---


## Definition

> [!abstract] Core Statement
> **Differencing** transforms a time series by computing ==the change between consecutive observations== to achieve stationarity.

$$\Delta y_t = y_t - y_{t-1}$$

---

## Orders of Differencing

| Order | Formula | Purpose |
|-------|---------|---------|
| **1st** | $y_t - y_{t-1}$ | Remove trend |
| **2nd** | $\Delta^2 y_t = \Delta y_t - \Delta y_{t-1}$ | Remove quadratic trend |
| **Seasonal** | $y_t - y_{t-s}$ | Remove seasonal pattern |

---

## Python Implementation

```python
import pandas as pd
import numpy as np

# First difference
diff1 = df['y'].diff()

# Second difference
diff2 = df['y'].diff().diff()

# Seasonal difference (lag 12)
seasonal_diff = df['y'].diff(12)

# Using statsmodels
from statsmodels.tsa.stattools import adfuller
result = adfuller(diff1.dropna())
print(f"ADF p-value: {result[1]:.4f}")
```

---

## R Implementation

```r
# First difference
diff(y)

# Second difference
diff(y, differences = 2)

# Seasonal difference
diff(y, lag = 12)
```

---

## When to Difference

Use ADF test: If p > 0.05 → not stationary → difference.

---

## Related Concepts

- [[stats/05_Time_Series/Stationarity (ADF & KPSS)\|Stationarity]] - Goal of differencing
- [[stats/05_Time_Series/ARIMA Models\|ARIMA Models]] - I = integrated = differenced
- [[Unit Root Tests\|Unit Root Tests]] - ADF, KPSS tests

---

## References

- **Book:** Hamilton, J. D. (1994). *Time Series Analysis*. Princeton. [Princeton Link](https://press.princeton.edu/books/hardcover/9780691042893/time-series-analysis)
