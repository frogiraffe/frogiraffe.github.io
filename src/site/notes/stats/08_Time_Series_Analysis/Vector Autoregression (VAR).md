---
{"dg-publish":true,"permalink":"/stats/08-time-series-analysis/vector-autoregression-var/","tags":["Time-Series","Multivariate"]}
---

## Overview

> [!abstract] Definition
> **VAR (Vector Autoregression)** is a stochastic process model used to capture the linear interdependencies among multiple time series. It generalizes the univariate autoregressive (AR) model by allowing for more than one evolving variable.

---

## 1. Logic of VAR

In a VAR model, each variable depends on:
1. Its own past values (lags).
2. The past values of **all other variables** in the system.

**Example (2 variables):**
$$ Y_{1,t} = c_1 + \phi_{11}Y_{1,t-1} + \phi_{12}Y_{2,t-1} + e_{1,t} $$
$$ Y_{2,t} = c_2 + \phi_{21}Y_{1,t-1} + \phi_{22}Y_{2,t-1} + e_{2,t} $$

This treats all variables as **endogenous** (simultaneously determined).

---

## 2. Prerequisites

1.  **Stationarity:** All variables in the system must be stationary. If they are $I(1)$ (integrated/trending) but cointegrated, use **VECM** (Vector Error Correction Model) instead.
2.  **Granger Causality:** Often used to verify if one series actually helps predict the other.
3.  **Lag Selection:** Use Information Criteria (AIC, BIC) to determine the optimal lag length.

---

## 3. Impulse Response Functions (IRF)

The coefficients in VAR are hard to interpret directly. Instead, we use **IRF**:
- **Question:** "If $Y_1$ experiences a sudden shock (increase), how does $Y_2$ react over time?"
- **Plot:** Shows the trajectory of the response over subsequent periods.

---

## 4. Python Implementation Example

```python
from statsmodels.tsa.api import VAR
import pandas as pd

# Data: DataFrame with multiple columns
# df = pd.read_csv(...) 

model = VAR(df)

# 1. Select Lag Order
lag_order = model.select_order(maxlags=15)
print(lag_order.summary())

# 2. Fit Model
results = model.fit(maxlags=2)
print(results.summary())

# 3. Impulse Response Analysis
irf = results.irf(10)
irf.plot(orth=False)

# 4. Granger Causality
print(results.test_causality('TargetVar', 'CauseVar'))
```

---

## 5. Related Concepts

- [[stats/08_Time_Series_Analysis/ARIMA Models\|ARIMA Models]] - Univariate case.
- [[stats/08_Time_Series_Analysis/Granger Causality\|Granger Causality]] - Causality test within VAR.
- [[stats/08_Time_Series_Analysis/Stationarity (ADF & KPSS)\|Stationarity (ADF & KPSS)]] - Requirement.

---

## References

- **Historical:** Sims, C. A. (1980). Macroeconomics and reality. *Econometrica*. [JSTOR](https://www.jstor.org/stable/1912017)
- **Book:** Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*. Springer. [Springer Link](https://link.springer.com/book/10.1007/978-3-540-27752-1)
- **Book:** Hamilton, J. D. (1994). *Time Series Analysis*. Princeton University Press. (Chapter 11) [Princeton University Press](https://press.princeton.edu/books/hardcover/9780691042893/time-series-analysis)
