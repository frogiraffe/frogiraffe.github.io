---
{"dg-publish":true,"permalink":"/stats/05-time-series/auto-correlation-acf-and-pacf/","tags":["Time-Series","Diagnostics","Visualization"]}
---

## Definition

> [!abstract] Core Statement
> **ACF (Auto-Correlation Function)** measures the correlation between a time series and its lagged values (e.g., $Y_t$ vs $Y_{t-1}$).
> **PACF (Partial Auto-Correlation Function)** measures the correlation between $Y_t$ and $Y_{t-k}$ *after removing* the effects of the intermediate lags ($Y_{t-1} \dots Y_{t-k+1}$).

These are the primary tools for identifying the order (p, q) of [[stats/05_Time_Series/ARIMA Models\|ARIMA Models]].

---

## Purpose

1.  **Identify Seasonality:** Spikes at regular intervals (e.g., every 12 lags for monthly data).
2.  **Determine Model Order:** Use the shape of ACF/PACF plots to choose $p$ (AR) and $q$ (MA).
3.  **Residual Checking:** Are the errors "White Noise"? (Ideally, ACF should be zero for all lags > 0).

---

## The Rules of Thumb (Box-Jenkins)

| Plot | AR Process ($p$) | MA Process ($q$) | ARMA ($p, q$) |
|------|------------------|------------------|---------------|
| **ACF** | Decays gradually (Geometric/Sinusoidal) | **Cuts off** after lag $q$ | Decays gradually |
| **PACF** | **Cuts off** after lag $p$ | Decays gradually | Decays gradually |

> [!tip] Mnemonic
> - **AR(p):** Look at **PACF**. Significant spike at $p$, then zero.
> - **MA(q):** Look at **ACF**. Significant spike at $q$, then zero.

---

## Worked Example: Identifying a Model

> [!example] Problem
> You plot ACF and PACF for a stationary series.
> 
> **Observation 1 (PACF):**
> -   Huge spike at Lag 1 ($r=0.8$).
> -   Spike at Lag 2 is small/insignificant.
> -   **Conclusion:** This suggests **AR(1)**.
> 
> **Observation 2 (ACF):**
> -   ACF starts high (0.8) and slowly decays (0.64, 0.51...).
> -   This confirms it is an AR process (gradual decay).
> 
> **Model Proposal:** ARIMA(1, 0, 0).

---

## Assumptions

- [ ] **Stationarity:** Both diagnostics assume mean and variance are constant. If ACF decays *very* slowly (linear), the data is non-stationary. Differencing ($d=1$) is required.

---

## Python Implementation

```python
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load Data
# data = pd.read_csv(...)['Sales']

# Plot
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(data, lags=20, ax=ax[0])
plot_pacf(data, lags=20, ax=ax[1])
plt.show()

# Interpretation:
# Blue shaded area is the 95% Confidence Interval. 
# Anything outside the blue zone is statistically significant.
```

---

## Common Pitfall

> [!warning] The "Intermediate" Trap
> Why do we need PACF?
> - If $Y_{t-1}$ causes $Y_t$, and $Y_{t-2}$ causes $Y_{t-1}$...
> - Then $Y_{t-2}$ will correlate with $Y_t$ purely because of the chain reaction.
> - **ACF** shows this "echo" (Lag 2 is correlated).
> - **PACF** removes the middleman ($Y_{t-1}$) and shows the *pure* correlation of Lag 2. (Result: Zero).
> - **Mistake:** Using ACF to set AR order usually leads to picking a $p$ that is way too high.

---

## Related Concepts

- [[stats/05_Time_Series/ARIMA Models\|ARIMA Models]] - The model built using these tools.
- [[stats/05_Time_Series/Stationarity (ADF & KPSS)\|Stationarity (ADF & KPSS)]] - Prerequisite.
- [[stats/02_Hypothesis_Testing/Ljung-Box Test\|Ljung-Box Test]] - Statistical test for "White Noise" (checking group of lags).
