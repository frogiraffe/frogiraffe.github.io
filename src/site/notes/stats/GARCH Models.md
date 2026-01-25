---
{"dg-publish":true,"permalink":"/stats/garch-models/","tags":["Statistics","Time-Series","Finance","Volatility"]}
---


# GARCH Models

## Overview

> [!abstract] Definition
> **GARCH** (Generalized AutoRegressive Conditional Heteroskedasticity) models are used to estimate the **volatility** of a time series. While ARIMA models the conditional mean, GARCH models the **conditional variance** ($\sigma_t^2$).

---

## 1. The Volatility Problem

In financial time series (stock returns), we often observe **Volatility Clustering**: Large changes tend to be followed by large changes, and small by small.
- This violates the constant variance (homoscedasticity) assumption of standard regression.
- GARCH treats this heteroscedasticity not as a bug, but as a feature to be modeled.

---

## 2. The Model Structure

**GARCH(p, q):**

$$ \sigma_t^2 = \omega + \sum_{i=1}^q \alpha_i \epsilon_{t-i}^2 + \sum_{j=1}^p \beta_j \sigma_{t-j}^2 $$

Where:
- $\omega$: Constant baseline variance.
- $\epsilon_{t-i}^2$: Past squared residuals (ARCH term) — "News" or "Shock".
- $\sigma_{t-j}^2$: Past variance (GARCH term) — "Persistence" (Memory).

**GARCH(1,1):** The most common specification.
$$ \sigma_t^2 = \omega + \alpha_1 \epsilon_{t-1}^2 + \beta_1 \sigma_{t-1}^2 $$

---

## Worked Example: Risk Management

> [!example] Value at Risk (VaR)
> You manage a portfolio.
> - **Today:** Return $r_t = -2\%$.
> - **Current Volatility:** $\sigma_t^2 = 1.5$.
> - **Model:** GARCH(1,1) with $\omega=0.1, \alpha=0.2, \beta=0.7$.
> 
> **Question:** Forecast tomorrow's volatility ($\sigma_{t+1}^2$).

**Solution:**
$$ \sigma_{t+1}^2 = \omega + \alpha \epsilon_t^2 + \beta \sigma_t^2 $$
1.  **Identifying Terms:**
    -   $\epsilon_t^2$ is the squared shock from today. If we assume mean return is 0, $\epsilon_t \approx -2$. So $\epsilon_t^2 = 4$.
    -   $\sigma_t^2 = 1.5$.

2.  **Calculation:**
    $$ \sigma_{t+1}^2 = 0.1 + (0.2 \times 4) + (0.7 \times 1.5) $$
    $$ \sigma_{t+1}^2 = 0.1 + 0.8 + 1.05 = 1.95 $$

**Conclusion:**
Volatility is predicted to **increase** from 1.5 to 1.95. The market is getting riskier because of today's large drop (shock).

---

## Assumptions

- [ ] **Stationarity:** The underlying series (returns) must be stationary (no trend).
- [ ] **Mean Reversion:** Volatility is expected to revert to a long-run average ($\alpha + \beta < 1$).
- [ ] **Clustering:** Volatility is not constant but clusters in periods of calm and stress.

---

## Limitations

> [!warning] Pitfalls
> 1.  **Explosive Models:** If $\alpha + \beta \ge 1$, the variance explodes to infinity (Integrated GARCH). Real financial data usually sums to ~0.99.
> 2.  **Ignores Direction:** Standard GARCH assumes positive shocks and negative shocks affect volatility equally. In reality, market crash (bad news) spikes volatility *more* than a rally (good news). Use **EGARCH** or **GJR-GARCH** for asymmetric effects.
> 3.  **High Frequency:** Does not work well on low-frequency data (e.g., yearly). Best for daily/intraday.

---

## 4. Interpretation Guide

| Coefficient | Meaning |
|-------------|---------|
| **$\alpha$ (ARCH)** | Reaction to new "shocks". High $\alpha$ = spiky volatility. |
| **$\beta$ (GARCH)** | Persistence of old variance. High $\beta$ = volatility dies out slowly. |
| **$\omega > 0$** | Baseline variance. Essential for stability. |
| **$\alpha + \beta \approx 1$** | High persistence (common in finance). |

---

## 5. Python Implementation (arch package)


*Note: The standard library `statsmodels` has limited GARCH support. `arch` is the industry standard.*

```python
from arch import arch_model
import numpy as np

# Sample Returns
returns = np.random.normal(0, 1, 1000)

# Fit GARCH(1,1)
model = arch_model(returns, vol='Garch', p=1, q=1)
results = model.fit(disp='off')

print(results.summary())

# Forecasting Volatility
forecasts = results.forecast(horizon=5)
print(forecasts.variance[-1:])
```

---

## 6. Related Concepts

- [[stats/ARIMA Models\|ARIMA Models]] - Often combined (ARIMA-GARCH).
- [[Heteroscedasticity\|Heteroscedasticity]] - The phenomenon being modeled.
- [[stats/Stationarity (ADF & KPSS)\|Stationarity (ADF & KPSS)]]
