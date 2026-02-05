---
{"dg-publish":true,"permalink":"/stats/08-time-series-analysis/granger-causality/","tags":["time-series","causal-inference","econometrics"]}
---


## Definition

> [!abstract] Core Statement
> **Granger Causality** tests whether past values of one time series ($X$) help predict another time series ($Y$) beyond what $Y$'s own past values can predict. It measures ==predictive causality==, not true causality.

---

> [!tip] Intuition (ELI5): The Weather Report
> Imagine you want to predict tomorrow's umbrella sales.
> - If knowing today's *weather forecast* helps predict sales **better** than just knowing yesterday's sales, then weather "Granger-causes" umbrella sales.
> - It doesn't mean weather *actually causes* sales — maybe both are caused by the season. But weather is a useful predictor.

---

## Purpose

1. **Forecasting:** Identify useful predictors in time series
2. **Lead-Lag Relationships:** Determine which series "leads" another
3. **Policy Analysis:** Check if policy changes predict economic outcomes
4. **Finance:** Test if one asset's returns predict another's

---

## When to Use

> [!success] Use Granger Causality When...
> - You have **two or more stationary time series**
> - You want to test **predictive relationships** (not true causation)
> - Data is **temporal** with clear ordering
> - You're exploring **lead-lag dynamics**

> [!failure] Granger Causality Does NOT Imply...
> - **True Causation**: Correlation across time ≠ causation
> - **Direction**: X Granger-causes Y doesn't mean X *causes* Y
> - **Absence of confounders**: Both may be caused by a third variable

---

## Theoretical Background

### The Test

Test whether including lags of $X$ significantly improves prediction of $Y$:

**Restricted Model:**
$$Y_t = \alpha + \sum_{i=1}^{p} \beta_i Y_{t-i} + \epsilon_t$$

**Unrestricted Model:**
$$Y_t = \alpha + \sum_{i=1}^{p} \beta_i Y_{t-i} + \sum_{j=1}^{q} \gamma_j X_{t-j} + \epsilon_t$$

**Null Hypothesis:** $\gamma_1 = \gamma_2 = \dots = \gamma_q = 0$ (X does not Granger-cause Y)

Use **F-test** or **Chi-square test** to compare models.

### Requirements

1. **Stationarity:** Both series must be stationary (use [[stats/08_Time_Series_Analysis/Stationarity (ADF & KPSS)\|ADF test]])
2. **Lag Selection:** Use AIC/BIC to choose optimal lag length
3. **No Contemporaneous Effects:** Tests only lagged effects

---

## Python Implementation

```python
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
import matplotlib.pyplot as plt

# ========== GENERATE DATA ==========
np.random.seed(42)
n = 200

# X leads Y by 2 periods
x = np.cumsum(np.random.randn(n))
y = np.zeros(n)
for t in range(2, n):
    y[t] = 0.5 * y[t-1] + 0.3 * x[t-2] + np.random.randn()

df = pd.DataFrame({'X': x, 'Y': y})

# ========== CHECK STATIONARITY ==========
def check_stationarity(series, name):
    result = adfuller(series, autolag='AIC')
    print(f'{name}: ADF Statistic = {result[0]:.4f}, p-value = {result[1]:.4f}')
    if result[1] > 0.05:
        print(f'  ⚠️ {name} is NOT stationary. Consider differencing.')
    return result[1] < 0.05

# Difference to make stationary
df['X_diff'] = df['X'].diff().dropna()
df['Y_diff'] = df['Y'].diff().dropna()
df = df.dropna()

check_stationarity(df['X_diff'], 'X_diff')
check_stationarity(df['Y_diff'], 'Y_diff')

# ========== GRANGER CAUSALITY TEST ==========
print("\n" + "="*50)
print("Does X Granger-cause Y?")
print("="*50)
gc_results = grangercausalitytests(df[['Y_diff', 'X_diff']], maxlag=4, verbose=True)

# ========== INTERPRETATION ==========
# If p-value < 0.05 for any test (ssr_ftest, ssr_chi2test), 
# reject null: X DOES Granger-cause Y

# ========== BIDIRECTIONAL TEST ==========
print("\n" + "="*50)
print("Does Y Granger-cause X? (Reverse)")
print("="*50)
gc_reverse = grangercausalitytests(df[['X_diff', 'Y_diff']], maxlag=4, verbose=True)
```

---

## R Implementation

```r
library(lmtest)
library(vars)
library(tseries)

# ========== GENERATE DATA ==========
set.seed(42)
n <- 200
x <- cumsum(rnorm(n))
y <- rep(0, n)
for (t in 3:n) {
  y[t] <- 0.5 * y[t-1] + 0.3 * x[t-2] + rnorm(1)
}

df <- data.frame(X = x, Y = y)

# ========== CHECK STATIONARITY ==========
adf.test(diff(df$X))
adf.test(diff(df$Y))

# Difference the data
df$X_diff <- c(NA, diff(df$X))
df$Y_diff <- c(NA, diff(df$Y))
df <- na.omit(df)

# ========== GRANGER CAUSALITY (lmtest) ==========
# Does X Granger-cause Y?
grangertest(Y_diff ~ X_diff, order = 4, data = df)

# Does Y Granger-cause X? (Reverse)
grangertest(X_diff ~ Y_diff, order = 4, data = df)

# ========== USING VAR PACKAGE ==========
library(vars)
var_data <- df[, c("Y_diff", "X_diff")]
var_model <- VAR(var_data, p = 4, type = "const")

# Granger causality
causality(var_model, cause = "X_diff")
causality(var_model, cause = "Y_diff")
```

---

## Interpretation Guide

| Result | Meaning |
|--------|---------|
| **p < 0.05 for X→Y** | X Granger-causes Y (X helps predict Y) |
| **p > 0.05 for X→Y** | No evidence X Granger-causes Y |
| **Both directions significant** | Bidirectional Granger causality (feedback) |
| **Neither significant** | No predictive relationship detected |

---

## Common Pitfalls

> [!warning] Real-World Traps
>
> **1. Non-Stationary Data**
> - *Problem:* Granger test on non-stationary series gives spurious results
> - *Solution:* Always test stationarity first; difference if needed
>
> **2. Confusing with True Causality**
> - *Problem:* "X Granger-causes Y" interpreted as "X causes Y"
> - *Reality:* Could be a common cause (confounder) or coincidence
> - *Solution:* Use proper language: "X has predictive power for Y"
>
> **3. Wrong Lag Selection**
> - *Problem:* Too few lags miss effects; too many reduce power
> - *Solution:* Use information criteria (AIC/BIC) or test multiple lags
>
> **4. Simultaneous/Instantaneous Effects**
> - *Problem:* Granger test only captures lagged effects
> - *Solution:* If contemporaneous effects matter, use Structural VAR

---

## Worked Example

> [!example] Does Advertising Granger-Cause Sales?
> 
> **Data:** Monthly advertising spend and sales for 5 years
> 
> **Steps:**
> 1. Test stationarity → Both need differencing
> 2. Select lag = 3 (via AIC)
> 3. Granger test: Ad → Sales p = 0.002 ✓
> 4. Reverse test: Sales → Ad p = 0.45 ✗
> 
> **Conclusion:** Past advertising helps predict future sales (but not vice versa). This suggests a lead-lag relationship consistent with ad effectiveness.

---

## Related Concepts

**Prerequisites:**
- [[stats/08_Time_Series_Analysis/Stationarity (ADF & KPSS)\|Stationarity (ADF & KPSS)]] — Required assumption
- [[stats/08_Time_Series_Analysis/Auto-Correlation (ACF & PACF)\|Auto-Correlation (ACF & PACF)]] — Understanding temporal structure
- [[stats/08_Time_Series_Analysis/Vector Autoregression (VAR)\|Vector Autoregression (VAR)]] — Multivariate framework

**Extensions:**
- **Toda-Yamamoto Procedure** — Works with non-stationary data
- **Spectral Granger Causality** — Frequency domain analysis

**Alternatives:**
- [[stats/07_Causal_Inference/Causal Inference\|Causal Inference]] — For true causal identification
- [[stats/07_Causal_Inference/Instrumental Variables (IV)\|Instrumental Variables (IV)]] — When you need causality, not just prediction

---

## References

- **Historical:** Granger, C. W. J. (1969). Investigating causal relations by econometric models and cross-spectral methods. *Econometrica*, 37(3), 424-438. [JSTOR](https://www.jstor.org/stable/1912791)
- **Book:** Hamilton, J. D. (1994). *Time Series Analysis*. Princeton University Press. (Chapter 11) [Publisher](https://press.princeton.edu/books/hardcover/9780691042893/time-series-analysis)
- **Book:** Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*. Springer. [Springer Link](https://link.springer.com/book/10.1007/978-3-540-27752-1)
- **Article:** Toda, H. Y., & Yamamoto, T. (1995). Statistical inference in vector autoregressions with possibly integrated processes. *Journal of Econometrics*, 66(1-2), 225-250. [ScienceDirect](https://doi.org/10.1016/0304-4076(94)01616-8)
