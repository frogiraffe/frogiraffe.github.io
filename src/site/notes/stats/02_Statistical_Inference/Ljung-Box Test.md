---
{"dg-publish":true,"permalink":"/stats/02-statistical-inference/ljung-box-test/","tags":["Time-Series","Hypothesis-Testing","Autocorrelation"]}
---


## Definition

> [!abstract] Core Statement
> The **Ljung-Box Test** tests whether ==any autocorrelations up to lag h are non-zero==. It's used to diagnose whether residuals from a time series model are white noise.

---

## Hypotheses

$$H_0: \rho_1 = \rho_2 = \dots = \rho_h = 0$$ (residuals are white noise)
$$H_1: \text{At least one } \rho_k \neq 0$$

---

## Test Statistic

$$Q = n(n+2) \sum_{k=1}^{h} \frac{\hat{\rho}_k^2}{n-k} \sim \chi^2_{h-p-q}$$

Where p, q = ARIMA orders (subtract from df).

---

## Python Implementation

```python
from statsmodels.stats.diagnostic import acorr_ljungbox

# Test residuals from ARIMA model
result = acorr_ljungbox(residuals, lags=[10, 20, 30], return_df=True)
print(result)
# p > 0.05 → residuals are white noise (good!)
```

---

## R Implementation

```r
# After fitting ARIMA
Box.test(residuals(model), lag = 20, type = "Ljung-Box")
```

---

## Interpretation

| p-value | Interpretation |
|---------|----------------|
| > 0.05 | Residuals are white noise ✓ |
| < 0.05 | Significant autocorrelation remains |

---

## Related Concepts

- [[stats/08_Time_Series_Analysis/Auto-Correlation (ACF & PACF)\|Auto-Correlation (ACF & PACF)]] - Visual check
- [[stats/08_Time_Series_Analysis/ARIMA Models\|ARIMA Models]] - Model diagnostics
- [[stats/08_Time_Series_Analysis/Durbin-Watson Test\|Durbin-Watson Test]] - Tests lag-1 only

---

## References

- **Article:** Ljung, G. M., & Box, G. E. (1978). On a measure of lack of fit in time series models. *Biometrika*, 65(2), 297-303. [JSTOR](https://www.jstor.org/stable/2335207)
