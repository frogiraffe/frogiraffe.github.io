---
{"dg-publish":true,"permalink":"/30-knowledge/stats/03-regression-analysis/robust-regression/","tags":["regression","modeling"]}
---


## Definition

> [!abstract] Core Statement
> **Robust Regression** methods are ==resistant to outliers== by down-weighting or ignoring extreme observations. Common approaches include M-estimation (Huber), LAD (Least Absolute Deviations), and MM-estimation.

---

## Methods

| Method | Loss Function | Robustness |
|--------|---------------|------------|
| **OLS** | $\sum e_i^2$ | None |
| **LAD (Median)** | $\sum \|e_i\|$ | High |
| **Huber** | Quadratic + Linear | Moderate |
| **MM-estimation** | Bounded influence | Very high |

---

## Python Implementation

```python
from sklearn.linear_model import HuberRegressor
import statsmodels.api as sm

# Huber regression
huber = HuberRegressor(epsilon=1.35)
huber.fit(X, y)

# Robust Linear Model
rlm = sm.RLM(y, X, M=sm.robust.norms.HuberT()).fit()
print(rlm.summary())
```

---

## R Implementation

```r
library(MASS)

# Robust regression (MM-estimation)
rlm_model <- rlm(y ~ x, data = df, method = "MM")
summary(rlm_model)

# LAD (Quantile regression at median)
library(quantreg)
rq(y ~ x, data = df, tau = 0.5)
```

---

## When to Use

- Data contains **outliers** that shouldn't be removed
- OLS residuals show extreme influential points
- Need stable coefficients despite contamination

---

## Related Concepts

- [[30_Knowledge/Stats/03_Regression_Analysis/Quantile Regression\|Quantile Regression]] - LAD is median regression
- [[30_Knowledge/Stats/03_Regression_Analysis/Cook's Distance\|Cook's Distance]] - Identify influential points
- [[30_Knowledge/Stats/03_Regression_Analysis/Weighted Least Squares (WLS)\|Weighted Least Squares (WLS)]] - Known variance structure

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Relationship is highly non-linear
> - Severe multicollinearity exists

---

## References

- **Book:** Rousseeuw, P. J., & Leroy, A. M. (2003). *Robust Regression and Outlier Detection*. Wiley. [Wiley Link](https://www.wiley.com/en-us/Robust+Regression+and+Outlier+Detection-p-9780471196720)
