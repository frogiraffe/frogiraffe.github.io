---
{"dg-publish":true,"permalink":"/stats/05-time-series/granger-causality/","tags":["Time-Series","Causal-Inference"]}
---

## Overview

> [!abstract] Definition
> **Granger Causality** tests if past values of $X$ help predict $Y$. "Predictive Causality."

---

## 1. Python Implementation

```python
from statsmodels.tsa.stattools import grangercausalitytests
# Data: [Target, Predictor]
grangercausalitytests(df[['Y', 'X']], maxlag=4)
```

---

## 2. R Implementation

```r
library(lmtest)

# Granger Causality Test
# order: lag length
grangertest(Y ~ X, order = 4, data = df)

# Note: Formula is Y ~ X (Does X Granger-cause Y?)
# Significant p-value (< 0.05) means YES.
```

---

## 3. Related Concepts

- [[stats/05_Time_Series/Vector Autoregression (VAR)\|Vector Autoregression (VAR)]]
- [[stats/05_Time_Series/Time Series Analysis\|Time Series Analysis]]

---

## References

- **Historical:** Granger, C. W. J. (1969). Investigating causal relations by econometric models and cross-spectral methods. *Econometrica*. [JSTOR](https://www.jstor.org/stable/1912791)
- **Book:** Hamilton, J. D. (1994). *Time Series Analysis*. Princeton University Press. [Book Info](https://press.princeton.edu/books/hardcover/9780691042893/time-series-analysis)
- **Book:** Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*. Springer. [Springer Link](https://link.springer.com/book/10.1007/978-3-540-27752-1)
