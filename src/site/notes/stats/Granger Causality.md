---
{"dg-publish":true,"permalink":"/stats/granger-causality/","tags":["Statistics","Time-Series","Causal-Inference"]}
---


# Granger Causality

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

- [[stats/Vector Autoregression (VAR)\|Vector Autoregression (VAR)]]
- [[Time Series Analysis\|Time Series Analysis]]
