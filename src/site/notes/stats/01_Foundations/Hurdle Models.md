---
{"dg-publish":true,"permalink":"/stats/01-foundations/hurdle-models/","tags":["Count-Data","GLM","Zero-Inflation"]}
---


## Definition

> [!abstract] Core Statement
> **Hurdle Models** are two-part models for count data with ==excess zeros==: Part 1 models zero vs. non-zero (binary), Part 2 models positive counts (truncated count model).

---

## Structure

$$P(Y = 0) = \pi$$
$$P(Y = k | Y > 0) = \frac{f(k)}{1 - f(0)} \quad \text{for } k = 1, 2, \dots$$

Where f is Poisson or Negative Binomial.

---

## vs Zero-Inflated Models

| Hurdle | Zero-Inflated |
|--------|---------------|
| All zeros from one process | Zeros from two processes |
| "Did event happen?" | "Is this a structural zero?" |

---

## Python Implementation

```python
# Using statsmodels (limited support)
# Manual two-part approach:
import statsmodels.api as sm
import numpy as np

# Part 1: Binary (zero vs non-zero)
y_binary = (y > 0).astype(int)
logit = sm.Logit(y_binary, X).fit()

# Part 2: Truncated Poisson on positive counts
y_positive = y[y > 0]
X_positive = X[y > 0]
# (Use truncated Poisson or NB)
```

---

## R Implementation

```r
library(pscl)

# Hurdle model
hurdle_model <- hurdle(y ~ x1 + x2 | z1 + z2, 
                       data = df,
                       dist = "negbin")
summary(hurdle_model)
```

---

## When to Use

- Data has excess zeros not explained by Poisson mean
- Zeros represent a different process (never customers vs. current non-buyers)

---

## Related Concepts

- [[Zero-Inflated Poisson (ZIP)\|Zero-Inflated Poisson (ZIP)]] - Alternative for excess zeros
- [[stats/03_Regression_Analysis/Negative Binomial Regression\|Negative Binomial Regression]] - Handles overdispersion
- [[stats/01_Foundations/Overdispersion\|Overdispersion]] - Often accompanies zero-inflation

---

## References

- **Historical:** Cragg, J. G. (1971). Some Statistical Models for Limited Dependent Variables. *Econometrica*, 39(5), 829-844. [JSTOR](https://www.jstor.org/stable/1912139)
- **Book:** Cameron, A. C., & Trivedi, P. K. (2013). *Regression Analysis of Count Data* (2nd ed.). Cambridge University Press. [Cambridge Link](https://www.cambridge.org/core/books/regression-analysis-of-count-data/40D5B7EA1E9804B760737C82D79FA1A9)
