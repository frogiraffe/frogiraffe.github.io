---
{"dg-publish":true,"permalink":"/30-knowledge/stats/03-regression-analysis/aic-akaike-information-criterion/","tags":["regression","modeling"]}
---


## Definition

> [!abstract] Overview
> The **Akaike Information Criterion (AIC)** is an estimator of prediction error and thereby relative quality of statistical models for a given set of data. Given a collection of models for the data, AIC estimates the quality of each model, relative to each of the other models.

$$
AIC = 2k - 2\ln(\hat{L})
$$

Where:
*   $k$ is the number of estimated parameters in the model.
*   $\hat{L}$ is the maximum value of the likelihood function for the model.

## Purpose
*   **Model Selection:** Helps choose the best model by balancing **goodness of fit** (likelihood) against **model complexity** (number of parameters).
*   **Penalty for Complexity:** It penalizes overfitting by adding $2k$, discouraging models with too many parameters that don't significantly improve the fit.

## Interpretation
*   **Lower is Better:** The model with the lowest AIC is preferred.
*   **Relative Metric:** The absolute value of AIC doesn't matter; only the differences between AIC values of candidate models matter.

## Comparison with BIC
*   [[30_Knowledge/Stats/03_Regression_Analysis/BIC (Bayesian Information Criterion)\|BIC (Bayesian Information Criterion)]] imposes a stronger penalty for model complexity than AIC.
*   AIC tends to pick more complex models, while BIC tends to pick simpler models.

## Related Concepts
*   [[30_Knowledge/Stats/01_Foundations/Likelihood Function\|Likelihood Function]]
*   [[30_Knowledge/Stats/03_Regression_Analysis/BIC (Bayesian Information Criterion)\|BIC (Bayesian Information Criterion)]]
*   Stepwise Regression

## When to Use

> [!success] Use AIC (Akaike Information Criterion) When...
> - Refer to standard documentation
> - Refer to standard documentation

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Assumptions are violated
> - Alternative methods are more appropriate

---

## Python Implementation

```python
import numpy as np
import pandas as pd

# Example implementation of AIC (Akaike Information Criterion)
# See documentation for details

data = np.random.randn(100)
print(f"Mean: {np.mean(data):.3f}")
print(f"Std: {np.std(data):.3f}")
```

---

## R Implementation

```r
# AIC (Akaike Information Criterion) in R
set.seed(42)

# Example implementation
data <- rnorm(100)
summary(data)
```

---

## References

1. See related concepts for further reading
