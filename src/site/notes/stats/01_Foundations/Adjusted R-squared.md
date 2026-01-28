---
{"dg-publish":true,"permalink":"/stats/01-foundations/adjusted-r-squared/","tags":["Regression","Model-Selection","Metrics"]}
---


## Definition

> [!abstract] Core Statement
> **Adjusted R-squared** modifies R² to ==penalize for the number of predictors==, preventing spurious improvement from adding useless variables.

$$\bar{R}^2 = 1 - \frac{(1-R^2)(n-1)}{n-p-1}$$

Where p = number of predictors.

---

## Key Differences

| Metric | Behavior |
|--------|----------|
| **R²** | Always increases with more predictors |
| **Adjusted R²** | Decreases if predictor doesn't help |

---

## Interpretation

| Value | Interpretation |
|-------|---------------|
| Higher than R² | Can't happen |
| Close to R² | Predictors are useful |
| Much lower than R² | Too many useless predictors |

---

## Python Implementation

```python
import statsmodels.api as sm

model = sm.OLS(y, X).fit()
print(f"R²: {model.rsquared:.4f}")
print(f"Adjusted R²: {model.rsquared_adj:.4f}")
```

---

## R Implementation

```r
model <- lm(y ~ x1 + x2, data = df)
summary(model)$adj.r.squared
```

---

## Related Concepts

- [[stats/01_Foundations/Adjusted R-squared\|R-Squared]] - Unadjusted version
- [[stats/03_Regression_Analysis/AIC (Akaike Information Criterion)\|AIC (Akaike Information Criterion)]] - Alternative criterion
- [[stats/01_Foundations/Model Selection\|Model Selection]] - Choosing predictors

---

## References

- **Book:** James, G., et al. (2013). *An Introduction to Statistical Learning*. Springer. [Book Website](https://www.statlearning.com/)
