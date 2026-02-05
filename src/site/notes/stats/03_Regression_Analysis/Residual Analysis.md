---
{"dg-publish":true,"permalink":"/stats/03-regression-analysis/residual-analysis/","tags":["probability","regression","diagnostics","model-validation"]}
---


## Definition

> [!abstract] Core Statement
> **Residual Analysis** examines the ==differences between observed and predicted values== ($e_i = y_i - \hat{y}_i$) to validate regression assumptions and detect model inadequacies.

---

## Key Residual Types

| Type | Formula | Use |
|------|---------|-----|
| **Raw** | $e_i = y_i - \hat{y}_i$ | Basic error |
| **Standardized** | $e_i / \hat{\sigma}$ | Compare across scales |
| **Studentized** | $e_i / (s \sqrt{1-h_{ii}})$ | Account for leverage |

---

## What to Check

1. **Linearity:** Residuals vs fitted → random scatter
2. **Normality:** Q-Q plot → straight line
3. **Homoscedasticity:** Constant spread of residuals
4. **Independence:** No pattern over time/order

---

## Python Implementation

```python
import statsmodels.api as sm
import matplotlib.pyplot as plt

model = sm.OLS(y, X).fit()
residuals = model.resid

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Residuals vs Fitted
axes[0,0].scatter(model.fittedvalues, residuals)
axes[0,0].axhline(0, color='red')
axes[0,0].set_title('Residuals vs Fitted')

# Q-Q Plot
sm.qqplot(residuals, line='45', ax=axes[0,1])

# Scale-Location
axes[1,0].scatter(model.fittedvalues, abs(residuals)**0.5)
axes[1,0].set_title('Scale-Location')

# Residuals vs Leverage
sm.graphics.influence_plot(model, ax=axes[1,1])
plt.tight_layout()
```

---

## R Implementation

```r
model <- lm(y ~ x, data = df)

# Standard diagnostic plots
par(mfrow = c(2, 2))
plot(model)
```

---

## Related Concepts

- [[stats/03_Regression_Analysis/Residual Plot\|Residual Plot]] - Visualization
- [[stats/03_Regression_Analysis/Heteroscedasticity\|Heteroscedasticity]] - Non-constant variance
- [[stats/03_Regression_Analysis/Cook's Distance\|Cook's Distance]] - Influential points

---

## References

- **Book:** Faraway, J. J. (2014). *Linear Models with R*. CRC Press. [CRC Press Link](https://www.routledge.com/Linear-Models-with-R/Faraway/p/book/9781439887332)
