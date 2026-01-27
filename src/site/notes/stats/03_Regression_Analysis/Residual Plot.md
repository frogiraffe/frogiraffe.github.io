---
{"dg-publish":true,"permalink":"/stats/03-regression-analysis/residual-plot/","tags":["Regression","Diagnostics","Visualization"]}
---


## Definition

> [!abstract] Core Statement
> A **Residual Plot** displays ==residuals versus fitted values or predictors== to assess regression assumptions: linearity, homoscedasticity, and independence.

![Residual Plot Visualization](https://upload.wikimedia.org/wikipedia/commons/e/e3/Residuals_for_Linear_Regression_Fit.png)

---

## Patterns to Look For

| Pattern | Diagnosis | Solution |
|---------|-----------|----------|
| Random scatter | ✓ Good fit | None needed |
| Curved pattern | Non-linearity | Add polynomial terms |
| Funnel shape | Heteroscedasticity | Transform Y or use WLS |
| Clusters | Missing predictor | Add variables |

---

## Python Implementation

```python
import statsmodels.api as sm
import matplotlib.pyplot as plt

model = sm.OLS(y, X).fit()

plt.scatter(model.fittedvalues, model.resid)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted')
plt.show()
```

---

## R Implementation

```r
model <- lm(y ~ x, data = df)
plot(model, which = 1)  # Residuals vs Fitted
```

---

## Related Concepts

- [[stats/03_Regression_Analysis/Residual Analysis\|Residual Analysis]] - Full diagnostics
- [[stats/03_Regression_Analysis/Heteroscedasticity\|Heteroscedasticity]] - Funnel pattern
- [[stats/03_Regression_Analysis/Breusch-Pagan Test\|Breusch-Pagan Test]] - Formal test

---

## References

- **Book:** Fox, J. (2015). *Applied Regression Analysis*. SAGE. [Sage Link](https://us.sagepub.com/en-us/nam/applied-regression-analysis-and-generalized-linear-models/book237240)
