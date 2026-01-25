---
{"dg-publish":true,"permalink":"/stats/03-regression-analysis/heteroscedasticity/","tags":["Regression","Assumptions","Diagnostics"]}
---


## Definition

> [!abstract] Overview
> **Heteroscedasticity** refers to the circumstance in which the variability of a variable is unequal across the range of values of a second variable that predicts it.
>
> In simple terms: **The "spread" of the errors (residuals) is not constant.**

- **Homoscedasticity:** Constant variance (The ideal state).
- **Heteroscedasticity:** "Cone shape" in residual plots.

---

## 1. Consequences

If present in Linear Regression:
1.  **Coefficients are Unbiased:** The line is still "correct" on average.
2.  **Standard Errors are Wrong:** Usually underestimated.
3.  **P-values are Wrong:** You might find a variable Significant when it is not (Type I Error).

---

## 2. Detection

1.  **Visual:** Plot Residuals vs Fitted Values. Look for a "Fan" or "Cone" shape.
2.  **Statistical Tests:**
    - [[stats/03_Regression_Analysis/Breusch-Pagan Test\|Breusch-Pagan Test]]
    - [[stats/03_Regression_Analysis/White Test\|White Test]]

---

## 3. Solutions

1.  **Log Transformation:** Compresses high values, often stabilizing variance.
2.  **Robust Standard Errors (HC0, HC1, HC3):** Adjusts the standard error calculation without changing the coefficients.
3.  **Weighted Least Squares (WLS):** Give less weight to points with high variance.

---

## 4. Python Implementation

```python
import statsmodels.api as sm
import statsmodels.stats.api as sms
from matplotlib import pyplot as plt

# Fit model
model = sm.OLS(y, X).fit()

# 1. Breusch-Pagan Test
test = sms.het_breuschpagan(model.resid, model.model.exog)
print(f"P-value: {test[1]:.4f}")
if test[1] < 0.05:
    print("Heteroscedasticity detected!")

# 2. Plotting
plt.scatter(model.fittedvalues, model.resid)
plt.axhline(0, color='red')
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.show()

# 3. Robust Standard Errors (Fix)
robust_model = model.get_robustcov_results(cov_type='HC3')
print(robust_model.summary())
```

---

## Related Concepts

- [[stats/03_Regression_Analysis/Simple Linear Regression\|Simple Linear Regression]]
- [[stats/01_Foundations/Log Transformation\|Log Transformation]]
- [[stats/03_Regression_Analysis/Robust Regression\|Robust Regression]]
