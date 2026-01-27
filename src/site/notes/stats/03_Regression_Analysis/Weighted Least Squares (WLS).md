---
{"dg-publish":true,"permalink":"/stats/03-regression-analysis/weighted-least-squares-wls/","tags":["Regression","Linear-Models","Heteroscedasticity"]}
---

## Definition

> [!abstract] Core Statement
> **Weighted Least Squares (WLS)** is a modification of OLS used when the assumption of ==constant variance (Homoscedasticity) is violated== (i.e., Heteroscedasticity is present). WLS assigns **weights** to observations inversely proportional to their error variance, giving less weight to noisy observations.

---

## Purpose

1.  **Correct for Heteroscedasticity:** Obtain unbiased and efficient coefficient estimates when error variance is not constant.
2.  **Improve Inference:** Produce valid standard errors and p-values.
3.  **Handle Known Variance Structures:** When you know or can model how variance changes with $X$.

---

## When to Use

> [!success] Use WLS When...
> - The [[stats/03_Regression_Analysis/Breusch-Pagan Test\|Breusch-Pagan Test]] or [[stats/03_Regression_Analysis/White Test\|White Test]] indicates **heteroscedasticity**.
> - You have a known or estimable relationship between variance and some variable.
> - Error spread is clearly a function of $X$ (e.g., variance increases with predicted value).

> [!failure] Alternatives
> - If heteroscedasticity is present but you don't want to model it explicitly, use **Robust Standard Errors** (`cov_type='HC3'` in Python, `vcovHC` in R).

---

## Theoretical Background

### The OLS Problem with Heteroscedasticity

In OLS, we minimize:
$$ \sum (y_i - \hat{y}_i)^2 $$
This treats all residuals equally. But if $Var(\varepsilon_i) = \sigma_i^2$ varies, high-variance observations contribute more noise, distorting estimates.

### The WLS Solution

WLS minimizes the **weighted** sum of squares:
$$ \sum w_i (y_i - \hat{y}_i)^2 $$
where $w_i = \frac{1}{\sigma_i^2}$ (inverse of variance).

**Effect:** Observations with high variance get low weight; observations with low variance get high weight.

### Choosing Weights

If the functional form of heteroscedasticity is known (e.g., $Var(\varepsilon) \propto X^2$), then $w = 1/X^2$.
If unknown, a common approach:
1.  Fit OLS.
2.  Regress $\log(\text{residuals}^2)$ on $X$.
3.  Use predicted values to construct weights.

---

## Assumptions

- [ ] **Correct Weight Specification:** Weights must accurately reflect the inverse of error variance. Misspecified weights can make things worse.
- [ ] All other OLS assumptions (Linearity, Independence, Normality of residuals, No multicollinearity).

---

## Limitations

> [!warning] Pitfalls
> 1.  **Weight Misspecification:** If you choose the wrong weights, WLS can be worse than OLS.
> 2.  **Complexity:** Requires modeling the variance function, which may not be straightforward.
> 3.  **Simpler Alternative Exists:** Often, **Robust Standard Errors** are easier to implement and sufficient for inference.

---

## Python Implementation

```python
import statsmodels.api as sm
import numpy as np

# 1. Fit OLS first
X_ols = sm.add_constant(X)
model_ols = sm.OLS(y, X_ols).fit()

# 2. Estimate Weights (Assuming Var ~ fitted values)
# Use absolute residuals as proxy for variance
fitted = model_ols.fittedvalues
residuals_abs = np.abs(model_ols.resid)

# Model: |residual| ~ fitted to estimate variance function
var_model = sm.OLS(residuals_abs, sm.add_constant(fitted)).fit()
estimated_variance = var_model.fittedvalues ** 2
weights = 1 / estimated_variance

# 3. Fit WLS
model_wls = sm.WLS(y, X_ols, weights=weights).fit()
print(model_wls.summary())

# Alternative: Just use Robust Standard Errors
model_robust = sm.OLS(y, X_ols).fit(cov_type='HC3')
print(model_robust.summary())
```

---

## R Implementation

```r
# 1. Fit OLS
model_ols <- lm(Y ~ X, data = df)

# 2. Estimate Weights (Example: Variance proportional to X)
# Common approach: Use fitted values or known structure
weights <- 1 / (df$X^2)  # If Var ~ X^2

# 3. Fit WLS
model_wls <- lm(Y ~ X, data = df, weights = weights)
summary(model_wls)

# Alternative: Robust Standard Errors
library(sandwich)
library(lmtest)
coeftest(model_ols, vcov = vcovHC(model_ols, type = "HC3"))
```

---

## Worked Numerical Example

> [!example] Income Prediction with Heteroscedasticity
> **Scenario:** Predicting Income from Years_of_Education
> 
> **Problem:** Higher education → higher variance in income (doctors, lawyers vs teachers)
> 
> **OLS Results:**
> - β_Education = $3,500, SE = 800, p = 0.002
> - Breusch-Pagan test: χ² = 18.5, p = 0.001 (Heteroscedasticity detected!)
> - Residual plot shows "fan shape" (variance increases with X)
> 
> **WLS (weights = 1/σ²_i):**
> - β_Education = $4,200, SE = 650, p < 0.001
> - Residual plot: no more fan shape
> 
> **Interpretation:**
> - OLS underestimated the effect ($3,500 vs $4,200)
> - WLS gives tighter SE (650 vs 800) = more precise
> - WLS gives more weight to observations with stable variance

---

## Interpretation Guide

| Output | Interpretation | Edge Case Notes |
|--------|----------------|------------------|
| Breusch-Pagan p < 0.05 | Heteroscedasticity detected. WLS justified. | If p = 0.06, borderline. Check residual plot visually. |
| WLS SE < OLS SE | WLS more efficient (tighter CIs). | Expected outcome when heteroscedasticity present. |
| WLS SE > OLS SE | Weights may be incorrect or unnecessary. | Recheck weight specification. May not need WLS. |
| Sign flip (WLS vs OLS) | Severe heteroscedasticity biased OLS. | Investigate: Outliers may be driving OLS estimate. |
| β_WLS ≈ β_OLS but SE differs | Heteroscedasticity affects precision, not bias. | WLS still preferable for valid inference. |

---

## Common Pitfall Example

> [!warning] Incorrect Weight Specification
> **Bad Practice:** Using arbitrary weights without justification
> 
> **Example:**
> - Analyst suspects heteroscedasticity
> - Arbitrarily decides: weight_i = 1/X_i
> - Result: Biased estimates!
> 
> **Correct Approach:**
> 1. Diagnose heteroscedasticity (Breusch-Pagan, White test, residual plot)
> 2. Model the variance: regress |residuals| on X
> 3. Use fitted values: weight_i = 1/(fitted_variance_i)
> 4. Or use robust standard errors (easier alternative)
> 
> **When in doubt:** Use `statsmodels` robust SE (`cov_type='HC3'`) instead of manual WLS

---

## Related Concepts

- [[stats/03_Regression_Analysis/Breusch-Pagan Test\|Breusch-Pagan Test]] - Diagnoses heteroscedasticity.
- [[stats/03_Regression_Analysis/White Test\|White Test]] - General heteroscedasticity test.
- [[stats/03_Regression_Analysis/Simple Linear Regression\|Simple Linear Regression]] - The unweighted baseline.
- [[stats/01_Foundations/Robust Standard Errors\|Robust Standard Errors]] - Simpler alternative for inference.

---

## References

- **Book:** Draper, N. R., & Smith, H. (1998). *Applied Regression Analysis* (3rd ed.). Wiley. [Wiley Link](https://www.wiley.com/en-us/Applied+Regression+Analysis%2C+3rd+Edition-p-9780471170822)
- **Book:** Montgomery, D. C., Peck, E. A., & Vining, G. G. (2012). *Introduction to Linear Regression Analysis* (5th ed.). Wiley. [Wiley Link](https://www.wiley.com/en-us/Introduction+to+Linear+Regression+Analysis%2C+5th+Edition-p-9780470542811)
- **Article:** Carroll, R. J., & Ruppert, D. (1988). *Transformation and Weighting in Regression*. Chapman and Hall. [DOI Link](https://doi.org/10.1007/978-1-4899-3114-6)
