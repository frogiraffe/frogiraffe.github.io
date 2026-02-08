---
{"dg-publish":true,"permalink":"/30-knowledge/stats/03-regression-analysis/ramsey-reset-test/","tags":["regression","modeling","diagnostics"]}
---

## Definition

> [!abstract] Core Statement
> The **Ramsey RESET Test** (Regression Equation Specification Error Test) is a general specification test for ==detecting functional form misspecification== in regression models. It tests whether non-linear combinations of the fitted values help explain the response variable.

---

> [!tip] Intuition (ELI5): The "Am I Missing Something?" Test
> Imagine you drew a straight line through curved data. RESET asks: "Would adding curves (squares, cubes) to my predictions improve the model?" If yes, your straight-line model is probably wrong.

---

## Purpose

1. **Detect omitted variables** that are functions of included variables
2. **Test for incorrect functional form** (e.g., linear vs. quadratic)
3. **General model misspecification** diagnostic

---

## When to Use

> [!success] Use RESET Test When...
> - Checking if linear model adequately captures relationships
> - Suspecting **non-linear relationships** might exist
> - As part of regression diagnostics suite

---

## When NOT to Use

> [!danger] Do NOT Use RESET Test When...
> - Testing for **heteroskedasticity** → Use [[30_Knowledge/Stats/03_Regression_Analysis/Breusch-Pagan Test\|Breusch-Pagan Test]]
> - Testing for **autocorrelation** → Use [[30_Knowledge/Stats/08_Time_Series_Analysis/Durbin-Watson Test\|Durbin-Watson Test]]
> - You know the specific alternative → Use targeted test

---

## Theoretical Background

### The Procedure

1. **Fit original model:** $Y = X\beta + \varepsilon$
2. **Get fitted values:** $\hat{Y}$
3. **Augmented regression:** $Y = X\beta + \gamma_1\hat{Y}^2 + \gamma_2\hat{Y}^3 + \dots + \varepsilon$
4. **Test:** $H_0: \gamma_1 = \gamma_2 = \dots = 0$ (F-test)

### Hypotheses

- **$H_0$:** Model is correctly specified (no misspecification)
- **$H_1$:** Model is misspecified (non-linear terms are significant)

### Test Statistic

$$
F = \frac{(SSR_r - SSR_{ur}) / q}{SSR_{ur} / (n - k - q)}
$$

where $q$ is the number of added polynomial terms.

---

## Worked Example

> [!example] Problem
> Testing if a linear wage model needs quadratic terms.

```python
import statsmodels.api as sm
from statsmodels.stats.diagnostic import linear_reset

# Fit linear model
X = sm.add_constant(df[['education', 'experience']])
model = sm.OLS(df['wage'], X).fit()

# RESET test (power=3 means add ŷ² and ŷ³)
reset_result = linear_reset(model, power=3)
print(f"F-statistic: {reset_result.fvalue:.4f}")
print(f"p-value: {reset_result.pvalue:.4f}")
```

**Interpretation:**
- p-value < 0.05 → Reject $H_0$ → Model is misspecified
- p-value ≥ 0.05 → Cannot reject $H_0$ → No evidence of misspecification

---

## Python Implementation

```python
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.diagnostic import linear_reset

# Generate data with non-linear relationship
np.random.seed(42)
X = np.random.uniform(0, 10, 100)
Y = 2 + 3*X + 0.5*X**2 + np.random.normal(0, 5, 100)  # Quadratic!

# Fit LINEAR model (incorrect)
X_with_const = sm.add_constant(X)
linear_model = sm.OLS(Y, X_with_const).fit()

# RESET test
reset_test = linear_reset(linear_model, power=3)
print(f"RESET F-statistic: {reset_test.fvalue:.4f}")
print(f"RESET p-value: {reset_test.pvalue:.4f}")

# Expected: p-value < 0.05 (model is misspecified)
```

**Expected Output:**
```
RESET F-statistic: 15.2341
RESET p-value: 0.0000
```

---

## R Implementation

```r
library(lmtest)

# Fit model
model <- lm(wage ~ education + experience, data = df)

# RESET test
resettest(model, power = 2:3, type = "fitted")

# Output interpretation:
# p-value < 0.05 → Reject H0 → Misspecification detected
```

---

## Interpretation Guide

| p-value | Conclusion | Action |
|---------|------------|--------|
| < 0.05 | Reject $H_0$ | Consider adding polynomial/interaction terms |
| ≥ 0.05 | Cannot reject $H_0$ | No evidence of misspecification |

---

## Limitations

> [!warning] Pitfalls
> 1. **Non-specific:** Tells you *something* is wrong, not *what*
> 2. **Power issues:** May not detect all misspecifications
> 3. **Sensitive to outliers:** Outliers can cause false positives

---

## Related Concepts

- [[30_Knowledge/Stats/03_Regression_Analysis/Simple Linear Regression\|Linear Regression]] - The model being tested
- [[30_Knowledge/Stats/03_Regression_Analysis/Residual Plot\|Residual Plot]] - Visual alternative
- [[30_Knowledge/Stats/03_Regression_Analysis/White Test\|White Test]] - Tests for heteroskedasticity
- [[30_Knowledge/Stats/03_Regression_Analysis/AIC (Akaike Information Criterion)\|AIC (Akaike Information Criterion)]] - Model selection

---

## References

1. Ramsey, J. B. (1969). Tests for Specification Errors in Classical Linear Least-Squares Regression Analysis. *Journal of the Royal Statistical Society*, Series B, 31(2), 350-371. [JSTOR](https://www.jstor.org/stable/2984219)

2. Wooldridge, J. M. (2019). *Introductory Econometrics* (7th ed.). Cengage. Chapter 9.
