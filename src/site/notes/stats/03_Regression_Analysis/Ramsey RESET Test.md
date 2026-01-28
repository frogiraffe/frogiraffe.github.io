---
{"dg-publish":true,"permalink":"/stats/03-regression-analysis/ramsey-reset-test/","tags":["Diagnostics","Regression"]}
---

## Overview

> [!abstract] Definition
> The **Ramsey Regression Equation Specification Error Test (RESET)** determines whether a linear regression model has been incorrectly specified. Specifically, it tests for omitted variables or incorrect functional forms (e.g., non-linear relationships missing).

---

## 1. Procedure

1. Estimate the OLS model: $Y = X\beta + u$.
2. Calculate the fitted values: $\hat{Y}$.
3. Create auxiliary regression: Add powers of fitted values ($\hat{Y}^2, \hat{Y}^3, \dots$) as new predictors to the original model.
   $$ Y = X\beta + \gamma_1 \hat{Y}^2 + \gamma_2 \hat{Y}^3 + v $$
4. Test Significance: Test if $\gamma_1 = \gamma_2 = 0$ using an F-test.

---

## 2. Interpretation

- **Null Hypothesis ($H_0$):** The model is correctly specified.
- **Alternative ($H_1$):** The model is misspecified.

**Significant Result:** Indicates that non-linear combinations of the predictors help explain the response, suggesting that terms (like squares or interactions) are missing from the original model.

---

## 3. Python Implementation Example

```python
import statsmodels.stats.diagnostic as dg

# Results: [F-Stat, p-value, df_num, df_denom]
reset = dg.linear_reset(model, power=2, test_type='fitted')
print(f"RESET p-value: {reset.pvalue:.4f}")
```

---

## 4. Related Concepts

- [[stats/03_Regression_Analysis/Multiple Linear Regression\|Multiple Linear Regression]] - Model being tested.
- [[stats/01_Foundations/Log Transformation\|Log Transformation]] - Potential fix for non-linearity.

---

## References

- **Historical:** Ramsey, J. B. (1969). Tests for specification errors in classical linear least squares regression analysis. *Journal of the Royal Statistical Society: Series B*, 31(2), 350-371. [DOI Link](https://doi.org/10.1111/j.2517-6161.1969.tb00796.x)
- **Book:** Wooldridge, J. M. (2015). *Introductory Econometrics: A Modern Approach* (6th ed.). Cengage Learning. [Cengage Link](https://www.cengage.com/c/introductory-econometrics-a-modern-approach-6e-wooldridge/9781305270107/)
- **Book:** Gujarati, D. N. (2009). *Basic Econometrics* (5th ed.). McGraw-Hill. [McGraw-Hill Link](https://www.mheducation.com/highered/product/basic-econometrics-gujarati-porter/M9780073375779.html)