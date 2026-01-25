---
{"dg-publish":true,"permalink":"/stats/white-test/","tags":["Statistics","Diagnostics","Regression","Heteroscedasticity"]}
---


# White Test

## Overview

> [!abstract] Definition
> The **White Test** is a general statistical test used to detect **heteroscedasticity** in a regression model. Unlike the Breusch-Pagan test, it does not assume any specific form of heteroscedasticity, making it more robust but also less powerful in specific cases.

---

## 1. Procedure

1. Obtain OLS residuals $u_i$.
2. Run auxiliary regression of squared residuals $u_i^2$ on:
   - All original predictors ($X_1 \dots X_p$).
   - The squared terms ($X_1^2 \dots X_p^2$).
   - The cross-products ($X_1 X_2, \dots$).
3. Compute $LM = n R^2$ from this auxiliary regression.

---

## 2. Comparison: White vs. Breusch-Pagan

| Feature | Breusch-Pagan | White Test |
|---------|---------------|------------|
| **Assumed Form** | Linear function of X | General (Quadratic/Interaction) |
| **Sensitivity** | Linear Heteroscedasticity | Non-linear Heteroscedasticity and Specification Error |
| **Degrees of Freedom** | Lower ($p$) | High ($p + p(p+1)/2$) |
| **Power** | Higher (if linear) | Lower (due to high df) |

---

## 3. Python Implementation Example

```python
from statsmodels.stats.diagnostic import het_white

lm, lm_p, f, f_p = het_white(model.resid, model.model.exog)
print(f"White Test p-value: {lm_p:.4f}")
```

---

## 4. Related Concepts

- [[stats/Breusch-Pagan Test\|Breusch-Pagan Test]] - Alternative heteroscedasticity test.
- [[stats/Multiple Linear Regression\|Multiple Linear Regression]] - Framework.
- [[stats/Weighted Least Squares (WLS)\|Weighted Least Squares (WLS)]] - Remediation.
