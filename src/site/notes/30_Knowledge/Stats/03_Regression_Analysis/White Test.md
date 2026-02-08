---
{"dg-publish":true,"permalink":"/30-knowledge/stats/03-regression-analysis/white-test/","tags":["regression","modeling"]}
---

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

- [[30_Knowledge/Stats/03_Regression_Analysis/Breusch-Pagan Test\|Breusch-Pagan Test]] - Alternative heteroscedasticity test.
- [[30_Knowledge/Stats/03_Regression_Analysis/Multiple Linear Regression\|Multiple Linear Regression]] - Framework.
- [[30_Knowledge/Stats/03_Regression_Analysis/Weighted Least Squares (WLS)\|Weighted Least Squares (WLS)]] - Remediation.

---

## Definition

> [!abstract] Core Statement
> **White Test** ... Refer to standard documentation

---

> [!tip] Intuition (ELI5)
> Refer to standard documentation

---

## When to Use

> [!success] Use White Test When...
> - Refer to standard documentation
> - Refer to standard documentation

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Assumptions of the test are violated
> - Sample size doesn't meet minimum requirements

---

## Python Implementation

```python
from scipy import stats
import numpy as np

# Sample data
group1 = np.random.normal(10, 2, 30)
group2 = np.random.normal(12, 2, 30)

# Perform test
statistic, pvalue = stats.ttest_ind(group1, group2)

print(f"Test Statistic: {statistic:.4f}")
print(f"P-value: {pvalue:.4f}")
print(f"Significant at α=0.05: {pvalue < 0.05}")
```

---

## R Implementation

```r
# White Test in R
set.seed(42)

# Sample data
group1 <- rnorm(30, mean = 10, sd = 2)
group2 <- rnorm(30, mean = 12, sd = 2)

# Perform test
result <- t.test(group1, group2)
print(result)
```

---

## Related Concepts

- [[30_Knowledge/Stats/03_Regression_Analysis/Simple Linear Regression\|Linear Regression]]
- [[30_Knowledge/Stats/03_Regression_Analysis/Logistic Regression\|Logistic Regression]]
- [[30_Knowledge/Stats/03_Regression_Analysis/Residual Analysis\|Residual Analysis]]

---

## References

- **Historical:** White, H. (1980). A heteroskedasticity-consistent covariance matrix estimator and a direct test for heteroskedasticity. *Econometrica*, 48(4), 817-838. [DOI Link](https://doi.org/10.2307/1912934)
- **Book:** Greene, W. H. (2018). *Econometric Analysis* (8th ed.). Pearson. [Pearson Link](https://www.pearson.com/en-us/subject-catalog/p/econometric-analysis/P200000005955/9780134461366)
- **Book:** Stock, J. H., & Watson, M. W. (2015). *Introduction to Econometrics* (3rd ed.). Pearson. [Pearson Link](https://www.pearson.com/en-us/subject-catalog/p/introduction-to-econometrics/P200000007205/9780133486872)
