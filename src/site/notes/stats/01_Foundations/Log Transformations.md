---
{"dg-publish":true,"permalink":"/stats/01-foundations/log-transformations/","tags":["Data-Preprocessing","Transformations"]}
---

## Overview

> [!abstract] Definition
> **Log Transformation** involves applying the natural logarithm ($\ln(x)$) or log-base-10 to a variable.
> *   Goal: Make skewed data more **Normal**.
> *   Goal: Stabilize **Variance** (Homoscedasticity).
> *   Goal: Linearize **Multiplicative** relationships.

> [!warning] Constraint
> You cannot take the log of 0 or negative numbers.
> *   Fix: Use $\ln(x + 1)$ or Box-Cox Transformation.

---

## 1. Interpretation

Using Log-Transformed Variables changes the interpretation of regression coefficients.

| Model | Equation | Interpretation of $\beta$ |
|-------|----------|---------------------------|
| **Log-Level** | $\ln(Y) = \beta X$ | 1 unit change in $X$ $\to$ $\beta \times 100$\% change in $Y$. |
| **Level-Log** | $Y = \beta \ln(X)$ | 1\% change in $X$ $\to$ $\beta / 100$ unit change in $Y$. |
| **Log-Log** | $\ln(Y) = \beta \ln(X)$ | 1\% change in $X$ $\to$ $\beta$\% change in $Y$ (Elasticity). |

---

## 2. Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Skewed Data
data = np.random.exponential(size=1000)

# Log Transform
# log1p calculates log(1+x) to handle zeros safely
log_data = np.log1p(data)

# Visual Check
fig, ax = plt.subplots(1, 2)
ax[0].hist(data, bins=30); ax[0].set_title("Original (Skewed)")
ax[1].hist(log_data, bins=30); ax[1].set_title("Log-Transformed (Normal-ish)")
plt.show()

# Box-Cox (Automated best power fit)
boxcox_data, lam = stats.boxcox(data + 0.001) # Data must be positive
print(f"Optimal Lambda: {lam:.3f}") # If lambda ~ 0, Log is best.
```

---

## 3. R Implementation

```r
# 1. Log Transform
df$log_income <- log(df$income)

# 2. Log(x+1)
df$log_income_safe <- log1p(df$income)

# 3. Box-Cox
library(MASS)
boxcox(lm(income ~ 1, data=df))
```

---

## 4. Related Concepts

- [[stats/01_Foundations/Normal Distribution\|Normal Distribution]]
- [[stats/03_Regression_Analysis/Simple Linear Regression\|Simple Linear Regression]]

---

## References

- **Historical:** Box, G. E. P., & Cox, D. R. (1964). An analysis of transformations. *Journal of the Royal Statistical Society: Series B*, 26(2), 211-252. [JSTOR](https://www.jstor.org/stable/2984418)
- **Book:** Tukey, J. W. (1977). *Exploratory Data Analysis*. Addison-Wesley. [WorldCat](https://www.worldcat.org/title/exploratory-data-analysis/oclc/3038677)
- **Article:** Bartlett, M. S. (1947). The use of transformations. *Biometrics*, 3(1), 39-52. [JSTOR](https://www.jstor.org/stable/3001536)
