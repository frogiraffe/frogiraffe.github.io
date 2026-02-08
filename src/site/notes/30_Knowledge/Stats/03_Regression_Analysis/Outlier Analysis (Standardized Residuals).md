---
{"dg-publish":true,"permalink":"/30-knowledge/stats/03-regression-analysis/outlier-analysis-standardized-residuals/","tags":["regression","modeling"]}
---

## Overview

> [!abstract] Definition
> **Outliers** are data points that deviate significantly from the rest of the observations.
> *   **Standardized Residuals:** The model errors divided by their standard deviation. This puts errors on a Z-score scale.

> [!important] Rule of Thumb
> *   $|Z| > 2$: Suspect. (Occurs 5% of time).
> *   $|Z| > 3$: **Outlier.** (Occurs 0.3% of time).

---

## 1. Python Implementation

```python
import numpy as np

# 1. Get Standardized Residuals
influence = model.get_influence()
standardized_residuals = influence.resid_studentized_internal

# 2. Filter Outliers (|Z| > 3)
outliers = np.where(np.abs(standardized_residuals) > 3)[0]
print(f"Outlier Indices: {outliers}")
```

---

## 2. R Implementation

```r
# 1. Get Standardized Residuals (rstandard)
res_std <- rstandard(model)

# 2. Identify
outliers <- which(abs(res_std) > 3)
print(outliers)

# 3. Plot
plot(model, 3) # Scale-Location plot helps see them
```

---

## 3. Related Concepts

- [[30_Knowledge/Stats/03_Regression_Analysis/Cook's Distance\|Cook's Distance]] - Measuring *Influence* (Does the outlier matter?).
- [[30_Knowledge/Stats/03_Regression_Analysis/Leverage (Hat Matrix)\|Leverage (Hat Matrix)]]

---

## Definition

> [!abstract] Core Statement
> **Outlier Analysis (Standardized Residuals)** ... Refer to standard documentation

---

> [!tip] Intuition (ELI5)
> Refer to standard documentation

---

## When to Use

> [!success] Use Outlier Analysis (Standardized Residuals) When...
> - Refer to standard documentation
> - Refer to standard documentation

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Assumptions are violated
> - Alternative methods are more appropriate

---

## Python Implementation

```python
import numpy as np
import pandas as pd

# Example implementation of Outlier Analysis (Standardized Residuals)
# See documentation for details

data = np.random.randn(100)
print(f"Mean: {np.mean(data):.3f}")
print(f"Std: {np.std(data):.3f}")
```

---

## R Implementation

```r
# Outlier Analysis (Standardized Residuals) in R
set.seed(42)

# Example implementation
data <- rnorm(100)
summary(data)
```

---

## Related Concepts

- [[30_Knowledge/Stats/03_Regression_Analysis/Simple Linear Regression\|Linear Regression]]
- [[30_Knowledge/Stats/03_Regression_Analysis/Logistic Regression\|Logistic Regression]]
- [[30_Knowledge/Stats/03_Regression_Analysis/Residual Analysis\|Residual Analysis]]

---

## References

- **Book:** Barnett, V., & Lewis, T. (1994). *Outliers in Statistical Data* (3rd ed.). Wiley. [Wiley Link](https://www.wiley.com/en-us/Outliers+in+Statistical+Data%2C+3rd+Edition-p-9780471930945)
- **Book:** Rousseeuw, P. J., & Leroy, A. M. (2005). *Robust Regression and Outlier Detection*. Wiley. [Wiley Link](https://doi.org/10.1002/0471725382)
- **Article:** Grubbs, F. E. (1969). Procedures for detecting outlying observations in samples. *Technometrics*, 11(1), 1-21. [DOI Link](https://doi.org/10.1080/00401706.1969.10487810)