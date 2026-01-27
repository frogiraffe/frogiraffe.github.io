---
{"dg-publish":true,"permalink":"/stats/03-regression-analysis/outlier-analysis-standardized-residuals/","tags":["Diagnostics","Outliers"]}
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

- [[stats/03_Regression_Analysis/Cook's Distance\|Cook's Distance]] - Measuring *Influence* (Does the outlier matter?).
- [[stats/03_Regression_Analysis/Leverage (Hat Matrix)\|Leverage (Hat Matrix)]]

---

## References

- **Book:** Barnett, V., & Lewis, T. (1994). *Outliers in Statistical Data* (3rd ed.). Wiley. [Wiley Link](https://www.wiley.com/en-us/Outliers+in+Statistical+Data%2C+3rd+Edition-p-9780471930945)
- **Book:** Rousseeuw, P. J., & Leroy, A. M. (2005). *Robust Regression and Outlier Detection*. Wiley. [Wiley Link](https://doi.org/10.1002/0471725382)
- **Article:** Grubbs, F. E. (1969). Procedures for detecting outlying observations in samples. *Technometrics*, 11(1), 1-21. [DOI Link](https://doi.org/10.1080/00401706.1969.10487810)