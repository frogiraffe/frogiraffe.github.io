---
{"dg-publish":true,"permalink":"/stats/03-regression-analysis/cook-s-distance/","tags":["Diagnostics","Regression","Outliers","Influential-Points"]}
---

## Definition

> [!abstract] Core Statement
> **Cook's Distance (Cook's D)** is a measure of the ==influence== of a single data point on a regression model. It estimates how much the model's coefficients would change if that specific observation were removed.

---

## Purpose

1.  **Identify Influential Points:** Distinguish between harmless outliers and data points that drastically distort the model.
2.  **Clean Data:** Decide whether to keep, remove, or investigate specific observations.

---

## When to Use

> [!success] Use Cook's D When...
> - You have fitted a **Linear Regression** or GLM.
> - You suspect **outliers** might be dominating the results.
> - Diagnostic plots show points unrelated to the rest of the data.

> [!failure] Alternatives
> - **Leverage:** Measures how extreme $X$ is, but not if it affects the fit.
> - **Studentized Residuals:** Measures how far $Y$ is from prediction, but not its influence.
> - **Cook's D combines both.**

---

## Theoretical Background

### The Formula

$$
D_i = \frac{\sum_{j=1}^n (\hat{Y}_j - \hat{Y}_{j(i)})^2}{p \cdot MSE}
$$

- $\hat{Y}_j$: Prediction using all data.
- $\hat{Y}_{j(i)}$: Prediction using all data **except observation $i$**.
- $p$: Number of parameters.
- $MSE$: Mean Squared Error.

**Intuition:** $D_i$ is large if removing point $i$ causes the predicted values ($\hat{Y}$) to move a lot.

### Thresholds

- **Conservative:** $D_i > 1$ represents a highly influential point.
- **Sensitive:** $D_i > 4/n$ is a common cutoff for "worthy of investigation."

---

## Worked Numerical Example

> [!example] The Billionaire in the Neighborhood
> **Scenario:** Predicting Home Value from Sq Ft.
> **Data:** 20 homes.
> 
> **The Outlier:** A small historic mansion fit for a king.
> - **Sq Ft:** 2,000 (Average).
> - **Price:** $10,000,000 (100x average).
> 
> **Influence:**
> - **OLS with Outlier:** Slope = $2,000/sqft. (Distorted upwards).
> - **OLS without Outlier:** Slope = $200/sqft.
> 
> **Cook's D Result:**
> - $D_{mansion} = 8.5$ (Huge!).
> - **Action:** This single point is totally changing the model. Remove it or fit a robust model.

---

## Assumptions

- [ ] **Linear Model context.**
- [ ] **Valid OLS estimation.**

---

## Limitations

> [!warning] Pitfalls
> 1.  **Masking Effect:** Two outliers near each other can "hide" each other's influence.
> 2.  **Swamping:** A cluster of good points can make a valid extreme point look like an outlier.
> 3.  **Automatic Removal:** High Cook's D $\neq$ "Delete this data". It means "Investigate this data". It might be a data entry error, or it might be the most interesting discovery (e.g., a new phenomenon).

---

## Python Implementation

```python
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt

# Fit model
model = sm.OLS(y, X).fit()

# Calculate Cook's Distance
influence = model.get_influence()
cooks_d = influence.cooks_distance[0] # [0] is values, [1] is p-values

# Plot
plt.stem(np.arange(len(cooks_d)), cooks_d, markerfmt=",")
plt.title("Cook's Distance Plot")
plt.xlabel("Observation Index")
plt.ylabel("Cook's Distance")
plt.show()

# Identify Influential Points
n = len(X)
threshold = 4/n
influential_points = np.where(cooks_d > threshold)[0]
print(f"Influential Indices: {influential_points}")
```

---

## R Implementation

```r
# Fit Model
model <- lm(Price ~ SqFt, data = houses)

# Cook's Distance
cooks <- cooks.distance(model)

# Plot
plot(model, 4) # Plot 4 is Cook's D
abline(h = 4/nrow(houses), col="red")

# Extract Indices
which(cooks > 4/nrow(houses))
```

---

## Interpretation Guide

| Output | Interpretation | Action |
|--------|----------------|--------|
| $D_i = 0.01$ | Negligible influence. | Keep. |
| $D_i = 0.8$ | Moderate influence. | Inspect. Is it a valid data point? |
| $D_i = 5.2$ | Massive influence. | **Critical:** Model is unstable. Re-run model without this point to see impact. |

---

## Related Concepts

- [[stats/03_Regression_Analysis/Residual Analysis\|Residual Analysis]]
- [[stats/03_Regression_Analysis/Leverage (Hat Matrix)\|Leverage (Hat Matrix)]] - Potential for influence.
- [[stats/03_Regression_Analysis/Robust Regression\|Robust Regression]] - Alternative that downweights outliers (e.g., RANSAC, Huber).
- [[stats/03_Regression_Analysis/VIF (Variance Inflation Factor)\|VIF (Variance Inflation Factor)]]

---

## References

- **Historical:** Cook, R. D. (1977). Detection of influential observation in linear regression. *Technometrics*, 19(1), 15-18. [DOI Link](https://doi.org/10.1080/00401706.1977.10489493)
- **Book:** Belsley, D. A., Kuh, E., & Welsch, R. E. (2005). *Regression Diagnostics: Identifying Influential Data and Sources of Collinearity*. Wiley. [Wiley Link](https://doi.org/10.1002/0471725315)
- **Book:** Fox, J. (2019). *Regression Diagnostics: An Introduction* (2nd ed.). Sage. [Sage Link](https://us.sagepub.com/en-us/nam/regression-diagnostics/book265104)