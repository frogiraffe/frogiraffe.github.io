---
{"dg-publish":true,"permalink":"/stats/03-regression-analysis/leverage-hat-matrix/","tags":["Diagnostics","Regression","Influential-Points"]}
---


# Leverage (Hat Matrix)

## Overview

> [!abstract] Definition
> **Leverage** ($h_{ii}$) measures how far an observation's independent variable values are from the mean of those values. High-leverage points are outliers in the **X-space** and have the potential to exert significant influence on the regression coefficients.

![Leverage and Influence](/img/user/stats/images/leverage_influence_1769306417401.png)

---

## 1. Mathematical Derivation

In OLS regression ($Y = X\beta + \varepsilon$), the fitted values $\hat{Y}$ are obtained by projecting $Y$ onto the space spanned by $X$:

$$ \hat{Y} = X(X^TX)^{-1}X^T Y = HY $$

The $n \times n$ matrix $H = X(X^TX)^{-1}X^T$ is called the **Hat Matrix** because it puts the "hat" on $Y$.
The diagonal elements $h_{ii}$ are the leverage values.

---

## 2. Properties

- Bounds: $1/n \leq h_{ii} \leq 1$.
- Sum: $\sum h_{ii} = p$ (number of parameters including intercept).
- Mean Leverage: $\bar{h} = p/n$.

---

## 3. Identification Thresholds

A common rule of thumb identifies a point as high leverage if:

$$ h_{ii} > 2 \times \bar{h} = \frac{2p}{n} $$

Points exceeding $3p/n$ are considered extremely high leverage.

---

## 4. Worked Example: The Outlier CEO

> [!example] Problem
> You are modeling **Income vs Age** for a small town.
> - Most people ($n=20$) are aged 20-60, earning \$30k-\$100k.
> - **Person X (CEO):** Age = **95**, Income = \$50,000.
> 
> **Question:** Does Person X have high leverage? Is it influential?

**Analysis:**

1.  **Check X-Space (Age):**
    -   Mean Age $\approx 40$.
    -   Person X Age = 95. This is far from the centroid.
    -   **Result:** High Leverage ($h_{ii}$ will be large).

2.  **Check Y-Space (Residual):**
    -   If the model predicts Income for a 95-year-old is roughly \$40k-\$60k (retirement), and actual is \$50k...
    -   **Residual is small.**

3.  **Conclusion:**
    -   Person X is a **High Leverage** point (extreme Age).
    -   However, because the income fits the trend, it is **Low Influence**. It merely anchors the regression line, reducing standard errors (Good Leverage).
    -   *Contrast:* If the CEO earned \$10M, they would be High Leverage AND High Influence (pulling the slope up).

---

## 5. Assumptions

- [ ] **Linearity:** Leverage assumes the relationship is linear.
- [ ] **Correct Specification:** If the model needs $X^2$ and you only use $X$, leverage values might be misleading about "extreme" points.

---

## 6. Limitations

> [!warning] Pitfalls
> 1.  **Good vs Bad Leverage:** Don't delete points just because they have high leverage! If they follow the trend, they are valuable data points that increase precision. Only remove if they are *errors* or fundamentally different populations.
> 2.  **The "Masking" Effect:** Two high-leverage points close to each other can mask each other's influence.
> 3.  **Data Entry Errors:** High leverage often flags typos (e.g., Age=950 instead of 95). Always check source data.

---

## 7. Leverage vs. Influence

High leverage is a **necessary but insufficient** condition for high influence.
- **Good Leverage:** A point follows the trend of the rest of the data but is extreme in X. It reduces the standard error of estimates.
- **Bad Leverage:** A point is extreme in X and deviates from the trend (large residual). This pulls the regression line towards itself.

See [[stats/03_Regression_Analysis/Cook's Distance\|Cook's Distance]] for the combined measure of Influence.

## Interpretation Guide

| Metric | Rule of Thumb | Action |
|--------|---------------|--------|
| **$h_{ii} > 2p/n$** | Moderate Leverage | Investigate. Check for data entry errors. |
| **$h_{ii} > 3p/n$** | Result is High Leverage | **Danger zone.** Check Cook's Distance to see if it's influential. |
| **$1/n$** | Minimum possible leverage | Perfectly average observation X-wise. |
| **1.0** | Maximum possible leverage | Parameter is determined solely by this point (DF used up). |

---

## 8. Python Implementation Example

```python
import numpy as np
import statsmodels.api as sm

# Fit Model
model = sm.OLS(y, X).fit()

# Get Influence
influence = model.get_influence()
leverage = influence.hat_matrix_diag

# Threshold
p = len(model.params)
n = len(y)
threshold = 2 * p / n

high_leverage_points = np.where(leverage > threshold)[0]
print(f"High Leverage Indices: {high_leverage_points}")
```

---

## 6. Related Concepts

- [[stats/03_Regression_Analysis/Cook's Distance\|Cook's Distance]] - Measure of influence combining leverage and residuals.
- [[stats/03_Regression_Analysis/Outlier Analysis (Standardized Residuals)\|Outlier Analysis (Standardized Residuals)]] - Outliers in Y-space.
- [[stats/03_Regression_Analysis/Multiple Linear Regression\|Multiple Linear Regression]] - Framework.
